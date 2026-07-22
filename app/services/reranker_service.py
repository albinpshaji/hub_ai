import asyncio
import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder
from app.config import settings

logger = logging.getLogger(__name__)

_model: CrossEncoder | None = None
_model_device: str | None = None

def get_reranker_model(device: str | None = None) -> CrossEncoder:
    """Lazy loads and caches the CrossEncoder model in memory."""
    global _model, _model_device
    target_device = device if device is not None else settings.reranker_device
    
    if _model is None or _model_device != target_device:
        model_name = settings.reranker_model
        logger.info(f"Initializing Cross-Encoder model {model_name} on device {target_device}...")
        try:
            _model = CrossEncoder(model_name, device=target_device)
            _model_device = target_device
            logger.info(f"Cross-Encoder model {model_name} loaded successfully on {target_device}.")
        except Exception as exc:
            if target_device != "cpu":
                logger.warning(
                    f"Failed to load Cross-Encoder on device {target_device}: {exc}. "
                    f"Attempting automatic fallback to 'cpu'..."
                )
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                _model = CrossEncoder(model_name, device="cpu")
                _model_device = "cpu"
                logger.info(f"Cross-Encoder model {model_name} loaded successfully on cpu (fallback).")
            else:
                raise exc
    return _model


def _predict_sync(query: str, passages: List[str]) -> List[float]:
    """Synchronous CPU/GPU bound prediction run with fallback to CPU on error."""
    global _model, _model_device
    pairs = [(query, passage) for passage in passages]
    
    try:
        model = get_reranker_model()
        scores = model.predict(pairs)
        return [float(score) for score in scores]
    except Exception as exc:
        logger.warning(
            f"Reranker prediction failed on device {_model_device}: {exc}. "
            f"Attempting CPU fallback for prediction..."
        )
        try:
            _model = None
            _model_device = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            model = get_reranker_model(device="cpu")
            scores = model.predict(pairs)
            return [float(score) for score in scores]
        except Exception as cpu_exc:
            logger.error(f"CPU fallback for reranker prediction failed: {cpu_exc}")
            raise cpu_exc


async def rerank_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    """
    Rerank candidate passages using BAAI/bge-reranker-base.
    Runs in a background thread to prevent blocking the async loop.
    """
    if not candidates:
        return []

    passages = [c["text"] for c in candidates]
    
    # Offload CPU/GPU execution to thread pool
    scores = await asyncio.to_thread(_predict_sync, query, passages)
    
    reranked = []
    for idx, candidate in enumerate(candidates):
        item = dict(candidate)
        item["rerank_score"] = scores[idx]
        reranked.append(item)
        
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked
