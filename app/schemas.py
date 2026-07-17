from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid


class ChatStreamRequest(BaseModel):
    messages: List[Dict[str, Any]]
    think: bool = True


class SummarizeRequest(BaseModel):
    text: str


class EmbeddingRequest(BaseModel):
    text: str


class IngestRequest(BaseModel):
    user_id: uuid.UUID
    document_id: uuid.UUID
    text: str
    filename: str = ""
    session_id: Optional[uuid.UUID] = None


class ChatWithToolsRequest(BaseModel):
    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]] = None
    think: bool = True


class IngestImagesRequest(BaseModel):
    user_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    image_metadata: List[Dict[str, Any]]
    session_id: Optional[uuid.UUID] = None


class SearchRequest(BaseModel):
    user_id: uuid.UUID
    query: str
    limit: int = 4
    retrieval_mode: str = "semantic"
    use_hyde: bool = False
    allowed_document_ids: Optional[List[uuid.UUID]] = None
    session_id: Optional[uuid.UUID] = None
    selected_document_ids: Optional[List[uuid.UUID]] = None
    use_reranker: bool = False
    include_meta: bool = False


class ExtractRequest(BaseModel):
    file_path: str
    file_type: str


class WebSearchRequest(BaseModel):
    query: str
    max_results: int = 5


class ExtractVisualsRequest(BaseModel):
    pdf_path: str


class ReinspectPageRequest(BaseModel):
    pdf_path: str
    page_number: int
    specific_question: str


class VisionQARequest(BaseModel):
    base64_image: str
    question: str
