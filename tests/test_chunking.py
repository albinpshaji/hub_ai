import pytest
import pytest_asyncio
import uuid

from app.config import settings
from app.services import vector_service
from app.services.vector_service import chunk_text

def test_chunk_text_small_paragraphs():
    """
    Verifies that multiple small paragraphs are grouped together into chunks
    until the chunk size limit is reached.
    """
    text = (
        "Paragraph one is here.\n\n"
        "Paragraph two is here.\n\n"
        "Paragraph three is here."
    )
    # chunk_size of 55 allows Paragraph 1 & 2 to merge (~46 chars), while Paragraph 3 is separate
    chunks = chunk_text(text, chunk_size=55, overlap=10)
    assert len(chunks) == 2
    assert chunks[0] == "Paragraph one is here.\n\nParagraph two is here."
    assert "Paragraph three is here." in chunks[1]

def test_chunk_text_large_paragraph_sentence_split():
    """
    Verifies that an individual paragraph exceeding chunk_size is split
    on sentence boundaries.
    """
    text = (
        "This is sentence one. "
        "This is sentence two. "
        "This is sentence three. "
        "This is sentence four."
    )
    # chunk_size=50 restricts each chunk to ~2 sentences, splitting on punctuation
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 50
    # Sentences should be complete
    assert chunks[0].strip() == "This is sentence one. This is sentence two."
    
def test_chunk_text_very_long_sentence_word_split():
    """
    Verifies that a single sentence exceeding chunk_size is split
    at word boundaries rather than characters.
    """
    # A single sentence with 10 words, each word 10 letters -> 110 characters
    long_sentence = " ".join(["wordtenlet"] * 10)
    assert len(long_sentence) == 109
    
    # chunk_size=40 means it must split by words
    chunks = chunk_text(long_sentence, chunk_size=40, overlap=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 40
        # Ensure it didn't cut inside a word
        words = chunk.split()
        for w in words:
            assert w == "wordtenlet"

def test_chunk_text_respects_overlap():
    """
    Verifies that consecutive chunks carry overlap forward correctly.
    """
    text = (
        "This is first part of the long text. "
        "This is second part of the long text."
    )
    chunks = chunk_text(text, chunk_size=45, overlap=15)
    assert len(chunks) == 2
    
    # Chunk 1 should start with the aligned overlap segment: "the long text."
    assert chunks[1].startswith("the long text.")


@pytest.mark.asyncio
async def test_search_relevant_chunks_semantic_uses_hyde_query(monkeypatch):
    captured = {}

    async def fake_generate(query):
        captured["hyde_input"] = query
        return "hypothetical answer about refresh token rotation"

    async def fake_semantic_candidates(user_id, query, limit, allowed_document_ids):
        captured["semantic_query"] = query
        return [
            (
                {
                    "text": "Refresh token rotation revokes old tokens.",
                    "filename": "auth.txt",
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                },
                0.9,
            )
        ]

    monkeypatch.setattr(vector_service, "generate_hypothetical_document", fake_generate)
    monkeypatch.setattr(vector_service, "_semantic_candidates", fake_semantic_candidates)

    results = await vector_service.search_relevant_chunks(
        user_id=uuid.uuid4(),
        query="How does refresh work?",
        use_hyde=True,
        retrieval_mode="semantic",
    )

    assert captured["hyde_input"] == "How does refresh work?"
    assert captured["semantic_query"] == "hypothetical answer about refresh token rotation"
    assert results[0]["filename"] == "auth.txt"


@pytest.mark.asyncio
async def test_search_relevant_chunks_hybrid_uses_hyde_only_for_semantic(monkeypatch):
    captured = {}

    async def fake_generate(query):
        return "hypothetical document about banana invoices"

    async def fake_semantic_candidates(user_id, query, limit, allowed_document_ids):
        captured["semantic_query"] = query
        return [
            (
                {
                    "text": "A banana invoice can be reconciled.",
                    "filename": "semantic.txt",
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                },
                0.7,
            )
        ]

    async def fake_keyword_candidates(user_id, query, keywords, limit, allowed_document_ids):
        captured["keyword_query"] = query
        captured["keywords"] = keywords
        return [
            (
                {
                    "text": "The original query terms are still searchable.",
                    "filename": "keyword.txt",
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                },
                0.8,
            )
        ]

    monkeypatch.setattr(vector_service, "generate_hypothetical_document", fake_generate)
    monkeypatch.setattr(vector_service, "_semantic_candidates", fake_semantic_candidates)
    monkeypatch.setattr(vector_service, "_keyword_candidates", fake_keyword_candidates)

    await vector_service.search_relevant_chunks(
        user_id=uuid.uuid4(),
        query="banana invoice lookup",
        use_hyde=True,
        retrieval_mode="hybrid",
    )

    assert captured["semantic_query"] == "hypothetical document about banana invoices"
    assert captured["keyword_query"] == "banana invoice lookup"
    assert "banana" in captured["keywords"]


@pytest.mark.asyncio
async def test_search_relevant_chunks_keyword_mode_skips_hyde(monkeypatch):
    captured = {}

    async def fail_generate(query):
        raise AssertionError("HyDE should not run for keyword retrieval")

    async def fake_keyword_candidates(user_id, query, keywords, limit, allowed_document_ids):
        captured["keyword_query"] = query
        return [
            (
                {
                    "text": "Keyword retrieval uses the original query.",
                    "filename": "keyword.txt",
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                },
                0.8,
            )
        ]

    monkeypatch.setattr(vector_service, "generate_hypothetical_document", fail_generate)
    monkeypatch.setattr(vector_service, "_keyword_candidates", fake_keyword_candidates)

    results = await vector_service.search_relevant_chunks(
        user_id=uuid.uuid4(),
        query="exact phrase",
        use_hyde=True,
        retrieval_mode="keyword",
    )

    assert captured["keyword_query"] == "exact phrase"
    assert results[0]["match_type"] == "keyword"


@pytest.mark.asyncio
async def test_search_relevant_chunks_hyde_failure_falls_back_to_original_query(monkeypatch):
    captured = {}

    async def fail_generate(query):
        raise RuntimeError("ollama unavailable")

    async def fake_semantic_candidates(user_id, query, limit, allowed_document_ids):
        captured["semantic_query"] = query
        return [
            (
                {
                    "text": "Fallback retrieval still works.",
                    "filename": "fallback.txt",
                    "document_id": str(uuid.uuid4()),
                    "chunk_index": 0,
                },
                0.6,
            )
        ]

    monkeypatch.setattr(vector_service, "generate_hypothetical_document", fail_generate)
    monkeypatch.setattr(vector_service, "_semantic_candidates", fake_semantic_candidates)

    results = await vector_service.search_relevant_chunks(
        user_id=uuid.uuid4(),
        query="original fallback query",
        use_hyde=True,
        retrieval_mode="semantic",
    )

    assert captured["semantic_query"] == "original fallback query"
    assert results[0]["filename"] == "fallback.txt"


from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, TextIndexParams, TokenizerType, PayloadSchemaType

@pytest_asyncio.fixture
async def setup_qdrant_test_collection():
    original_collection = settings.qdrant_collection
    test_collection_name = f"temp_test_collection_{uuid.uuid4().hex}"
    settings.qdrant_collection = test_collection_name

    client = AsyncQdrantClient(url=settings.qdrant_url)

    if await client.collection_exists(test_collection_name):
        await client.delete_collection(test_collection_name)

    await client.create_collection(
        collection_name=test_collection_name,
        vectors_config=VectorParams(
            size=768,  # nomic-embed-text dimensions
            distance=Distance.COSINE
        )
    )

    await client.create_payload_index(
        collection_name=test_collection_name,
        field_name="text",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            lowercase=True
        )
    )

    await client.create_payload_index(
        collection_name=test_collection_name,
        field_name="filename",
        field_schema=TextIndexParams(
            type="text",
            tokenizer=TokenizerType.MULTILINGUAL,
            lowercase=True
        )
    )

    yield

    if await client.collection_exists(test_collection_name):
        await client.delete_collection(test_collection_name)
    settings.qdrant_collection = original_collection


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_relevant_chunks_prioritization(setup_qdrant_test_collection):
    from app.services.vector_service import store_document_vectors, search_relevant_chunks
    user_id = uuid.uuid4()
    session_id = uuid.uuid4()
    
    global_doc_id = uuid.uuid4()
    await store_document_vectors(
        user_id=user_id,
        document_id=global_doc_id,
        text="This contains some general info.",
        filename="global_priority.txt",
        session_id=None,
    )
    
    session_doc_id = uuid.uuid4()
    await store_document_vectors(
        user_id=user_id,
        document_id=session_doc_id,
        text="This contains some general info.",
        filename="session_priority.txt",
        session_id=session_id,
    )
    
    results = await search_relevant_chunks(
        user_id=user_id,
        query="general info",
        limit=2,
        allowed_document_ids=[global_doc_id, session_doc_id],
        session_id=session_id,
    )
    
    assert len(results) == 2
    # The session document must rank higher than the global one due to the +0.15 boost
    assert results[0]["filename"] == "session_priority.txt"
    assert results[1]["filename"] == "global_priority.txt"


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_relevant_chunks_filename_prioritization(setup_qdrant_test_collection):
    from app.services.vector_service import store_document_vectors, search_relevant_chunks
    user_id = uuid.uuid4()
    
    random_doc_id = uuid.uuid4()
    await store_document_vectors(
        user_id=user_id,
        document_id=random_doc_id,
        text="The quick brown fox jumps over the lazy dog.",
        filename="random_notes.txt",
        session_id=None,
    )
    
    resume_doc_id = uuid.uuid4()
    await store_document_vectors(
        user_id=user_id,
        document_id=resume_doc_id,
        text="The quick brown fox jumps over the lazy dog.",
        filename="resume_johndoe.pdf",
        session_id=None,
    )
    
    results = await search_relevant_chunks(
        user_id=user_id,
        query="explain my resume",
        limit=2,
        allowed_document_ids=[random_doc_id, resume_doc_id],
        session_id=None,
    )
    
    assert len(results) == 2
    assert results[0]["filename"] == "resume_johndoe.pdf"
    assert results[1]["filename"] == "random_notes.txt"


@pytest.mark.live
@pytest.mark.asyncio
async def test_search_relevant_chunks_selected_similarity_boost(setup_qdrant_test_collection):
    from app.services.vector_service import store_document_vectors, search_relevant_chunks
    user_id = uuid.uuid4()
    
    # Create two documents. One will be selected, the other not.
    selected_doc_id = uuid.uuid4()
    other_doc_id = uuid.uuid4()
    
    # Store selected doc (has matching filename keyword)
    await store_document_vectors(
        user_id=user_id,
        document_id=selected_doc_id,
        text="This contains some general info.",
        filename="selected_invoice.pdf",
    )
    
    # Store other doc (also has matching filename keyword, but not selected)
    await store_document_vectors(
        user_id=user_id,
        document_id=other_doc_id,
        text="This contains some general info.",
        filename="other_invoice.pdf",
    )
    
    # 1. Test filename match boost (+0.40) on selected doc
    # Both are similar to the query "invoice info" by filename.
    # But only selected_doc_id is in selected_document_ids.
    # Therefore, selected_doc_id should get the 0.40 boost and rank first.
    results = await search_relevant_chunks(
        user_id=user_id,
        query="invoice info",
        limit=2,
        allowed_document_ids=[selected_doc_id, other_doc_id],
        selected_document_ids=[selected_doc_id],
    )
    
    assert len(results) == 2
    assert results[0]["filename"] == "selected_invoice.pdf"
    
    # 2. Test content match boost (+0.40) on selected doc
    # Let's create two docs where the filename does NOT match, but the text content matches query keyword "banana".
    selected_banana_doc_id = uuid.uuid4()
    other_banana_doc_id = uuid.uuid4()
    
    await store_document_vectors(
        user_id=user_id,
        document_id=selected_banana_doc_id,
        text="The quick yellow banana is sweet.",
        filename="first.txt",
    )
    
    await store_document_vectors(
        user_id=user_id,
        document_id=other_banana_doc_id,
        text="The quick yellow banana is sweet.",
        filename="second.txt",
    )
    
    results_banana = await search_relevant_chunks(
        user_id=user_id,
        query="banana fruit",
        limit=2,
        allowed_document_ids=[selected_banana_doc_id, other_banana_doc_id],
        selected_document_ids=[selected_banana_doc_id],
    )
    
    assert len(results_banana) == 2
    # The selected document should be first since it got +0.40 boost
    assert results_banana[0]["filename"] == "first.txt"
