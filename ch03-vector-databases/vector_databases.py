"""
Vector Databases and Agent Memory

Code listings from Chapter 03, Book 3:
"Scaling Autonomous AI Agents: Engineering for Production at Real-World Scale"
by Dr. Vijay Raghavan

This file faithfully reproduces every code listing from the chapter, in book
order, with section banners showing the block number. Most listings are
runnable Python that builds incrementally; some are illustrative fragments
(log output, file trees, Dockerfile snippets, JSON examples) preserved as
docstrings so this file always remains valid Python.

To use a particular class or function, copy it into your own project and
provide the surrounding context (imports, dependencies) as needed.
"""

import logging
import os
import random
import threading
import time

_TRANSIENT_LLM_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)
try:  # Optional provider SDK; keep this file importable without it.
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
except ImportError:  # pragma: no cover - dependency not required for examples
    pass
else:
    _TRANSIENT_LLM_EXCEPTIONS = _TRANSIENT_LLM_EXCEPTIONS + (
        APIConnectionError,
        APIError,
        APITimeoutError,
        RateLimitError,
    )
_RETRIEVER_OPERATION_EXCEPTIONS = _TRANSIENT_LLM_EXCEPTIONS + (
    RuntimeError,
    ValueError,
    TypeError,
)


# ============================================================================
# Block 1 (chapter listing #1)
# ============================================================================
#
# These first blocks call into third-party SDKs (Pinecone, Weaviate, Qdrant,
# pgvector) and reference placeholder vectors. Real callers should supply an
# embedding model output; for syntactic completeness we define stand-in
# vectors here so the module is importable without external services.
# 1536 matches OpenAI ``text-embedding-3-small``; replace the constant
# below with whatever your embedding model produces (768 for many MiniLM
# variants, 3072 for ``text-embedding-3-large``, etc.).
_PLACEHOLDER_EMBEDDING_DIM = 1536
embedding_vector = [0.0] * _PLACEHOLDER_EMBEDDING_DIM  # placeholder: replace with real embedding
query_embedding = [0.0] * _PLACEHOLDER_EMBEDDING_DIM  # placeholder: replace with real embedding

# Guarded so import-time code does not require the Pinecone SDK,
# credentials, or a live network. Reading the block teaches the API;
# to run it, install ``pinecone-client`` and replace api_key.
if False:  # pragma: no cover -- illustrative; install SDK + set api_key
    import os
    import pinecone
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    # Create index for 1536-dimension OpenAI embeddings
    pc.create_index(
        name="knowledge-base",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

    index = pc.Index("knowledge-base")

    index.upsert(
        vectors=[
            {
                "id": "doc-001",
                "values": embedding_vector,  # 1536-dim list
                "metadata": {
                    "source": "internal-docs",
                    "category": "authentication",
                    "last_updated": "2025-03-15",
                },
            }
        ],
        namespace="production",
    )

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_metadata=True,
        filter={
            "category": {"$eq": "authentication"},
            "last_updated": {"$gte": "2025-01-01"},
        },
        namespace="production",
    )

# ============================================================================
# Block 2 (chapter listing #2)
# ============================================================================

# Guarded so importing this module does not require the Weaviate SDK
# or a running instance. Install ``weaviate-client`` and remove the
# ``if False`` to run.
if False:  # pragma: no cover -- illustrative; requires local Weaviate
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery

    client = weaviate.connect_to_local()

    client.collections.create(
        name="Document",
        vectorizer_config=Configure.Vectorizer.text2vec_openai(
            model="text-embedding-3-small"
        ),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="title", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="source_url", data_type=DataType.TEXT),
        ],
    )

    documents = client.collections.get("Document")

    # Add documents (vectorization happens automatically)
    documents.data.insert(
        {
            "content": "To reset your password, navigate to Settings > Security...",
            "title": "Password Reset Guide",
            "category": "authentication",
            "source_url": "https://docs.example.com/auth/password-reset",
        }
    )

    results = documents.query.hybrid(
        query="forgot my password",
        alpha=0.7,  # Weight toward vector search (0=keyword, 1=vector)
        limit=5,
        return_metadata=MetadataQuery(score=True),
    )

    for result in results.objects:
        print(f"Title: {result.properties['title']}")
        print(f"Score: {result.metadata.score}")

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

# Guarded: importing this module does not require the Qdrant SDK or
# a running instance. Install ``qdrant-client`` and remove the
# ``if False`` to run.
if False:  # pragma: no cover -- illustrative; requires local Qdrant
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        OptimizersConfigDiff,
        HnswConfigDiff,
    )

    client = QdrantClient(host="localhost", port=6333)

    client.create_collection(
        collection_name="knowledge_base",
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
            on_disk=True,  # Memory-map vectors for large datasets
        ),
        hnsw_config=HnswConfigDiff(
            m=16,  # Number of connections per layer
            ef_construct=100,  # Search depth during indexing
            full_scan_threshold=10000,
        ),
        optimizers_config=OptimizersConfigDiff(indexing_threshold=20000),
    )

    client.upsert(
        collection_name="knowledge_base",
        points=[
            PointStruct(
                id=1,
                vector=embedding_vector,
                payload={
                    "content": "Authentication uses OAuth 2.0...",
                    "category": "security",
                    "access_level": "internal",
                    "version": "2.1",
                },
            )
        ],
    )

    results = client.search(
        collection_name="knowledge_base",
        query_vector=query_embedding,
        limit=5,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="access_level", match=MatchValue(value="internal")
                )
            ]
        ),
        with_payload=True,
    )

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

# Guarded: importing this module does not require the psycopg2 SDK or
# a running Postgres. Install ``psycopg2-binary`` and ``pgvector`` and
# substitute real credentials from your secrets store to run.
if False:  # pragma: no cover -- illustrative; requires pgvector-enabled Postgres
    import psycopg2
    from pgvector.psycopg2 import register_vector

    conn = psycopg2.connect(
        host="localhost",
        database="knowledge_db",
        user="postgres",
        password="password",
    )
    register_vector(conn)

    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(1536),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create HNSW index for fast similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
        """)

        conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (content, embedding, metadata)
            VALUES (%s, %s, %s)
            """,
            (
                "Password reset requires email verification...",
                embedding_vector,
                '{"category": "auth", "source": "wiki"}',
            ),
        )
        conn.commit()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, metadata,
                   1 - (embedding <=> %s::vector) as similarity
            FROM documents
            WHERE metadata->>'category' = 'auth'
            ORDER BY embedding <=> %s::vector
            LIMIT 5
            """,
            (query_embedding, query_embedding),
        )
        results = cur.fetchall()

# ============================================================================
# Block 5 (chapter listing #5)
# ============================================================================

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass


_OPENAI_EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_OPENAI_CONFIGURABLE_DIMENSION_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
}


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.25,
        embed_batch_size: int = 100,
        max_texts_per_call: int = 10_000,
        dimension: Optional[int] = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be >= 0")
        if embed_batch_size < 1:
            raise ValueError("embed_batch_size must be >= 1")
        if max_texts_per_call < 1:
            raise ValueError("max_texts_per_call must be >= 1")
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be >= 1")
        model_dimension = _OPENAI_EMBEDDING_DIMENSIONS.get(model)
        if dimension is None:
            if model_dimension is None:
                raise ValueError(
                    "dimension is required for unknown embedding model"
                )
            dimension = model_dimension
        elif model_dimension is not None:
            if dimension > model_dimension:
                raise ValueError(
                    f"dimension must be <= {model_dimension} for {model}"
                )
            if (
                dimension != model_dimension
                and model not in _OPENAI_CONFIGURABLE_DIMENSION_MODELS
            ):
                raise ValueError(f"{model} does not support custom dimensions")
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )
        self.model = model
        self._dimension = dimension
        self._request_dimension = (
            dimension
            if model in _OPENAI_CONFIGURABLE_DIMENSION_MODELS
            and dimension != model_dimension
            else None
        )
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._embed_batch_size = embed_batch_size
        self._max_texts_per_call = max_texts_per_call

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        response = self._embedding_request(text)
        return response.data[0].embedding

    def _embedding_request(self, payload: Any) -> Any:
        last_error: Optional[BaseException] = None
        attempts = max(1, self._max_retries + 1)
        for attempt in range(attempts):
            try:
                kwargs = {"model": self.model, "input": payload}
                if self._request_dimension is not None:
                    kwargs["dimensions"] = self._request_dimension
                return self.client.embeddings.create(
                    **kwargs
                )
            except _TRANSIENT_LLM_EXCEPTIONS as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                time.sleep(self._retry_backoff * (2**attempt))
        raise RuntimeError("embedding request failed after retries") from last_error

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if len(texts) > self._max_texts_per_call:
            raise ValueError(
                f"embed_batch received {len(texts)} texts; "
                f"max_texts_per_call is {self._max_texts_per_call}"
            )
        out: List[List[float]] = []
        for i in range(0, len(texts), self._embed_batch_size):
            chunk = texts[i : i + self._embed_batch_size]
            response = self._embedding_request(chunk)
            if len(response.data) != len(chunk):
                raise ValueError(
                    f"embedding provider returned {len(response.data)} "
                    f"embeddings for {len(chunk)} input texts"
                )
            out.extend(item.embedding for item in response.data)
        return out


class SentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding using Sentence Transformers.

    Loads the model in ``__init__``; for ``BAAI/bge-large-en-v1.5`` this
    is roughly 1.3 GiB of weights resident per process. Run one
    instance per worker pool, not one per request. Pass ``device='cpu'``
    on CPU-only hosts, ``'cuda'``/``'cuda:0'``/etc. on GPU hosts, or
    ``'mps'`` on Apple silicon; on shared-GPU machines pin to a
    specific index to avoid contention with sibling workers.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: Optional[str] = None,
        batch_size: int = 32,
        lazy_load: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._device = device
        self._SentenceTransformer = SentenceTransformer
        self.batch_size = batch_size
        if lazy_load:
            # Defer the ~1.3 GiB weight load to first embed_* call so
            # worker startup is fast and unused replicas stay slim.
            self._model: Optional[Any] = None
            self._dimension = 0
        else:
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._SentenceTransformer(
                self._model_name, device=self._device
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    @property
    def dimension(self) -> int:
        if self._model is None:
            # Force load so callers observing the dimension before the
            # first embed call still see the model-defined value.
            _ = self.model
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return embeddings.tolist()

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

class InstructorEmbedding(EmbeddingModel):
    """Instructor embedding with task-specific prefixes.

    This is a pure-local model: the INSTRUCTOR weights are loaded into
    process on construction and ``encode`` runs entirely on CPU/GPU
    with no outbound network I/O. Unlike :class:`OpenAIEmbedding`, the
    class therefore does not expose ``timeout_seconds`` /
    ``max_retries`` / ``retry_backoff_seconds`` knobs because there is
    no remote call path to time out or retry.
    """

    def __init__(self, model_name: str = "hkunlp/instructor-large") -> None:
        from InstructorEmbedding import INSTRUCTOR

        self.model = INSTRUCTOR(model_name)
        self._dimension = 768

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(
        self,
        text: str,
        instruction: str = "Represent the document for retrieval:",
    ) -> List[float]:
        embedding = self.model.encode([[instruction, text]])
        return embedding[0].tolist()

    def embed_query(self, query: str) -> List[float]:
        instruction = "Represent the question for retrieving documents:"
        return self.embed_text(query, instruction)

    def embed_document(self, document: str) -> List[float]:
        instruction = "Represent the document for retrieval:"
        return self.embed_text(document, instruction)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        instruction = "Represent the document for retrieval:"
        embeddings = self.model.encode([[instruction, text] for text in texts])
        return [embedding.tolist() for embedding in embeddings]

# ============================================================================
# Block 7 (chapter listing #7)
# ============================================================================

def truncate_embedding(
    embedding: List[float], target_dim: int
) -> List[float]:
    """Truncate Matryoshka embedding to target dimension."""
    if len(embedding) < target_dim:
        raise ValueError(
            f"Cannot expand embedding from {len(embedding)} to {target_dim}"
        )

    truncated = embedding[:target_dim]
    # Re-normalize after truncation
    norm = np.linalg.norm(truncated)
    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding vector")
    return (np.array(truncated) / norm).tolist()

# ============================================================================
# Block 8 (chapter listing #8)
# ============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
from enum import Enum


class ChunkingStrategy(Enum):
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""

    content: str
    chunk_id: str
    document_id: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())


@dataclass
class Document:
    """Source document for chunking."""

    content: str
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentChunker(ABC):
    """Abstract base class for document chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split document into chunks."""
        pass


class FixedSizeChunker(DocumentChunker):
    """Chunk documents by fixed character count with overlap."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_document_chars: int = 2_000_000,
        max_chunks: int = 10_000,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be strictly less than chunk_size "
                f"(got chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
            )
        if max_document_chars < 1:
            raise ValueError("max_document_chars must be >= 1")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_document_chars = max_document_chars
        self.max_chunks = max_chunks

    def chunk(self, document: Document) -> List[DocumentChunk]:
        chunks = []
        text = document.content
        if len(text) > self.max_document_chars:
            raise ValueError(
                f"document has {len(text)} chars; "
                f"max_document_chars is {self.max_document_chars}"
            )
        start = 0
        chunk_index = 0

        while start < len(text):
            if chunk_index >= self.max_chunks:
                raise ValueError(
                    f"document would exceed max_chunks={self.max_chunks}"
                )
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(" ")
                if last_space > self.chunk_size * 0.8:
                    end = start + last_space
                    chunk_text = text[start:end]

            chunks.append(
                DocumentChunk(
                    content=chunk_text.strip(),
                    chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                    document_id=document.document_id,
                    start_char=start,
                    end_char=end,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "fixed_size",
                    },
                )
            )

            next_start = end - self.chunk_overlap
            # Guarantee forward progress: if the word-boundary adjustment
            # made ``end`` small enough that ``next_start <= start``, advance
            # by at least one character to avoid an infinite loop.
            if next_start <= start:
                next_start = start + 1
            start = next_start
            chunk_index += 1

        return chunks


class SentenceChunker(DocumentChunker):
    """Chunk documents by sentence boundaries."""

    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 1500,
        sentence_overlap: int = 1,
        max_document_chars: int = 1_000_000,
        max_sentences: int = 50_000,
        max_chunks: int = 10_000,
    ) -> None:
        if min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if sentence_overlap < 0:
            raise ValueError("sentence_overlap must be >= 0")
        if max_document_chars < 1:
            raise ValueError("max_document_chars must be >= 1")
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap
        self.max_document_chars = max_document_chars
        self.max_sentences = max_sentences
        self.max_chunks = max_chunks
        self._sentence_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+"
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document: Document) -> List[DocumentChunk]:
        if len(document.content) > self.max_document_chars:
            raise ValueError(
                "document content exceeds max_document_chars "
                f"({self.max_document_chars})"
            )
        sentences = self._split_sentences(document.content)
        if len(sentences) > self.max_sentences:
            raise ValueError(
                f"document has {len(sentences)} sentences; "
                f"limit is {self.max_sentences}"
            )
        chunks = []
        current_chunk_sentences = []
        current_size = 0
        chunk_index = 0
        char_position = 0

        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence)

            # Check if adding this sentence exceeds max size
            if (
                current_size + sentence_size > self.max_chunk_size
                and current_chunk_sentences
            ):
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                start_char = document.content.find(
                    current_chunk_sentences[0], char_position
                )

                if len(chunks) >= self.max_chunks:
                    raise ValueError(
                        f"chunk count exceeds max_chunks ({self.max_chunks})"
                    )
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                        document_id=document.document_id,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "sentence_count": len(current_chunk_sentences),
                            "chunking_strategy": "sentence",
                        },
                    )
                )

                # Start new chunk with overlap
                overlap_start = max(
                    0, len(current_chunk_sentences) - self.sentence_overlap
                )
                current_chunk_sentences = current_chunk_sentences[
                    overlap_start:
                ]
                current_size = sum(len(s) for s in current_chunk_sentences)
                char_position = start_char + len(chunk_text)
                chunk_index += 1

            current_chunk_sentences.append(sentence)
            current_size += sentence_size

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            start_char = document.content.find(
                current_chunk_sentences[0], char_position
            )

            if len(chunks) >= self.max_chunks:
                raise ValueError(
                    f"chunk count exceeds max_chunks ({self.max_chunks})"
                )
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                    document_id=document.document_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "sentence_count": len(current_chunk_sentences),
                        "chunking_strategy": "sentence",
                    },
                )
            )

        return chunks


class SemanticChunker(DocumentChunker):
    """Chunk documents based on semantic similarity between segments."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 200,
        max_chunk_size: int = 2000,
        embedding_batch_size: int = 128,
        max_sentences: int = 10_000,
        max_document_chars: int = 2_000_000,
        max_chunks: int = 10_000,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        if min_chunk_size < 1:
            raise ValueError("min_chunk_size must be >= 1")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        if max_document_chars < 1:
            raise ValueError("max_document_chars must be >= 1")
        if max_chunks < 1:
            raise ValueError("max_chunks must be >= 1")
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.embedding_batch_size = embedding_batch_size
        self.max_sentences = max_sentences
        self.max_document_chars = max_document_chars
        self.max_chunks = max_chunks

    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        For normalized embeddings (unit vectors), cosine similarity equals
        the dot product since ||a|| = ||b|| = 1. This measures the angular
        distance between vectors, making it ideal for semantic similarity
        where direction matters more than magnitude.

        Returns 0.0 when either input has effectively-zero norm so a
        buggy embedding model that emits a zero vector doesn't poison
        downstream chunking decisions with NaN.
        """
        a = np.array(vec1)
        b = np.array(vec2)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < np.finfo(float).eps:
            return 0.0
        return float(np.dot(a, b) / denom)

    def chunk(self, document: Document) -> List[DocumentChunk]:
        if len(document.content) > self.max_document_chars:
            raise ValueError(
                f"Document has {len(document.content)} characters; "
                f"max_document_chars is {self.max_document_chars}"
            )
        # First, split into sentences
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(document.content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []
        if len(sentences) > self.max_sentences:
            raise ValueError(
                f"Document has {len(sentences)} sentences; "
                f"max_sentences is {self.max_sentences}"
            )

        # Get embeddings in bounded batches so a long document does not
        # exceed provider limits or force one large in-memory request.
        embeddings: list[List[float]] = []
        for start in range(0, len(sentences), self.embedding_batch_size):
            batch = sentences[start : start + self.embedding_batch_size]
            embeddings.extend(self.embedding_model.embed_batch(batch))
        if len(embeddings) != len(sentences):
            raise ValueError(
                f"Embedding model returned {len(embeddings)} embeddings "
                f"for {len(sentences)} sentences"
            )

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = embeddings[0]
        # Track how many sentences contribute to the running mean so
        # that each addition is weighted 1/N rather than 1/2; see the
        # else-branch below for the incremental-mean update.
        current_chunk_count = 1
        sentence_starts = []
        search_offset = 0
        for sentence in sentences:
            start = document.content.find(sentence, search_offset)
            if start == -1:
                start = search_offset
            sentence_starts.append(start)
            search_offset = start + len(sentence)
        current_chunk_start = sentence_starts[0]
        chunk_index = 0

        for i in range(1, len(sentences)):
            sentence = sentences[i]
            sentence_embedding = embeddings[i]

            # Calculate similarity to current chunk
            similarity = self._cosine_similarity(
                current_chunk_embedding, sentence_embedding
            )

            current_size = sum(len(s) for s in current_chunk_sentences)

            # Decide whether to continue or break
            should_break = (
                similarity < self.similarity_threshold
                and current_size >= self.min_chunk_size
            ) or current_size + len(sentence) > self.max_chunk_size

            if should_break:
                # Save current chunk
                chunk_text = " ".join(current_chunk_sentences)
                start_char = current_chunk_start

                if len(chunks) >= self.max_chunks:
                    raise ValueError(
                        f"chunk count exceeds max_chunks ({self.max_chunks})"
                    )
                chunks.append(
                    DocumentChunk(
                        content=chunk_text,
                        chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                        document_id=document.document_id,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "chunking_strategy": "semantic",
                        },
                    )
                )

                current_chunk_sentences = [sentence]
                current_chunk_embedding = sentence_embedding
                current_chunk_count = 1
                current_chunk_start = sentence_starts[i]
                chunk_index += 1
            else:
                current_chunk_sentences.append(sentence)
                # Incremental mean: weight the new sentence 1/(N+1)
                # rather than 1/2, so older sentences are not
                # geometrically downweighted as the chunk grows.
                current_chunk_embedding = [
                    (a * current_chunk_count + b) / (current_chunk_count + 1)
                    for a, b in zip(
                        current_chunk_embedding, sentence_embedding
                    )
                ]
                current_chunk_count += 1

        # Save final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            start_char = current_chunk_start

            if len(chunks) >= self.max_chunks:
                raise ValueError(
                    f"chunk count exceeds max_chunks ({self.max_chunks})"
                )
            chunks.append(
                DocumentChunk(
                    content=chunk_text,
                    chunk_id=f"{document.document_id}_chunk_{chunk_index}",
                    document_id=document.document_id,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "chunking_strategy": "semantic",
                    },
                )
            )

        return chunks

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

@dataclass
class HierarchicalChunk(DocumentChunk):
    """Chunk with parent-child relationships."""

    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    level: int = 0


class HierarchicalChunker(DocumentChunker):
    """Create multi-level chunk hierarchy."""

    def __init__(
        self, level_configs: Optional[List[Dict[str, int]]] = None
    ) -> None:
        # Default: sections -> paragraphs -> sentences
        configs = level_configs or [
            {"min_size": 2000, "max_size": 5000},  # Level 0: Sections
            {"min_size": 500, "max_size": 1500},  # Level 1: Paragraphs
            {"min_size": 100, "max_size": 400},  # Level 2: Sentences
        ]
        if len(configs) != 3:
            raise ValueError("level_configs must define exactly 3 levels")
        for config in configs:
            if config["min_size"] < 1 or config["max_size"] < config["min_size"]:
                raise ValueError("invalid hierarchical chunk size bounds")
        self.level_configs = configs

    def chunk(self, document: Document) -> List[HierarchicalChunk]:
        all_chunks = []

        # Create top-level chunks
        top_chunker = FixedSizeChunker(
            chunk_size=self.level_configs[0]["max_size"], chunk_overlap=200
        )
        top_chunks = top_chunker.chunk(document)

        for i, top_chunk in enumerate(top_chunks):
            parent_chunk = HierarchicalChunk(
                content=top_chunk.content,
                chunk_id=f"{document.document_id}_L0_{i}",
                document_id=document.document_id,
                start_char=top_chunk.start_char,
                end_char=top_chunk.end_char,
                metadata={
                    **document.metadata,
                    "level": 0,
                    "chunking_strategy": "hierarchical",
                },
                level=0,
                child_ids=[],
            )

            # Create child chunks
            child_doc = Document(
                content=top_chunk.content,
                document_id=top_chunk.chunk_id,
                metadata=top_chunk.metadata,
            )

            child_chunker = SentenceChunker(
                min_chunk_size=self.level_configs[1]["min_size"],
                max_chunk_size=self.level_configs[1]["max_size"],
            )
            child_chunks = child_chunker.chunk(child_doc)

            for j, child in enumerate(child_chunks):
                child_hierarchical = HierarchicalChunk(
                    content=child.content,
                    chunk_id=f"{document.document_id}_L1_{i}_{j}",
                    document_id=document.document_id,
                    start_char=top_chunk.start_char + child.start_char,
                    end_char=top_chunk.start_char + child.end_char,
                    metadata={
                        **child.metadata,
                        "level": 1,
                        "parent_id": parent_chunk.chunk_id,
                    },
                    parent_id=parent_chunk.chunk_id,
                    level=1,
                )
                parent_chunk.child_ids.append(child_hierarchical.chunk_id)
                all_chunks.append(child_hierarchical)

            all_chunks.append(parent_chunk)

        return all_chunks

# ============================================================================
# Block 10 (chapter listing #10)
# ============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from collections import defaultdict
import math


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    retrieval_method: str


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        pass


class VectorRetriever(Retriever):
    """Semantic retrieval using vector similarity."""

    def __init__(
        self,
        vector_store: "VectorStore",
        embedding_model: EmbeddingModel,
        metrics: Optional[Any] = None,
        max_top_k: int = 100,
    ) -> None:
        if max_top_k < 1:
            raise ValueError("max_top_k must be >= 1")
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.metrics = metrics
        self.max_top_k = max_top_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > self.max_top_k:
            raise ValueError(f"top_k must be <= {self.max_top_k}")
        started = time.monotonic()
        try:
            query_embedding = self.embedding_model.embed_text(query)

            results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )
        except _RETRIEVER_OPERATION_EXCEPTIONS:
            if self.metrics and hasattr(self.metrics, "increment"):
                self.metrics.increment("retriever.error")
            raise
        finally:
            if self.metrics and hasattr(self.metrics, "record_timing"):
                self.metrics.record_timing(
                    "retriever.latency_ms",
                    (time.monotonic() - started) * 1000,
                )

        return [
            RetrievalResult(
                chunk_id=r["id"],
                content=r["content"],
                score=r["score"],
                metadata=r.get("metadata", {}),
                retrieval_method="vector",
            )
            for r in results
        ]


class BM25Retriever(Retriever):
    """Keyword retrieval using the BM25 ranking function.

    BM25 score for document D and query Q:

    score(D,Q) = sum_i IDF(q_i) * (f(q_i,D) * (k1+1)) /
                 (f(q_i,D) + k1 * (1 - b + b * |D|/avgdl))

    where f(q_i,D) is term frequency, |D| is doc length, avgdl is
    average doc length, k1 controls term saturation (typically 1.2-2.0),
    and b controls length normalization (typically 0.75).
    """

    # Scale rail: this in-memory implementation rebuilds its inverted index
    # on every add. Beyond ~100K chunks, back BM25 with Elasticsearch or
    # OpenSearch rather than hot-rebuilding in process memory.
    MAX_IN_MEMORY_DOCUMENTS = 100_000
    MAX_TOKENS_PER_DOCUMENT = 10_000
    MAX_QUERY_TERMS = 256

    def __init__(
        self,
        documents: List[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
        max_documents: int = MAX_IN_MEMORY_DOCUMENTS,
        max_tokens_per_document: int = MAX_TOKENS_PER_DOCUMENT,
        max_query_terms: int = MAX_QUERY_TERMS,
    ) -> None:
        if max_documents < 1:
            raise ValueError("max_documents must be >= 1")
        if max_tokens_per_document < 1:
            raise ValueError("max_tokens_per_document must be >= 1")
        if max_query_terms < 1:
            raise ValueError("max_query_terms must be >= 1")
        # The default values k1=1.5 and b=0.75 were empirically derived by
        # Robertson et al. and remain effective across diverse corpora.
        self.k1 = k1
        self.b = b
        self.max_documents = max_documents
        self.max_tokens_per_document = max_tokens_per_document
        self.max_query_terms = max_query_terms
        if len(documents) > self.max_documents:
            raise ValueError(
                f"BM25Retriever received {len(documents)} chunks "
                f"(hard cap {self.max_documents}). For larger corpora, "
                "use Elasticsearch/OpenSearch or another external keyword index."
            )
        # Soft warning at 80% of the cap: gives operators a runway to
        # plan the migration to an external index before the next ingest
        # batch hits the hard cap.
        soft_threshold = int(self.max_documents * 0.8)
        if len(documents) >= soft_threshold:
            logging.getLogger(__name__).warning(
                "BM25Retriever at %d/%d documents (>=80%% of cap); "
                "plan migration to external keyword index "
                "(Elasticsearch/OpenSearch) before the next ingest.",
                len(documents),
                self.max_documents,
            )
        self.documents = {doc.chunk_id: doc for doc in documents}
        self._build_index(documents)

    def _tokenize(self, text: str, max_tokens: Optional[int] = None) -> List[str]:
        """Simple tokenization."""
        tokens: List[str] = []
        for match in re.finditer(r"\b\w+\b", text.lower()):
            tokens.append(match.group(0))
            if max_tokens is not None and len(tokens) >= max_tokens:
                break
        return tokens

    def _build_index(self, documents: List[DocumentChunk]) -> None:
        """Build inverted index and document statistics."""
        self.doc_lengths = {}
        self.inverted_index = defaultdict(list)
        self.doc_freqs = defaultdict(int)
        total_length = 0

        for doc in documents:
            tokens = self._tokenize(
                doc.content, self.max_tokens_per_document + 1
            )
            if len(tokens) > self.max_tokens_per_document:
                raise ValueError(
                    f"BM25Retriever document {doc.chunk_id} has more than "
                    f"{self.max_tokens_per_document} tokens; chunk it before indexing."
                )
            self.doc_lengths[doc.chunk_id] = len(tokens)
            total_length += len(tokens)

            # Track unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] += 1

            # Build inverted index with term frequencies
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            for term, freq in term_freqs.items():
                self.inverted_index[term].append((doc.chunk_id, freq))

        self.avg_doc_length = (
            total_length / len(documents) if documents else 0
        )
        self.num_docs = len(documents)

    def _bm25_score(self, query_terms: List[str], doc_id: str) -> float:
        """Calculate BM25 score for a document."""
        score = 0.0
        doc_length = self.doc_lengths[doc_id]

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            # Find term frequency in this document
            tf = 0
            for did, freq in self.inverted_index[term]:
                if did == doc_id:
                    tf = freq
                    break

            if tf == 0:
                continue

            # Shifted IDF form used in Lucene / rank_bm25 (avoids negative IDF
            # for very common terms). Classical Robertson-Sparck Jones (1976)
            # gives the unshifted log(N/df); the shift is a practical
            # adjustment, not BM25+ (which is a separate TF lower-bound
            # contribution by Lv & Zhai, 2011).
            df = self.doc_freqs[term]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )
            score += idf * (numerator / denominator)

        return score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_terms = self._tokenize(query, self.max_query_terms)

        # Get candidate documents (those containing at least one query term)
        candidates = set()
        for term in query_terms:
            for doc_id, _ in self.inverted_index.get(term, []):
                candidates.add(doc_id)

        # Score all candidates
        scores = [
            (doc_id, self._bm25_score(query_terms, doc_id))
            for doc_id in candidates
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores:
            doc = self.documents[doc_id]
            if metadata_filter and any(
                doc.metadata.get(key) != value
                for key, value in metadata_filter.items()
            ):
                continue
            results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                    retrieval_method="bm25",
                )
            )
            if len(results) >= top_k:
                break

        return results


class HybridRetriever(Retriever):
    """Combine vector and keyword retrieval with fusion."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: BM25Retriever,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0.0 and 1.0")
        if fusion_method not in {"rrf", "linear"}:
            raise ValueError("fusion_method must be 'rrf' or 'linear'")
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha  # Weight for vector results (1-alpha for keyword)
        self.fusion_method = fusion_method

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """Combine results using Reciprocal Rank Fusion (RRF)."""
        scores = defaultdict(float)
        result_map = {}

        # Score from vector results
        for rank, result in enumerate(vector_results):
            scores[result.chunk_id] += 1 / (k + rank + 1)
            result_map[result.chunk_id] = result

        # Score from keyword results
        for rank, result in enumerate(keyword_results):
            scores[result.chunk_id] += 1 / (k + rank + 1)
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result

        # Sort by combined score
        sorted_ids = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True
        )

        return [
            RetrievalResult(
                chunk_id=cid,
                content=result_map[cid].content,
                score=scores[cid],
                metadata=result_map[cid].metadata,
                retrieval_method="hybrid_rrf",
            )
            for cid in sorted_ids
        ]

    def _linear_fusion(
        self,
        vector_results: List[RetrievalResult],
        keyword_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Combine results using linear score combination."""

        # Normalize scores to [0, 1]
        def normalize_scores(
            results: List[RetrievalResult],
        ) -> Dict[str, float]:
            if not results:
                return {}
            scores = {r.chunk_id: r.score for r in results}
            max_score = max(scores.values())
            min_score = min(scores.values())
            range_score = (
                max_score - min_score if max_score != min_score else 1
            )
            return {
                k: (v - min_score) / range_score for k, v in scores.items()
            }

        vector_scores = normalize_scores(vector_results)
        keyword_scores = normalize_scores(keyword_results)

        # Combine scores
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        combined_scores = {}
        result_map = {r.chunk_id: r for r in vector_results + keyword_results}

        for cid in all_ids:
            v_score = vector_scores.get(cid, 0)
            k_score = keyword_scores.get(cid, 0)
            combined_scores[cid] = (
                self.alpha * v_score + (1 - self.alpha) * k_score
            )

        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        return [
            RetrievalResult(
                chunk_id=cid,
                content=result_map[cid].content,
                score=combined_scores[cid],
                metadata=result_map[cid].metadata,
                retrieval_method="hybrid_linear",
            )
            for cid in sorted_ids
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_k: Optional[int] = None,
        keyword_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid approach."""
        vector_k = vector_k or top_k * 2
        keyword_k = keyword_k or top_k * 2

        vector_results = self.vector_retriever.retrieve(
            query, top_k=vector_k, metadata_filter=metadata_filter
        )
        keyword_results = self.keyword_retriever.retrieve(
            query, top_k=keyword_k, metadata_filter=metadata_filter
        )

        if self.fusion_method == "rrf":
            combined = self._reciprocal_rank_fusion(
                vector_results, keyword_results
            )
        else:
            combined = self._linear_fusion(vector_results, keyword_results)

        return combined[:top_k]

# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

from datetime import datetime, timezone
from typing import Set
from collections import OrderedDict
import hashlib


class IndexManager:
    """Manage vector index lifecycle and updates."""

    def __init__(
        self,
        vector_store: "VectorStore",
        embedding_model: EmbeddingModel,
        chunker: DocumentChunker,
        max_tracked_documents: int = 100_000,
        embedding_batch_size: int = 128,
    ) -> None:
        if max_tracked_documents < 1:
            raise ValueError("max_tracked_documents must be >= 1")
        if embedding_batch_size < 1:
            raise ValueError("embedding_batch_size must be >= 1")
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunker = chunker
        self._max_tracked_documents = max_tracked_documents
        self._embedding_batch_size = embedding_batch_size
        # Complete bounded manifest of documents this manager has indexed.
        # Hashes support change detection; the ID set supports
        # delete_missing=True without confusing an LRU cache for a full
        # index inventory.
        self._document_hashes: OrderedDict[str, str] = OrderedDict()
        self._indexed_document_ids: set[str] = set()

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _ensure_document_capacity(self, document_id: str) -> None:
        if (
            document_id not in self._indexed_document_ids
            and len(self._indexed_document_ids) >= self._max_tracked_documents
        ):
            raise RuntimeError(
                "indexed document manifest is full; raise "
                "max_tracked_documents or provide a persistent manifest"
            )

    def _get_existing_chunk_ids(self, document_id: str) -> Set[str]:
        """Get all chunk IDs for a document."""
        # Implementation depends on vector store capabilities
        return self.vector_store.get_ids_by_metadata(
            filter={"document_id": document_id}
        )

    def _remember_document_hash(
        self, document_id: str, content_hash: str
    ) -> None:
        """Track document hashes without losing the deletion manifest."""
        self._ensure_document_capacity(document_id)
        self._indexed_document_ids.add(document_id)
        self._document_hashes[document_id] = content_hash
        self._document_hashes.move_to_end(document_id)

    def add_document(self, document: Document) -> int:
        """Add new document to the index."""
        self._ensure_document_capacity(document.document_id)
        content_hash = self._compute_hash(document.content)

        # Check if document already exists
        existing_hash = self._document_hashes.get(document.document_id)
        if existing_hash is not None:
            self._document_hashes.move_to_end(document.document_id)
            if existing_hash == content_hash:
                return 0  # No changes needed
            else:
                # Document changed, update instead
                return self.update_document(document)

        # Chunk the document
        chunks = self.chunker.chunk(document)

        if not chunks:
            return 0

        inserted = 0
        indexed_at = datetime.now(timezone.utc).isoformat()
        for start in range(0, len(chunks), self._embedding_batch_size):
            batch = chunks[start : start + self._embedding_batch_size]
            embeddings = self.embedding_model.embed_batch(
                [chunk.content for chunk in batch]
            )
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Embedding model returned {len(embeddings)} embeddings "
                    f"for {len(batch)} chunks"
                )

            vectors_to_insert = [
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "content": chunk.content,
                    "metadata": {
                        **chunk.metadata,
                        "document_id": document.document_id,
                        "indexed_at": indexed_at,
                        "content_hash": content_hash,
                    },
                }
                for chunk, embedding in zip(batch, embeddings)
            ]
            self.vector_store.upsert(vectors_to_insert)
            inserted += len(batch)

        self._remember_document_hash(document.document_id, content_hash)

        return inserted

    def update_document(self, document: Document) -> int:
        """Update existing document in the index."""
        # Delete old chunks
        old_chunk_ids = self._get_existing_chunk_ids(document.document_id)
        if old_chunk_ids:
            self.vector_store.delete(list(old_chunk_ids))

        # Remove from hash tracking
        self._document_hashes.pop(document.document_id, None)

        # Re-add with new content
        return self.add_document(document)

    def delete_document(self, document_id: str) -> int:
        """Remove document from the index."""
        chunk_ids = self._get_existing_chunk_ids(document_id)

        if chunk_ids:
            self.vector_store.delete(list(chunk_ids))

        self._document_hashes.pop(document_id, None)
        self._indexed_document_ids.discard(document_id)

        return len(chunk_ids)

    def sync_documents(
        self, documents: List[Document], delete_missing: bool = True
    ) -> Dict[str, int]:
        """Sync index with document collection."""
        stats = {"added": 0, "updated": 0, "deleted": 0, "unchanged": 0}

        current_doc_ids = set()

        for document in documents:
            current_doc_ids.add(document.document_id)
            content_hash = self._compute_hash(document.content)

            if document.document_id not in self._document_hashes:
                # New document
                count = self.add_document(document)
                stats["added"] += count
            elif self._document_hashes[document.document_id] != content_hash:
                # Changed document
                count = self.update_document(document)
                stats["updated"] += count
            else:
                stats["unchanged"] += 1

        # Delete documents no longer in collection
        if delete_missing:
            indexed_doc_ids = set(self._indexed_document_ids)
            removed_ids = indexed_doc_ids - current_doc_ids

            for doc_id in removed_ids:
                count = self.delete_document(doc_id)
                stats["deleted"] += count

        return stats

# ============================================================================
# Block 12 (chapter listing #12)
# ============================================================================

@dataclass
class IndexHealth:
    """Health metrics for vector index."""

    total_vectors: int
    total_documents: int
    index_size_bytes: Optional[int]
    last_updated: datetime
    fragmentation_ratio: float
    query_latency_p50_ms: float
    query_latency_p99_ms: float


class IndexMonitor:
    """Monitor vector index health and performance."""

    LATENCY_WINDOW = 1000

    def __init__(self, vector_store: "VectorStore") -> None:
        from collections import deque
        import threading

        self.vector_store = vector_store
        # ``deque(maxlen=N)`` evicts in O(1) and is bounded
        # automatically; the prior list + slice trim was O(n) per
        # append past 1000 and not thread-safe under concurrent
        # ``record_query_latency`` from multiple worker threads.
        self._query_latencies: deque[float] = deque(
            maxlen=self.LATENCY_WINDOW
        )
        self._latency_lock = threading.Lock()

    def record_query_latency(self, latency_ms: float) -> None:
        """Record a query latency measurement."""
        with self._latency_lock:
            self._query_latencies.append(latency_ms)

    def get_health(self) -> IndexHealth:
        """Get current index health metrics.

        When no queries have been observed, latency percentiles are
        reported as NaN rather than 0 so callers can distinguish
        "no data yet" from "perfectly healthy".
        """
        stats = self.vector_store.get_stats()

        # Snapshot under the lock so concurrent record_query_latency
        # cannot mutate the deque while we sort it.
        with self._latency_lock:
            snapshot = list(self._query_latencies)

        if snapshot:
            latencies = sorted(snapshot)
            # min() guards against p99 picking an out-of-range index
            # on small samples (len * 0.99 rounds down).
            p50 = latencies[min(int(len(latencies) * 0.5), len(latencies) - 1)]
            p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
        else:
            p50 = float("nan")
            p99 = float("nan")

        return IndexHealth(
            total_vectors=stats.get("vector_count", 0),
            total_documents=stats.get("document_count", 0),
            index_size_bytes=stats.get("size_bytes"),
            last_updated=datetime.fromisoformat(
                stats.get(
                    "last_updated", datetime.now(timezone.utc).isoformat()
                )
            ),
            fragmentation_ratio=stats.get("fragmentation", 0.0),
            query_latency_p50_ms=p50,
            query_latency_p99_ms=p99,
        )

    def should_reindex(self) -> Tuple[bool, str]:
        """Determine if index should be rebuilt."""
        health = self.get_health()

        if health.fragmentation_ratio > 0.3:
            return True, "High fragmentation detected"

        # NaN compares as False in any direction; that's what we want
        # here ("no data yet" should not trigger a reindex).
        if health.query_latency_p99_ms > 500:
            return True, "Query latency degradation"

        return False, "Index healthy"

# ============================================================================
# Block 13 (chapter listing #13)
# ============================================================================

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable


@runtime_checkable
class _ChatCompletionsResource(Protocol):
    def create(self, *, model: str, messages: list[dict],
               **kwargs: Any) -> Any: ...


@runtime_checkable
class _ChatNamespace(Protocol):
    completions: _ChatCompletionsResource


@runtime_checkable
class LLMClient(Protocol):
    """Structural type for the OpenAI-style chat client this chapter uses.

    The RAG pipelines below call ``self.llm_client.chat.completions.
    create(model=..., messages=...)`` synchronously. The Protocol
    captures that nested surface so static checkers verify the contract.
    Anthropic-style clients (``messages.create``) and async chat APIs do
    not satisfy this Protocol directly; wrap them in a thin adapter
    exposing ``.chat.completions.create``.
    """

    chat: _ChatNamespace


@dataclass
class RAGResponse:
    """Response from RAG pipeline."""

    answer: str
    sources: List[RetrievalResult]
    confidence: float
    metadata: Dict[str, Any]


class RAGPipeline:
    """Basic RAG pipeline for agent knowledge retrieval."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 4000,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.25,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> None:
        if max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be >= 0")
        resolved_model = model or os.getenv("RAG_MODEL", "gpt-4o")
        if not resolved_model:
            raise ValueError("model must be non-empty")
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_context_tokens = max_context_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.model = resolved_model
        self.temperature = temperature
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that answers questions based on "
            "the provided context. If the context doesn't contain enough "
            "information to answer the question, say so. Always cite your "
            "sources by referencing the source documents."
        )

    def _format_context(self, results: List[RetrievalResult]) -> str:
        """Format retrieved results as context string."""
        context_parts = []

        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "Unknown")
            context_parts.append(
                f"[Source {i}] ({source}):\n{result.content}\n"
            )

        return "\n".join(context_parts)

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4

    def _truncate_context(
        self, results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Truncate results to fit token budget."""
        truncated = []
        total_tokens = 0

        for result in results:
            result_tokens = self._estimate_tokens(result.content)
            if total_tokens + result_tokens > self.max_context_tokens:
                break
            truncated.append(result)
            total_tokens += result_tokens

        return truncated

    def _generate_answer(
        self,
        results: List[RetrievalResult],
        context: str,
        question: str,
    ) -> RAGResponse:
        """Generate an answer for already-retrieved context."""
        user_message = f"""Context information:
{context}

Question: {question}

Please answer the question based on the context provided. Cite specific sources when possible."""

        # Generate response with bounded transient retries. If every
        # provider attempt fails, degrade rather than crashing the request:
        # the caller can inspect "retrieval succeeded; generation failed."
        response = None
        generation_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=self.temperature,
                    timeout=self.timeout_seconds,
                )
                break
            except _TRANSIENT_LLM_EXCEPTIONS as exc:
                generation_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))
            except (RuntimeError, ValueError) as exc:
                generation_error = exc
                break

        if response is None:
            logging.getLogger(__name__).warning(
                "RAG generation failed for question=%r: %s",
                question,
                generation_error,
            )
            return RAGResponse(
                answer="",
                sources=results,
                confidence=0.0,
                metadata={
                    "retrieval_count": len(results),
                    "generation_error": str(generation_error),
                },
            )

        answer = response.choices[0].message.content
        avg_score = sum(r.score for r in results) / len(results)

        return RAGResponse(
            answer=answer,
            sources=results,
            confidence=min(avg_score, 1.0),
            metadata={
                "retrieval_count": len(results),
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
            },
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RAGResponse:
        """Execute RAG query."""
        # Retrieve relevant context
        results = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
                metadata={"retrieval_count": 0},
            )

        # Truncate to fit context window
        results = self._truncate_context(results)
        if not results:
            return RAGResponse(
                answer="Retrieved documents exceeded the context budget.",
                sources=[],
                confidence=0.0,
                metadata={"retrieval_count": 0, "context_truncated": True},
            )
        context = self._format_context(results)
        return self._generate_answer(results, context, question)

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

class QueryExpander:
    """Expand queries for better retrieval coverage."""

    def __init__(
        self,
        llm_client: LLMClient,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        retry_backoff: float = 0.25,
        model: str = "gpt-4o-mini",
        max_variations: int = 5,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if not model:
            raise ValueError("model must be non-empty")
        if max_variations < 1:
            raise ValueError("max_variations must be >= 1")
        self.llm_client = llm_client
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.model = model
        self.max_variations = max_variations

    def _completion(self, prompt: str, temperature: float) -> Any:
        for attempt in range(self.max_retries):
            try:
                return self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=self.timeout_seconds,
                )
            except _TRANSIENT_LLM_EXCEPTIONS:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
        # Defensive: with max_retries >= 1 (enforced in __init__), the
        # loop either returns a response or raises. This explicit return
        # documents the intent and prevents an implicit ``None`` from
        # silently propagating if future edits change the loop structure.
        return None

    def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate query variations."""
        if num_variations < 1:
            raise ValueError("num_variations must be >= 1")
        if num_variations > self.max_variations:
            raise ValueError(
                f"num_variations must be <= {self.max_variations}"
            )
        prompt = (
            f"Generate {num_variations} alternative phrasings of "
            "this search query.\n"
            "Each variation should capture the same intent but use "
            "different words.\n\n"
            f"Original query: {query}\n\n"
            "Return only the variations, one per line."
        )

        response = self._completion(prompt, temperature=0.7)

        raw = response.choices[0].message.content.strip().split("\n")
        variations = [v.strip() for v in raw if v.strip()]
        # Clamp to the requested number to defend against
        # over-generative responses; fall back to the original query
        # if the provider returned nothing usable.
        if not variations:
            return [query]
        return [query] + variations[:num_variations]


class MultiQueryRAG(RAGPipeline):
    """RAG with query expansion for better recall."""

    def __init__(
        self, retriever: Retriever, llm_client: LLMClient, **kwargs: Any
    ) -> None:
        super().__init__(retriever, llm_client, **kwargs)
        self.query_expander = QueryExpander(
            llm_client,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            retry_backoff=self.retry_backoff,
        )

    def query(
        self,
        question: str,
        top_k: int = 5,
        expand_queries: bool = True,
        **kwargs: Any,
    ) -> RAGResponse:
        if not expand_queries:
            return super().query(question, top_k, **kwargs)

        # Expand query
        queries = self.query_expander.expand(question)

        # Retrieve for each query
        all_results = []
        seen_ids = set()

        for q in queries:
            results = self.retriever.retrieve(query=q, top_k=top_k)
            for result in results:
                if result.chunk_id not in seen_ids:
                    all_results.append(result)
                    seen_ids.add(result.chunk_id)

        # Re-rank combined results
        all_results.sort(key=lambda x: x.score, reverse=True)

        # Continue with standard RAG pipeline. ``_generate_answer`` is the
        # extension point subclasses provide; the hasattr guard above
        # ensures it exists by the time we reach this point.
        if not all_results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
                metadata={
                    "retrieval_count": 0,
                    "expanded_queries": len(queries),
                },
            )
        results = self._truncate_context(all_results[: top_k * 2])
        if not results:
            return RAGResponse(
                answer="Retrieved documents exceeded the context budget.",
                sources=[],
                confidence=0.0,
                metadata={
                    "retrieval_count": len(all_results),
                    "expanded_queries": len(queries),
                    "context_truncated": True,
                },
            )
        context = self._format_context(results)
        return self._generate_answer(results, context, question)


# ============================================================================
# Block 15 (chapter listing #15)
# ============================================================================

class CrossEncoderReranker:
    """Re-rank results using cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        max_candidates: int = 200,
        batch_size: int = 32,
        max_concurrency: int = 1,
    ) -> None:
        if max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        # Most cross-encoder backends are not thread-safe for inference
        # and the model itself is 100-500MB, so we serialize predict()
        # calls by default. Operators with a thread-safe backend or
        # dedicated GPU may raise ``max_concurrency`` to permit burst
        # parallelism without contention.
        self.max_concurrency = max_concurrency
        self._predict_semaphore = threading.Semaphore(max_concurrency)

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Re-rank results based on query relevance."""
        if not results:
            return results
        if top_k is not None and top_k < 1:
            return []

        candidates = results[: self.max_candidates]
        scores: List[float] = []
        for start in range(0, len(candidates), self.batch_size):
            batch = candidates[start : start + self.batch_size]
            pairs = [(query, r.content) for r in batch]
            with self._predict_semaphore:
                batch_scores = self.model.predict(pairs)
            scores.extend(float(score) for score in batch_scores)

        # Create new results with updated scores
        reranked = []
        for result, score in zip(candidates, scores):
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=score,
                    metadata={
                        **result.metadata,
                        "original_score": result.score,
                    },
                    retrieval_method=f"{result.retrieval_method}+rerank",
                )
            )

        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def unload(self) -> None:
        """Release the cross-encoder model reference.

        Long-lived processes can call this when reranking traffic stops
        to free model memory (the cross-encoder weights are typically
        100MB-500MB). After ``unload()`` the instance must not be used
        for further ``rerank`` calls.
        """
        self.model = None

# ============================================================================
# Block 16 (chapter listing #16)
# ============================================================================

class SelfRAGPipeline:
    """RAG with self-evaluation and adaptive retrieval."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        confidence_threshold: float = 0.7,
        timeout_seconds: float = 30.0,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff: float = 0.25,
        circuit_breaker_failures: int = 3,
        circuit_open_seconds: float = 30.0,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be > 0")
        if circuit_breaker_failures < 1:
            raise ValueError("circuit_breaker_failures must be >= 1")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be >= 0")
        if circuit_open_seconds <= 0:
            raise ValueError("circuit_open_seconds must be > 0")
        self.retriever = retriever
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold
        self.timeout_seconds = timeout_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.circuit_breaker_failures = circuit_breaker_failures
        self.circuit_open_seconds = circuit_open_seconds
        self._llm_failures = 0
        self._llm_circuit_open_until = 0.0

    def _completion(
        self,
        prompt: str,
        temperature: float,
        model: str = "gpt-4o-mini",
        deadline: Optional[float] = None,
    ) -> Any:
        now = time.monotonic()
        if now < self._llm_circuit_open_until:
            raise RuntimeError("LLM circuit breaker is open")
        for attempt in range(self.max_retries):
            try:
                remaining = (
                    self.timeout_seconds
                    if deadline is None
                    else max(0.0, deadline - time.monotonic())
                )
                if remaining <= 0:
                    raise TimeoutError("SelfRAG request deadline exceeded")
                response = self.llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    timeout=min(self.timeout_seconds, remaining),
                )
                self._llm_failures = 0
                self._llm_circuit_open_until = 0.0
                return response
            except _TRANSIENT_LLM_EXCEPTIONS:
                self._llm_failures += 1
                if self._llm_failures >= self.circuit_breaker_failures:
                    self._llm_circuit_open_until = (
                        time.monotonic() + self.circuit_open_seconds
                    )
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError("completion loop exited without returning")

    def _generate_answer_with_retry(
        self,
        results: List[RetrievalResult],
        question: str,
        deadline: Optional[float] = None,
    ) -> RAGResponse:
        for attempt in range(self.max_retries):
            try:
                return self._generate_answer(results, question, deadline)
            except _TRANSIENT_LLM_EXCEPTIONS:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError(
            "generate_answer_with_retry exhausted all attempts without returning"
        )

    def _generate_answer(
        self,
        results: List[RetrievalResult],
        question: str,
        deadline: Optional[float] = None,
    ) -> RAGResponse:
        """Generate an answer from retrieved context."""
        if not results:
            return RAGResponse(
                answer="I couldn't find relevant context to answer the question.",
                sources=[],
                confidence=0.0,
                metadata={"retrieval_used": True, "retrieval_count": 0},
            )

        context = "\n\n".join(
            f"[Source {i}] {r.content}" for i, r in enumerate(results, 1)
        )
        prompt = (
            "Answer the question using the retrieved context.\n"
            "Cite the source numbers when they support the answer."
            "\n\nContext:\n"
            f"{context}\n\n"
            f"Question: {question}"
        )

        response = self._completion(
            prompt, temperature=0.1, model="gpt-4o", deadline=deadline
        )
        avg_score = sum(r.score for r in results) / len(results)
        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=results,
            confidence=min(avg_score, 1.0),
            metadata={
                "retrieval_used": True,
                "retrieval_count": len(results),
                "model": "gpt-4o",
            },
        )

    def _needs_retrieval(
        self, question: str, deadline: Optional[float] = None
    ) -> bool:
        """Determine if question requires retrieval."""
        import textwrap

        prompt = textwrap.dedent(
            f"""\
            Determine if answering this question requires looking up specific
            information or if it can be answered from general knowledge.

            Question: {question}

            Respond with only "RETRIEVE" or "GENERAL".
            """
        ).rstrip("\n")

        response = self._completion(prompt, temperature=0, deadline=deadline)

        return "RETRIEVE" in response.choices[0].message.content.upper()

    def _evaluate_relevance(
        self,
        question: str,
        context: str,
        deadline: Optional[float] = None,
    ) -> float:
        """Evaluate if retrieved context is relevant."""
        import textwrap

        prompt = textwrap.dedent(
            f"""\
            Rate how relevant this context is for answering the question.
            Scale: 0.0 (completely irrelevant) to 1.0 (perfectly relevant)

            Question: {question}

            Context: {context}

            Respond with only a number between 0.0 and 1.0.
            """
        ).rstrip("\n")

        response = self._completion(prompt, temperature=0, deadline=deadline)

        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Execute self-evaluated RAG query."""
        deadline = time.monotonic() + self.request_timeout_seconds

        # Check if retrieval is needed
        if not self._needs_retrieval(question, deadline):
            # Answer directly without retrieval
            response = self._completion(
                question,
                temperature=0.1,
                model="gpt-4o",
                deadline=deadline,
            )

            return RAGResponse(
                answer=response.choices[0].message.content,
                sources=[],
                confidence=0.8,
                metadata={"retrieval_used": False},
            )

        # Retrieve and evaluate
        if time.monotonic() >= deadline:
            raise TimeoutError("SelfRAG request deadline exceeded")
        results = self.retriever.retrieve(query=question, top_k=top_k)

        if results:
            context = "\n".join(r.content for r in results[:3])
            relevance = self._evaluate_relevance(question, context, deadline)

            if relevance < self.confidence_threshold:
                # Context not relevant enough, try broader retrieval
                if time.monotonic() >= deadline:
                    raise TimeoutError("SelfRAG request deadline exceeded")
                results = self.retriever.retrieve(
                    query=question, top_k=top_k * 2
                )

        # Generate answer with context using the subclass's hook.
        # The hasattr guard at function entry ensures _generate_answer
        # exists by the time we reach this point.
        return self._generate_answer_with_retry(results, question, deadline)


# ============================================================================
# Block 17 (chapter listing #17)
# ============================================================================

"""
Code Navigation (line numbers are approximate):
- VectorStore ABC ... ~20
- PineconeStore ... ~70
- WeaviateStore ... ~170
- ChromaStore ... ~250
- QdrantStore ... ~340
- Production Configuration ... ~430
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, TypeVar

_QdrantResult = TypeVar("_QdrantResult")
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import hashlib
import json


class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    def upsert(self, vectors: List[Dict[str, Any]]) -> int:
        """Insert or update vectors."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """Delete vectors by ID."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pass

    @abstractmethod
    def get_ids_by_metadata(
        self,
        filter: Dict[str, Any],
        *,
        max_results: int = 100_000,
        page_size: int = 1024,
    ) -> set[str]:
        """Get vector IDs matching a metadata filter, bounded by max_results."""
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore."""

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        dimension: int = 1536,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 0.1,
        retry_jitter: float = 0.05,
        max_batch_size: int = 512,
        max_top_k: int = 100,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host=host, port=port, timeout=timeout)
        self.collection_name = collection_name
        self.dimension = dimension
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if retry_backoff < 0:
            raise ValueError("retry_backoff must be >= 0")
        if retry_jitter < 0:
            raise ValueError("retry_jitter must be >= 0")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_top_k < 1:
            raise ValueError("max_top_k must be >= 1")
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.retry_jitter = retry_jitter
        self.max_batch_size = max_batch_size
        self.max_top_k = max_top_k

        # Create collection if it doesn't exist
        collections = self._with_retry(self.client.get_collections).collections
        if collection_name not in [c.name for c in collections]:
            self._with_retry(
                self.client.create_collection,
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension, distance=Distance.COSINE
                ),
            )

    def _with_retry(
        self,
        operation: Callable[..., _QdrantResult],
        *args: Any,
        **kwargs: Any,
    ) -> _QdrantResult:
        """Run a Qdrant operation with bounded transient retries."""
        last_error = None
        transient_errors = (TimeoutError, ConnectionError, OSError)
        try:
            from httpx import TimeoutException, TransportError

            transient_errors = transient_errors + (
                TimeoutException,
                TransportError,
            )
        except ImportError:
            pass
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (ValueError, TypeError):
                raise
            except transient_errors as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(
                    (self.retry_backoff * (2**attempt))
                    + random.uniform(0, self.retry_jitter)
                )
        if last_error is None:
            # Should not happen: __init__ enforces max_retries >= 1, so the
            # loop above ran at least one attempt and either returned or
            # populated last_error. Raise an explicit RuntimeError so the
            # failure mode is loud rather than masquerading as ``raise None``.
            raise RuntimeError(
                "retry loop exited without attempting any call"
            )
        raise last_error

    def upsert(self, vectors: List[Dict[str, Any]]) -> int:
        from qdrant_client.models import PointStruct

        inserted = 0
        for start in range(0, len(vectors), self.max_batch_size):
            batch = vectors[start : start + self.max_batch_size]
            points = [
                PointStruct(
                    id=v["id"],
                    vector=v["vector"],
                    payload={"content": v["content"], **v.get("metadata", {})},
                )
                for v in batch
            ]
            self._with_retry(
                self.client.upsert,
                collection_name=self.collection_name,
                points=points,
            )
            inserted += len(points)

        return inserted

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if top_k > self.max_top_k:
            raise ValueError(f"top_k must be <= {self.max_top_k}")
        query_filter = None
        if metadata_filter:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ]
            query_filter = Filter(must=conditions)

        results = self._with_retry(
            self.client.search,
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "id": str(r.id),
                "content": r.payload.get("content", ""),
                "score": r.score,
                "metadata": {
                    k: v for k, v in r.payload.items() if k != "content"
                },
            }
            for r in results
        ]

    def delete(self, ids: List[str]) -> int:
        from qdrant_client.models import PointIdsList

        self._with_retry(
            self.client.delete,
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

        return len(ids)

    def get_stats(self) -> Dict[str, Any]:
        info = self._with_retry(self.client.get_collection, self.collection_name)
        return {
            "vector_count": info.points_count,
            # Qdrant's Python client does not expose disk byte size
            # directly; the previous version mislabeled payload_schema
            # (a dict of field types) as size_bytes. Surface both
            # fields with their real names instead. ``size_bytes`` is
            # ``None`` to signal "unavailable" rather than "zero".
            "payload_schema": info.payload_schema,
            "size_bytes": None,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def get_ids_by_metadata(
        self,
        filter: Dict[str, Any],
        *,
        max_results: int = 100_000,
        page_size: int = 1024,
    ) -> set[str]:
        """Get all IDs matching metadata filter.

        Pages through Qdrant's scroll cursor until exhausted. The
        previous implementation passed ``limit=10000`` and read only
        the first page, which silently truncated large collections
        and let downstream updates/deletes leak orphaned vectors.
        ``max_results`` prevents broad filters from materializing an
        unbounded ID set in memory.
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        if page_size < 1:
            raise ValueError("page_size must be >= 1")

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter.items()
        ]

        ids: set[str] = set()
        next_offset = None
        while True:
            # Compute remaining capacity; add 1 so the next iteration
            # can detect a single overflow point and raise rather than
            # silently truncate. ``max(..., 1)`` ensures the request is
            # always valid even when remaining is 0.
            remaining = max_results - len(ids)
            limit = min(page_size, max(remaining + 1, 1))
            points, next_offset = self._with_retry(
                self.client.scroll,
                collection_name=self.collection_name,
                scroll_filter=Filter(must=conditions),
                limit=limit,
                offset=next_offset,
                with_payload=False,
            )
            for point in points:
                ids.add(str(point.id))
                if len(ids) > max_results:
                    raise RuntimeError(
                        f"metadata filter matched more than {max_results} IDs"
                    )
            if next_offset is None:
                break
            if len(ids) >= max_results:
                raise RuntimeError(
                    f"metadata filter matched at least {max_results} IDs; "
                    "narrow the filter or raise max_results"
                )
        return ids

    def close(self) -> None:
        """Release the underlying Qdrant client, if it supports closing.

        Older versions of ``qdrant-client`` did not expose ``close``,
        so we guard with ``hasattr`` to stay compatible. Calling this
        from application shutdown lets the HTTP/gRPC transport drop
        its sockets promptly instead of waiting for GC.
        """
        if hasattr(self.client, "close"):
            self.client.close()


@dataclass
class KnowledgeSource:
    """Represents a source of knowledge documents."""

    source_id: str
    source_type: str  # confluence, notion, filesystem, etc.
    config: Dict[str, Any]
    last_synced: Optional[datetime] = None


@dataclass
class KnowledgeDocument:
    """Document in the knowledge base."""

    document_id: str
    title: str
    content: str
    source_id: str
    source_url: Optional[str] = None
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


class KnowledgeBase:
    """Enterprise knowledge base with RAG capabilities."""

    # Scale rail: chunks are retained locally to support BM25 rebuilds and
    # stats. Beyond this threshold, stream chunks from the vector store
    # rather than holding the whole corpus in process memory.
    MAX_IN_MEMORY_CHUNKS = 500_000
    MAX_IN_MEMORY_DOCUMENTS = 100_000

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        chunker: Optional[DocumentChunker] = None,
        max_in_memory_chunks: int = MAX_IN_MEMORY_CHUNKS,
        max_in_memory_documents: int = MAX_IN_MEMORY_DOCUMENTS,
        ingest_batch_size: int = 64,
    ) -> None:
        if max_in_memory_chunks < 1:
            raise ValueError("max_in_memory_chunks must be >= 1")
        if max_in_memory_documents < 1:
            raise ValueError("max_in_memory_documents must be >= 1")
        if ingest_batch_size < 1:
            raise ValueError("ingest_batch_size must be >= 1")
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.chunker = chunker or SentenceChunker()
        self.max_in_memory_chunks = max_in_memory_chunks
        self.max_in_memory_documents = max_in_memory_documents
        self.ingest_batch_size = ingest_batch_size

        # Initialize retrievers
        self._documents: Dict[str, KnowledgeDocument] = {}
        self._chunks: List[DocumentChunk] = []
        self._scale_warning_emitted = False
        self._index_lock = threading.RLock()
        self._defer_keyword_rebuilds = 0
        self._keyword_index_dirty = False

        # Build retrieval pipeline
        self.vector_retriever = VectorRetriever(
            vector_store=vector_store, embedding_model=embedding_model
        )

        self.rag_pipeline = None  # Initialized after documents loaded

    def add_document(self, document: KnowledgeDocument) -> int:
        """Add a document to the knowledge base."""
        # Convert to chunking format
        doc = Document(
            content=document.content,
            document_id=document.document_id,
            metadata={
                "title": document.title,
                "source_id": document.source_id,
                "source_url": document.source_url,
                **document.metadata,
            },
        )

        # Chunk the document
        chunks = self.chunker.chunk(doc)

        if not chunks:
            return 0

        # Generate embeddings and upsert in bounded batches so a large
        # source document cannot allocate every vector at once.
        stored_chunks = 0
        for start in range(0, len(chunks), self.ingest_batch_size):
            batch = chunks[start : start + self.ingest_batch_size]
            contents = [chunk.content for chunk in batch]
            embeddings = self.embedding_model.embed_batch(contents)
            if len(embeddings) != len(batch):
                raise ValueError(
                    f"Embedding model returned {len(embeddings)} "
                    f"embeddings for {len(batch)} chunks"
                )

            vectors = [
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk, embedding in zip(batch, embeddings)
            ]
            self.vector_store.upsert(vectors)
            stored_chunks += len(batch)

        # Track locally. Keep all in-process indexes under one lock so a
        # reader cannot observe documents/chunks while the BM25 index still
        # represents a previous snapshot.
        with self._index_lock:
            self._documents[document.document_id] = document
            while len(self._documents) > self.max_in_memory_documents:
                self._documents.pop(next(iter(self._documents)))
            available_slots = self.max_in_memory_chunks - len(self._chunks)
            if available_slots > 0:
                self._chunks.extend(chunks[:available_slots])

            if len(chunks) > available_slots and not self._scale_warning_emitted:
                import warnings
                warnings.warn(
                    f"KnowledgeBase reached its in-memory keyword cap "
                    f"({self.max_in_memory_chunks} chunks). Additional chunks "
                    "were written to the vector store only; use an external "
                    "keyword index for full-corpus BM25 at this scale.",
                    stacklevel=2,
                )
                self._scale_warning_emitted = True

            self._keyword_index_dirty = True
            if self._defer_keyword_rebuilds == 0:
                self._rebuild_keyword_index_locked()
                self._keyword_index_dirty = False

        return stored_chunks

    def _mark_keyword_index_dirty(self) -> None:
        """Rebuild keyword indexes unless a bulk import is deferring work."""
        with self._index_lock:
            self._keyword_index_dirty = True
            if self._defer_keyword_rebuilds == 0:
                self._rebuild_keyword_index_locked()
                self._keyword_index_dirty = False

    def _rebuild_keyword_index(self) -> None:
        """Rebuild keyword search index after document changes."""
        with self._index_lock:
            self._rebuild_keyword_index_locked()
            self._keyword_index_dirty = False

    def _rebuild_keyword_index_locked(self) -> None:
        """Rebuild keyword search index. Caller must hold _index_lock."""
        if self._chunks:
            self.keyword_retriever = BM25Retriever(self._chunks)

            self.hybrid_retriever = HybridRetriever(
                vector_retriever=self.vector_retriever,
                keyword_retriever=self.keyword_retriever,
                alpha=0.6,
                fusion_method="rrf",
            )

            self.rag_pipeline = RAGPipeline(
                retriever=self.hybrid_retriever, llm_client=self.llm_client
            )

    def query(
        self, question: str, top_k: int = 5, include_sources: bool = True
    ) -> Dict[str, Any]:
        """Query the knowledge base."""
        if not self.rag_pipeline:
            return {
                "answer": "Knowledge base is empty. Please add documents first.",
                "sources": [],
                "confidence": 0.0,
            }

        response = self.rag_pipeline.query(question, top_k=top_k)

        result = {
            "answer": response.answer,
            "confidence": response.confidence,
            "metadata": response.metadata,
        }

        if include_sources:
            result["sources"] = [
                {
                    "content": (
                        s.content[:200] + "..."
                        if len(s.content) > 200
                        else s.content
                    ),
                    "title": s.metadata.get("title", "Unknown"),
                    "source_url": s.metadata.get("source_url"),
                    "relevance_score": s.score,
                }
                for s in response.sources
            ]

        return result

    def bulk_import(
        self, documents: List[KnowledgeDocument], batch_size: int = 10
    ) -> Dict[str, int]:
        """Import multiple documents efficiently.

        ``add_document`` normally rebuilds the BM25 index after each
        document. For bulk import we defer rebuilds under a lock and run
        one rebuild at the end (O(N) instead of O(N^2)). Failures are
        logged with stack traces via ``logger.exception`` rather than
        printed.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        stats = {"total": len(documents), "chunks": 0, "failed": 0}

        with self._index_lock:
            self._defer_keyword_rebuilds += 1
        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                for doc in batch:
                    try:
                        chunks_added = self.add_document(doc)
                        stats["chunks"] += chunks_added
                    except (
                        TimeoutError,
                        ConnectionError,
                        OSError,
                        RuntimeError,
                        ValueError,
                    ):
                        stats["failed"] += 1
                        logging.getLogger(__name__).exception(
                            "Failed to import document %s", doc.document_id
                        )
                    except Exception:
                        # Generic fallback so an unexpected exception
                        # class on a single document does not abort the
                        # whole bulk import. Returns a partial-failure
                        # summary via the ``stats`` dict.
                        stats["failed"] += 1
                        logging.getLogger(__name__).exception(
                            "Unexpected failure importing document %s",
                            doc.document_id,
                        )
        finally:
            with self._index_lock:
                self._defer_keyword_rebuilds -= 1
                should_rebuild = (
                    self._defer_keyword_rebuilds == 0
                    and self._keyword_index_dirty
                )
                if should_rebuild:
                    self._rebuild_keyword_index_locked()
                    self._keyword_index_dirty = False

        if stats["failed"] > 0:
            # Structured counter event for ops dashboards. Production
            # callers should route this through their metrics client
            # (StatsD, OpenTelemetry, etc.); we emit a logger record with
            # an explicit metric name and value so the event can be
            # scraped from logs even without a metrics SDK installed.
            logging.getLogger(__name__).warning(
                "metric=bulk_import.failure value=%d total=%d",
                stats["failed"],
                stats["total"],
            )

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        vector_stats = self.vector_store.get_stats()

        return {
            "document_count": len(self._documents),
            "chunk_count": len(self._chunks),
            "vector_count": vector_stats.get("vector_count", 0),
            "sources": list(
                set(d.source_id for d in self._documents.values())
            ),
        }


def create_knowledge_base_example() -> KnowledgeBase:
    """Example of setting up and using the knowledge base."""

    # Initialize components
    embedding_model = OpenAIEmbedding(model="text-embedding-3-small")

    vector_store = QdrantVectorStore(
        collection_name="enterprise_kb",
        host="localhost",
        port=6333,
        dimension=embedding_model.dimension,
    )

    from openai import OpenAI

    llm_client = OpenAI()

    # Create knowledge base
    kb = KnowledgeBase(
        vector_store=vector_store,
        embedding_model=embedding_model,
        llm_client=llm_client,
        chunker=SentenceChunker(
            min_chunk_size=300, max_chunk_size=1000, sentence_overlap=2
        ),
    )

    # Add sample documents
    documents = [
        KnowledgeDocument(
            document_id="doc-001",
            title="Password Reset Procedure",
            content="""
            To reset your password, follow these steps:
            1. Navigate to the login page at https://app.example.com/login
            2. Click the "Forgot Password" link below the login form
            3. Enter your registered email address
            4. Check your email for a password reset link (valid for 24 hours)
            5. Click the link and enter your new password
            6. Your password must be at least 12 characters with mixed case and numbers
            
            If you don't receive the email within 5 minutes, check your spam folder.
            For additional help, contact IT Support at support@example.com.
            """,
            source_id="internal-wiki",
            source_url="https://wiki.example.com/it/password-reset",
            metadata={"category": "authentication", "department": "IT"},
        ),
        KnowledgeDocument(
            document_id="doc-002",
            title="VPN Setup Guide",
            content="""
            Remote Access VPN Configuration
            
            Prerequisites:
            - Company-issued laptop or approved personal device
            - Active employee account
            - VPN client software (available from IT portal)
            
            Setup Steps:
            1. Download the VPN client from https://it.example.com/vpn-client
            2. Install the application with default settings
            3. Launch the VPN client
            4. Server address: vpn.example.com
            5. Authentication: Use your company email and password
            6. Enable two-factor authentication when prompted
            
            Troubleshooting:
            - Connection timeout: Ensure you're not on a restricted network
            - Authentication failed: Verify your credentials are correct
            - Slow connection: Try connecting to a different VPN server region
            """,
            source_id="internal-wiki",
            source_url="https://wiki.example.com/it/vpn-setup",
            metadata={"category": "remote-access", "department": "IT"},
        ),
    ]

    # Import documents
    stats = kb.bulk_import(documents)
    print(
        f"Imported {stats['total']} documents with {stats['chunks']} chunks"
    )

    # Query the knowledge base
    response = kb.query(
        "How do I connect to the company network from home?", top_k=3
    )

    print("\n=== Query Response ===")
    print(f"Answer: {response['answer']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"\nSources:")
    for source in response["sources"]:
        print(
            f"  - {source['title']} (score: {source['relevance_score']:.3f})"
        )

    return kb


if __name__ == "__main__":
    kb = create_knowledge_base_example()
