"""
Vector Databases and Agent Memory

Code listings from Chapter 03, Book 3:
"Agentic AI in Production: Scaling and Applying Autonomous Systems"
by Dr. Vijay Raghavan

This file faithfully reproduces every code listing from the chapter, in book
order, with section banners showing the block number. Most listings are
runnable Python that builds incrementally; some are illustrative fragments
(log output, file trees, Dockerfile snippets, JSON examples) preserved as
docstrings so this file always remains valid Python.

To use a particular class or function, copy it into your own project and
provide the surrounding context (imports, dependencies) as needed.
"""


# ============================================================================
# Block 1 (chapter listing #1)
# ============================================================================

import pinecone
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

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
from typing import List
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


class OpenAIEmbedding(EmbeddingModel):
    """OpenAI embedding model implementation."""

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: str = None
    ):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "small" in model else 3072

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # OpenAI supports batching up to 2048 texts
        response = self.client.embeddings.create(
            model=self.model, input=texts
        )
        return [item.embedding for item in response.data]


class SentenceTransformerEmbedding(EmbeddingModel):
    """Local embedding using Sentence Transformers."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return embeddings.tolist()

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

class InstructorEmbedding(EmbeddingModel):
    """Instructor embedding with task-specific prefixes."""

    def __init__(self, model_name: str = "hkunlp/instructor-large"):
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

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> List[DocumentChunk]:
        chunks = []
        text = document.content
        start = 0
        chunk_index = 0

        while start < len(text):
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

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks


class SentenceChunker(DocumentChunker):
    """Chunk documents by sentence boundaries."""

    def __init__(
        self,
        min_chunk_size: int = 500,
        max_chunk_size: int = 1500,
        sentence_overlap: int = 1,
    ):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.sentence_overlap = sentence_overlap
        self._sentence_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\n+"
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = self._sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, document: Document) -> List[DocumentChunk]:
        sentences = self._split_sentences(document.content)
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
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def _cosine_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors.

        For normalized embeddings (unit vectors), cosine similarity equals
        the dot product since ||a|| = ||b|| = 1. This measures the angular
        distance between vectors, making it ideal for semantic similarity
        where direction matters more than magnitude.
        """
        a = np.array(vec1)
        b = np.array(vec2)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def chunk(self, document: Document) -> List[DocumentChunk]:
        # First, split into sentences
        sentence_pattern = re.compile(r"(?<=[.!?])\s+")
        sentences = sentence_pattern.split(document.content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        # Get embeddings for all sentences
        embeddings = self.embedding_model.embed_batch(sentences)

        chunks = []
        current_chunk_sentences = [sentences[0]]
        current_chunk_embedding = embeddings[0]
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
                start_char = document.content.find(current_chunk_sentences[0])

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
                chunk_index += 1
            else:
                current_chunk_sentences.append(sentence)
                # Update chunk embedding as average
                current_chunk_embedding = [
                    (a + b) / 2
                    for a, b in zip(
                        current_chunk_embedding, sentence_embedding
                    )
                ]

        # Save final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            start_char = document.content.find(current_chunk_sentences[0])

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

    def __init__(self, level_configs: List[Dict[str, int]] = None):
        # Default: sections -> paragraphs -> sentences
        self.level_configs = level_configs or [
            {"min_size": 2000, "max_size": 5000},  # Level 0: Sections
            {"min_size": 500, "max_size": 1500},  # Level 1: Paragraphs
            {"min_size": 100, "max_size": 400},  # Level 2: Sentences
        ]

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
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve relevant chunks for a query."""
        pass


class VectorRetriever(Retriever):
    """Semantic retrieval using vector similarity."""

    def __init__(
        self, vector_store: "VectorStore", embedding_model: EmbeddingModel
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[RetrievalResult]:
        query_embedding = self.embedding_model.embed_text(query)

        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
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

    def __init__(
        self, documents: List[DocumentChunk], k1: float = 1.5, b: float = 0.75
    ):
        # The default values k1=1.5 and b=0.75 were empirically derived by
        # Robertson et al. and remain effective across diverse corpora.
        self.k1 = k1
        self.b = b
        self.documents = {doc.chunk_id: doc for doc in documents}
        self._build_index(documents)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self, documents: List[DocumentChunk]):
        """Build inverted index and document statistics."""
        self.doc_lengths = {}
        self.inverted_index = defaultdict(list)
        self.doc_freqs = defaultdict(int)
        total_length = 0

        for doc in documents:
            tokens = self._tokenize(doc.content)
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

            # IDF calculation
            df = self.doc_freqs[term]
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        query_terms = self._tokenize(query)

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
        for doc_id, score in scores[:top_k]:
            doc = self.documents[doc_id]
            results.append(
                RetrievalResult(
                    chunk_id=doc_id,
                    content=doc.content,
                    score=score,
                    metadata=doc.metadata,
                    retrieval_method="bm25",
                )
            )

        return results


class HybridRetriever(Retriever):
    """Combine vector and keyword retrieval with fusion."""

    def __init__(
        self,
        vector_retriever: VectorRetriever,
        keyword_retriever: BM25Retriever,
        alpha: float = 0.5,
        fusion_method: str = "rrf",
    ):
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
        vector_k: int = None,
        keyword_k: int = None,
    ) -> List[RetrievalResult]:
        """Retrieve using hybrid approach."""
        vector_k = vector_k or top_k * 2
        keyword_k = keyword_k or top_k * 2

        vector_results = self.vector_retriever.retrieve(query, top_k=vector_k)
        keyword_results = self.keyword_retriever.retrieve(
            query, top_k=keyword_k
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
import hashlib


class IndexManager:
    """Manage vector index lifecycle and updates."""

    def __init__(
        self,
        vector_store: "VectorStore",
        embedding_model: EmbeddingModel,
        chunker: DocumentChunker,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.chunker = chunker
        self._document_hashes: Dict[str, str] = {}

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_existing_chunk_ids(self, document_id: str) -> Set[str]:
        """Get all chunk IDs for a document."""
        # Implementation depends on vector store capabilities
        return self.vector_store.get_ids_by_metadata(
            filter={"document_id": document_id}
        )

    def add_document(self, document: Document) -> int:
        """Add new document to the index."""
        content_hash = self._compute_hash(document.content)

        # Check if document already exists
        if document.document_id in self._document_hashes:
            if self._document_hashes[document.document_id] == content_hash:
                return 0  # No changes needed
            else:
                # Document changed, update instead
                return self.update_document(document)

        # Chunk the document
        chunks = self.chunker.chunk(document)

        if not chunks:
            return 0

        # Generate embeddings
        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(contents)

        # Insert into vector store
        vectors_to_insert = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors_to_insert.append(
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "content": chunk.content,
                    "metadata": {
                        **chunk.metadata,
                        "document_id": document.document_id,
                        "indexed_at": datetime.now(timezone.utc).isoformat(),
                        "content_hash": content_hash,
                    },
                }
            )

        self.vector_store.upsert(vectors_to_insert)
        self._document_hashes[document.document_id] = content_hash

        return len(chunks)

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
            indexed_doc_ids = set(self._document_hashes.keys())
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
    index_size_bytes: int
    last_updated: datetime
    fragmentation_ratio: float
    query_latency_p50_ms: float
    query_latency_p99_ms: float


class IndexMonitor:
    """Monitor vector index health and performance."""

    def __init__(self, vector_store: "VectorStore"):
        self.vector_store = vector_store
        self._query_latencies: List[float] = []

    def record_query_latency(self, latency_ms: float):
        """Record a query latency measurement."""
        self._query_latencies.append(latency_ms)
        # Keep last 1000 measurements
        if len(self._query_latencies) > 1000:
            self._query_latencies = self._query_latencies[-1000:]

    def get_health(self) -> IndexHealth:
        """Get current index health metrics."""
        stats = self.vector_store.get_stats()

        latencies = (
            sorted(self._query_latencies) if self._query_latencies else [0]
        )
        p50_idx = int(len(latencies) * 0.5)
        p99_idx = int(len(latencies) * 0.99)

        return IndexHealth(
            total_vectors=stats.get("vector_count", 0),
            total_documents=stats.get("document_count", 0),
            index_size_bytes=stats.get("size_bytes", 0),
            last_updated=datetime.fromisoformat(
                stats.get(
                    "last_updated", datetime.now(timezone.utc).isoformat()
                )
            ),
            fragmentation_ratio=stats.get("fragmentation", 0.0),
            query_latency_p50_ms=latencies[p50_idx],
            query_latency_p99_ms=latencies[p99_idx],
        )

    def should_reindex(self) -> Tuple[bool, str]:
        """Determine if index should be rebuilt."""
        health = self.get_health()

        if health.fragmentation_ratio > 0.3:
            return True, "High fragmentation detected"

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
        system_prompt: str = None,
        max_context_tokens: int = 4000,
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.max_context_tokens = max_context_tokens
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

    def query(
        self,
        question: str,
        top_k: int = 5,
        metadata_filter: Dict[str, Any] = None,
    ) -> RAGResponse:
        """Execute RAG query."""
        # Retrieve relevant context
        results = self.retriever.retrieve(query=question, top_k=top_k)

        if not results:
            return RAGResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                confidence=0.0,
                metadata={"retrieval_count": 0},
            )

        # Truncate to fit context window
        results = self._truncate_context(results)
        context = self._format_context(results)

        # Construct prompt
        user_message = f"""Context information:
{context}

Question: {question}

Please answer the question based on the context provided. Cite specific sources when possible."""

        # Generate response
        response = self.llm_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        # Calculate confidence based on retrieval scores
        avg_score = sum(r.score for r in results) / len(results)

        return RAGResponse(
            answer=answer,
            sources=results,
            confidence=min(avg_score, 1.0),
            metadata={
                "retrieval_count": len(results),
                "model": "gpt-4o",
                "tokens_used": response.usage.total_tokens,
            },
        )

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

class QueryExpander:
    """Expand queries for better retrieval coverage."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """Generate query variations."""
        prompt = f"""Generate {num_variations} alternative phrasings of this search query.
Each variation should capture the same intent but use different words.

Original query: {query}

Return only the variations, one per line."""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )

        variations = response.choices[0].message.content.strip().split("\n")
        return [query] + [v.strip() for v in variations if v.strip()]


class MultiQueryRAG(RAGPipeline):
    """RAG with query expansion for better recall."""

    def __init__(self, retriever: Retriever, llm_client: LLMClient, **kwargs):
        super().__init__(retriever, llm_client, **kwargs)
        self.query_expander = QueryExpander(llm_client)

    def query(
        self,
        question: str,
        top_k: int = 5,
        expand_queries: bool = True,
        **kwargs,
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

        # Continue with standard RAG pipeline
        results = self._truncate_context(all_results[: top_k * 2])
        context = self._format_context(results)

        # ... rest of generation logic

# ============================================================================
# Block 15 (chapter listing #15)
# ============================================================================

class CrossEncoderReranker:
    """Re-rank results using cross-encoder model."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: List[RetrievalResult], top_k: int = None
    ) -> List[RetrievalResult]:
        """Re-rank results based on query relevance."""
        if not results:
            return results

        # Score all query-document pairs
        pairs = [(query, r.content) for r in results]
        scores = self.model.predict(pairs)

        # Create new results with updated scores
        reranked = []
        for result, score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    content=result.content,
                    score=float(score),
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
    ):
        self.retriever = retriever
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

    def _needs_retrieval(self, question: str) -> bool:
        """Determine if question requires retrieval."""
        prompt = f"""Determine if answering this question requires looking up specific information 
or if it can be answered from general knowledge.

Question: {question}

Respond with only "RETRIEVE" or "GENERAL"."""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        return "RETRIEVE" in response.choices[0].message.content.upper()

    def _evaluate_relevance(self, question: str, context: str) -> float:
        """Evaluate if retrieved context is relevant."""
        prompt = f"""Rate how relevant this context is for answering the question.
Scale: 0.0 (completely irrelevant) to 1.0 (perfectly relevant)

Question: {question}

Context: {context}

Respond with only a number between 0.0 and 1.0."""

        response = self.llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        try:
            return float(response.choices[0].message.content.strip())
        except ValueError:
            return 0.5

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Execute self-evaluated RAG query."""

        # Check if retrieval is needed
        if not self._needs_retrieval(question):
            # Answer directly without retrieval
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
            )

            return RAGResponse(
                answer=response.choices[0].message.content,
                sources=[],
                confidence=0.8,
                metadata={"retrieval_used": False},
            )

        # Retrieve and evaluate
        results = self.retriever.retrieve(query=question, top_k=top_k)

        if results:
            context = "\n".join(r.content for r in results[:3])
            relevance = self._evaluate_relevance(question, context)

            if relevance < self.confidence_threshold:
                # Context not relevant enough, try broader retrieval
                results = self.retriever.retrieve(
                    query=question, top_k=top_k * 2
                )

        # Generate answer with context
        # ... standard RAG generation

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
from typing import List, Dict, Any, Optional
from datetime import datetime
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
        filter_metadata: Dict[str, Any] = None,
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


class QdrantVectorStore(VectorStore):
    """Qdrant implementation of VectorStore."""

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        dimension: int = 1536,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        if collection_name not in [c.name for c in collections]:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension, distance=Distance.COSINE
                ),
            )

    def upsert(self, vectors: List[Dict[str, Any]]) -> int:
        from qdrant_client.models import PointStruct

        points = [
            PointStruct(
                id=v["id"],
                vector=v["vector"],
                payload={"content": v["content"], **v.get("metadata", {})},
            )
            for v in vectors
        ]

        self.client.upsert(
            collection_name=self.collection_name, points=points
        )

        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_metadata: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        query_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_metadata.items()
            ]
            query_filter = Filter(must=conditions)

        results = self.client.search(
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

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )

        return len(ids)

    def get_stats(self) -> Dict[str, Any]:
        info = self.client.get_collection(self.collection_name)
        return {
            "vector_count": info.points_count,
            "size_bytes": info.payload_schema,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    def get_ids_by_metadata(self, filter: Dict[str, Any]) -> set:
        """Get all IDs matching metadata filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filter.items()
        ]

        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=conditions),
            limit=10000,
            with_payload=False,
        )[0]

        return {str(r.id) for r in results}


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

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        llm_client: LLMClient,
        chunker: DocumentChunker = None,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        self.chunker = chunker or SentenceChunker()

        # Initialize retrievers
        self._documents: Dict[str, KnowledgeDocument] = {}
        self._chunks: List[DocumentChunk] = []

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

        # Generate embeddings
        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed_batch(contents)

        # Prepare vectors for storage
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "vector": embedding,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
            )

        # Store in vector database
        self.vector_store.upsert(vectors)

        # Track locally
        self._documents[document.document_id] = document
        self._chunks.extend(chunks)

        # Rebuild BM25 index
        self._rebuild_keyword_index()

        return len(chunks)

    def _rebuild_keyword_index(self):
        """Rebuild keyword search index after document changes."""
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
        """Import multiple documents efficiently."""
        stats = {"total": len(documents), "chunks": 0, "failed": 0}

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            for doc in batch:
                try:
                    chunks_added = self.add_document(doc)
                    stats["chunks"] += chunks_added
                except Exception as e:
                    stats["failed"] += 1
                    print(f"Failed to import {doc.document_id}: {e}")

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


def create_knowledge_base_example():
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
