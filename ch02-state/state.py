"""
State Management

Code listings from Chapter 02, Book 3:
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


# ============================================================================
# Block 1 (chapter listing #1)
# ============================================================================

import json
import re
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, TypeVar, Generic
import redis
from redis.asyncio import Redis as AsyncRedis

T = TypeVar("T")


@dataclass
class StateMetadata:
    """Metadata attached to all state entries."""

    created_at: datetime
    updated_at: datetime
    version: int
    ttl_seconds: Optional[int] = None
    tags: dict[str, str] = field(default_factory=dict)


class StateStore(ABC, Generic[T]):
    """
    Abstract interface for state storage backends.

    All state stores implement this interface, enabling consistent
    access patterns regardless of the underlying storage technology.
    """

    @abstractmethod
    async def get(self, key: str) -> Optional[T]:
        """Retrieve state by key. Returns None if not found."""
        pass

    @abstractmethod
    async def set(
        self, key: str, value: T, ttl_seconds: Optional[int] = None
    ) -> None:
        """Store state with optional TTL."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete state. Returns True if key existed."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists without retrieving value."""
        pass

    @abstractmethod
    async def get_metadata(self, key: str) -> Optional[StateMetadata]:
        """Retrieve metadata for a state entry."""
        pass


class RedisStateStore(StateStore[dict[str, Any]]):
    """
    Redis-backed state store for hot data.

    Suitable for conversation state and other frequently accessed
    data where sub-millisecond latency is required.
    """

    def __init__(
        self,
        redis_client: AsyncRedis,
        key_prefix: str = "state:",
        default_ttl: int = 3600,
    ):
        self._redis = redis_client
        self._prefix = key_prefix
        self._default_ttl = default_ttl

    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _make_meta_key(self, key: str) -> str:
        return f"{self._prefix}meta:{key}"

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        data = await self._redis.get(self._make_key(key))
        if data is None:
            return None
        return json.loads(data)

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        ttl = ttl_seconds or self._default_ttl
        full_key = self._make_key(key)
        meta_key = self._make_meta_key(key)

        # Get existing metadata or create new
        existing_meta = await self.get_metadata(key)
        now = datetime.now(timezone.utc)

        if existing_meta:
            meta = StateMetadata(
                created_at=existing_meta.created_at,
                updated_at=now,
                version=existing_meta.version + 1,
                ttl_seconds=ttl,
            )
        else:
            meta = StateMetadata(
                created_at=now, updated_at=now, version=1, ttl_seconds=ttl
            )

        # Use pipeline for atomic update
        pipe = self._redis.pipeline()
        pipe.setex(full_key, ttl, json.dumps(value))
        pipe.setex(
            meta_key,
            ttl,
            json.dumps(
                {
                    "created_at": meta.created_at.isoformat(),
                    "updated_at": meta.updated_at.isoformat(),
                    "version": meta.version,
                    "ttl_seconds": meta.ttl_seconds,
                }
            ),
        )
        await pipe.execute()

    async def delete(self, key: str) -> bool:
        result = await self._redis.delete(
            self._make_key(key), self._make_meta_key(key)
        )
        return result > 0

    async def exists(self, key: str) -> bool:
        return await self._redis.exists(self._make_key(key)) > 0

    async def get_metadata(self, key: str) -> Optional[StateMetadata]:
        data = await self._redis.get(self._make_meta_key(key))
        if data is None:
            return None
        parsed = json.loads(data)
        return StateMetadata(
            created_at=datetime.fromisoformat(parsed["created_at"]),
            updated_at=datetime.fromisoformat(parsed["updated_at"]),
            version=parsed["version"],
            ttl_seconds=parsed.get("ttl_seconds"),
        )

    # ----- Low-level helpers used by coordination primitives -----
    # These public methods give cooperating components (e.g. lock
    # coordinators) controlled access to the underlying client without
    # reaching into the private ``_redis`` attribute.
    async def set_raw(
        self,
        key: str,
        value: str,
        nx: bool = False,
        ex: Optional[int] = None,
    ) -> bool:
        return bool(
            await self._redis.set(key, value, nx=nx, ex=ex)
        )

    async def get_raw(self, key: str) -> Optional[str]:
        return await self._redis.get(key)

    async def delete_raw(self, key: str) -> int:
        return await self._redis.delete(key)


# ============================================================================
# Block 2 (chapter listing #2)
# ============================================================================

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import asyncpg
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskState:
    """Represents the complete state of a task."""

    task_id: str
    task_type: str
    status: TaskStatus
    payload: dict[str, Any]
    checkpoint: Optional[dict[str, Any]]
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]


class PostgresTaskStore:
    """
    PostgreSQL-backed store for task state.

    Provides ACID guarantees for task operations, supporting
    reliable checkpointing and recovery.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS task_state (
        task_id VARCHAR(255) PRIMARY KEY,
        task_type VARCHAR(100) NOT NULL,
        status VARCHAR(50) NOT NULL,
        payload JSONB NOT NULL,
        checkpoint JSONB,
        attempt_count INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        completed_at TIMESTAMP WITH TIME ZONE,
        error_message TEXT,
        
        -- Indexes for common queries
        CONSTRAINT valid_status CHECK (
            status IN ('pending', 'running', 'paused', 'completed', 'failed')
        )
    );
    
    CREATE INDEX IF NOT EXISTS idx_task_status 
        ON task_state(status);
    CREATE INDEX IF NOT EXISTS idx_task_type_status 
        ON task_state(task_type, status);
    CREATE INDEX IF NOT EXISTS idx_task_created 
        ON task_state(created_at);
    """

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def create(cls, dsn: str) -> "PostgresTaskStore":
        """Factory method that initializes the schema."""
        # Explicit pool sizing + per-command timeout. Defaults give
        # an unbounded pool with no statement timeout, which can let
        # a single slow query hold a connection forever and starve
        # the rest of the agent under load.
        pool = await asyncpg.create_pool(
            dsn,
            min_size=5,
            max_size=20,
            command_timeout=10,
        )
        async with pool.acquire() as conn:
            await conn.execute(cls.SCHEMA)
        return cls(pool)

    async def create_task(
        self, task_id: str, task_type: str, payload: dict[str, Any]
    ) -> TaskState:
        """Create a new task in pending state."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO task_state (task_id, task_type, status, payload)
                VALUES ($1, $2, 'pending', $3)
                RETURNING *
                """,
                task_id,
                task_type,
                json.dumps(payload),
            )
            return self._row_to_task(row)

    async def get_task(self, task_id: str) -> Optional[TaskState]:
        """Retrieve task by ID."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM task_state WHERE task_id = $1", task_id
            )
            return self._row_to_task(row) if row else None

    async def update_checkpoint(
        self,
        task_id: str,
        checkpoint: dict[str, Any],
        expected_version: Optional[int] = None,
    ) -> bool:
        """
        Update task checkpoint with optimistic locking.

        If expected_version is provided, update only succeeds if
        the current attempt_count matches. This prevents lost updates
        when multiple workers process the same task.
        """
        async with self._pool.acquire() as conn:
            if expected_version is not None:
                result = await conn.execute(
                    """
                    UPDATE task_state 
                    SET checkpoint = $2, 
                        updated_at = NOW(),
                        attempt_count = attempt_count + 1
                    WHERE task_id = $1 
                        AND attempt_count = $3
                    """,
                    task_id,
                    json.dumps(checkpoint),
                    expected_version,
                )
            else:
                result = await conn.execute(
                    """
                    UPDATE task_state 
                    SET checkpoint = $2, updated_at = NOW()
                    WHERE task_id = $1
                    """,
                    task_id,
                    json.dumps(checkpoint),
                )
            return result == "UPDATE 1"

    async def transition_status(
        self,
        task_id: str,
        new_status: TaskStatus,
        error_message: Optional[str] = None,
    ) -> bool:
        """Transition task to new status.

        ``completed_at`` is bound as a parameter rather than spliced
        into the SQL string. The previous f-string variant was safe
        because ``new_status`` was always an enum value, but a future
        refactor that lets the status come from outside the enum
        would turn this into a SQL-injection vector. Parameterizing
        eliminates the class of bug entirely.
        """
        completed_at = (
            datetime.now(timezone.utc)
            if new_status == TaskStatus.COMPLETED
            else None
        )
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE task_state
                SET status = $2,
                    updated_at = NOW(),
                    completed_at = $4,
                    error_message = $3
                WHERE task_id = $1
                """,
                task_id,
                new_status.value,
                error_message,
                completed_at,
            )
            return result == "UPDATE 1"

    async def claim_pending_tasks(
        self, task_type: str, worker_id: str, limit: int = 10
    ) -> list[TaskState]:
        """
        Atomically claim pending tasks for processing.

        Uses SELECT FOR UPDATE SKIP LOCKED to enable multiple
        workers to claim tasks without conflicts.
        """
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(
                    """
                    UPDATE task_state
                    SET status = 'running',
                        updated_at = NOW(),
                        attempt_count = attempt_count + 1
                    WHERE task_id IN (
                        SELECT task_id FROM task_state
                        WHERE task_type = $1 AND status = 'pending'
                        ORDER BY created_at
                        LIMIT $2
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING *
                    """,
                    task_type,
                    limit,
                )
                return [self._row_to_task(row) for row in rows]

    def _row_to_task(self, row: asyncpg.Record) -> TaskState:
        return TaskState(
            task_id=row["task_id"],
            task_type=row["task_type"],
            status=TaskStatus(row["status"]),
            payload=json.loads(row["payload"]) if row["payload"] else {},
            checkpoint=(
                json.loads(row["checkpoint"]) if row["checkpoint"] else None
            ),
            attempt_count=row["attempt_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
            error_message=row["error_message"],
        )

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

import asyncio

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from decimal import Decimal


class DynamoDBStateStore(StateStore[dict[str, Any]]):
    """
    DynamoDB-backed state store for serverless environments.

    Provides automatic scaling and pay-per-request pricing,
    suitable for variable workloads.

    Implementation note: the official ``boto3`` client is synchronous.
    Calling it directly from an ``async def`` would block the event
    loop. We wrap each boto3 call with ``asyncio.to_thread(...)`` so
    the synchronous I/O runs on the default thread pool. For
    throughput-critical workloads, switch to ``aioboto3`` (or
    ``aiobotocore``) which provides native async clients.
    """

    def __init__(
        self,
        table_name: str,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
    ):
        self._table_name = table_name
        # Explicit production config: bound the per-request budget,
        # cap the connection pool, and use the SDK's adaptive retry
        # mode (exponential backoff with throttling-aware
        # rescheduling). The boto3 defaults are 60 s connect + 60 s
        # read and unbounded pooling, which silently lets a flaky
        # DynamoDB partition consume every async worker.
        from botocore.config import Config as BotoConfig
        boto_config = BotoConfig(
            connect_timeout=2,
            read_timeout=5,
            max_pool_connections=50,
            retries={"max_attempts": 4, "mode": "adaptive"},
        )
        self._dynamodb = boto3.resource(
            "dynamodb",
            region_name=region,
            endpoint_url=endpoint_url,
            config=boto_config,
        )
        self._table = self._dynamodb.Table(table_name)

    async def get(self, key: str) -> Optional[dict[str, Any]]:
        try:
            response = await asyncio.to_thread(
                self._table.get_item, Key={"pk": key}
            )
        except ClientError as exc:
            # Distinguish "not found" (legitimate cache miss) from
            # "DynamoDB call failed" (operational signal). Re-raising
            # surfaces throttling, network blips, and credential
            # issues to the caller's retry/escalation path instead of
            # silently degrading every read to a None result.
            import logging
            logging.getLogger(__name__).warning(
                "DynamoDB get_item failed for key=%s: %s", key, exc
            )
            raise
        item = response.get("Item")
        if item is None:
            return None
        # Convert Decimals back to native types
        return self._deserialize(item.get("data", {}))

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        item = {
            "pk": key,
            "data": self._serialize(value),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Get existing item for version tracking
        existing = await self.get_metadata(key)
        item["version"] = (existing.version + 1) if existing else 1
        item["created_at"] = (
            existing.created_at.isoformat()
            if existing
            else datetime.now(timezone.utc).isoformat()
        )

        if ttl_seconds:
            item["ttl"] = (
                int(datetime.now(timezone.utc).timestamp()) + ttl_seconds
            )

        await asyncio.to_thread(self._table.put_item, Item=item)

    async def delete(self, key: str) -> bool:
        try:
            await asyncio.to_thread(
                self._table.delete_item,
                Key={"pk": key},
                ConditionExpression=Attr("pk").exists(),
            )
            return True
        except ClientError as e:
            if (
                e.response["Error"]["Code"]
                == "ConditionalCheckFailedException"
            ):
                return False
            raise

    async def exists(self, key: str) -> bool:
        response = await asyncio.to_thread(
            self._table.get_item,
            Key={"pk": key},
            ProjectionExpression="pk",
        )
        return "Item" in response

    async def get_metadata(self, key: str) -> Optional[StateMetadata]:
        response = await asyncio.to_thread(
            self._table.get_item,
            Key={"pk": key},
            ProjectionExpression="created_at, updated_at, version, ttl",
        )
        item = response.get("Item")
        if not item:
            return None
        return StateMetadata(
            created_at=datetime.fromisoformat(item["created_at"]),
            updated_at=datetime.fromisoformat(item["updated_at"]),
            version=int(item["version"]),
            ttl_seconds=item.get("ttl"),
        )

    def _serialize(self, data: dict) -> dict:
        """Convert floats to Decimal for DynamoDB compatibility."""
        if isinstance(data, dict):
            return {k: self._serialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._serialize(v) for v in data]
        elif isinstance(data, float):
            return Decimal(str(data))
        return data

    def _deserialize(self, data: dict) -> dict:
        """Convert Decimals back to native Python types."""
        if isinstance(data, dict):
            return {k: self._deserialize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize(v) for v in data]
        elif isinstance(data, Decimal):
            if data % 1 == 0:
                return int(data)
            return float(data)
        return data

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

from dataclasses import dataclass
from typing import Protocol
import tiktoken


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # "user", "assistant", or "system"
    content: str
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: Optional[int] = None

    def __post_init__(self):
        if self.token_count is None:
            # Lazy token counting
            encoder = tiktoken.get_encoding("cl100k_base")
            self.token_count = len(encoder.encode(self.content))


@dataclass
class ConversationWindow:
    """A window of conversation history with token budget."""

    messages: list[Message]
    summary: Optional[str]
    total_tokens: int
    truncated: bool


class SummarizerProtocol(Protocol):
    """Protocol for message summarization."""

    async def summarize(self, messages: list[Message]) -> str:
        """Summarize a list of messages into a concise summary."""
        ...


class ConversationManager:
    """
    Manages conversation history with token-aware truncation.

    Maintains conversation context within token limits using
    a combination of sliding windows and summarization.
    """

    def __init__(
        self,
        state_store: StateStore[dict[str, Any]],
        summarizer: SummarizerProtocol,
        max_tokens: int = 8000,
        summary_threshold: int = 4000,
        min_recent_messages: int = 10,
    ):
        """
        Initialize conversation manager.

        Args:
            state_store: Backend for persisting conversations
            summarizer: Service for summarizing old messages
            max_tokens: Maximum tokens in context window
            summary_threshold: Trigger summarization above this
            min_recent_messages: Always keep this many recent messages
        """
        self._store = state_store
        self._summarizer = summarizer
        self._max_tokens = max_tokens
        self._summary_threshold = summary_threshold
        self._min_recent = min_recent_messages
        self._encoder = tiktoken.get_encoding("cl100k_base")

    async def add_message(
        self, conversation_id: str, message: Message
    ) -> None:
        """Add a message and trigger compaction if needed."""
        state = await self._load_state(conversation_id)
        state["messages"].append(self._message_to_dict(message))
        state["total_tokens"] += message.token_count

        # Check if compaction needed
        if state["total_tokens"] > self._summary_threshold:
            state = await self._compact(state)

        await self._store.set(f"conversation:{conversation_id}", state)

    async def get_context_window(
        self, conversation_id: str, max_tokens: Optional[int] = None
    ) -> ConversationWindow:
        """
        Get conversation context optimized for LLM input.

        Returns a window containing recent messages and optionally
        a summary of older context, fitted within token budget.
        """
        budget = max_tokens or self._max_tokens
        state = await self._load_state(conversation_id)

        messages = [self._dict_to_message(m) for m in state["messages"]]
        summary = state.get("summary")

        # Calculate tokens for summary
        summary_tokens = 0
        if summary:
            summary_tokens = len(self._encoder.encode(summary)) + 50  # Buffer

        available_for_messages = budget - summary_tokens

        # Select messages that fit in budget, prioritizing recent
        selected = []
        tokens_used = 0

        for message in reversed(messages):
            if tokens_used + message.token_count <= available_for_messages:
                selected.insert(0, message)
                tokens_used += message.token_count
            elif len(selected) < self._min_recent:
                # Force include minimum recent messages
                selected.insert(0, message)
                tokens_used += message.token_count
            else:
                break

        return ConversationWindow(
            messages=selected,
            summary=summary,
            total_tokens=tokens_used + summary_tokens,
            truncated=len(selected) < len(messages),
        )

    async def _compact(self, state: dict) -> dict:
        """Compact conversation by summarizing older messages."""
        messages = [self._dict_to_message(m) for m in state["messages"]]

        # Keep recent messages, summarize the rest
        keep_count = max(self._min_recent, len(messages) // 3)
        to_summarize = messages[:-keep_count]
        to_keep = messages[-keep_count:]

        if not to_summarize:
            return state

        # Include existing summary in new summarization
        if state.get("summary"):
            summary_context = f"Previous context: {state['summary']}\n\n"
            summary_message = Message(
                role="system",
                content=summary_context,
                timestamp=to_summarize[0].timestamp,
            )
            to_summarize = [summary_message] + to_summarize

        new_summary = await self._summarizer.summarize(to_summarize)

        return {
            "messages": [self._message_to_dict(m) for m in to_keep],
            "summary": new_summary,
            "total_tokens": sum(m.token_count for m in to_keep),
            "compaction_count": state.get("compaction_count", 0) + 1,
        }

    async def _load_state(self, conversation_id: str) -> dict:
        state = await self._store.get(f"conversation:{conversation_id}")
        if state is None:
            return {
                "messages": [],
                "summary": None,
                "total_tokens": 0,
                "compaction_count": 0,
            }
        return state

    def _message_to_dict(self, msg: Message) -> dict:
        return {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": msg.metadata,
            "token_count": msg.token_count,
        }

    def _dict_to_message(self, data: dict) -> Message:
        return Message(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
            token_count=data.get("token_count"),
        )

# ============================================================================
# Block 5 (chapter listing #5)
# ============================================================================

class ImportanceScorer:
    """Score messages by importance for retention decisions."""

    def __init__(self):
        self._high_importance_patterns = [
            r"(?i)important",
            r"(?i)remember",
            r"(?i)critical",
            r"(?i)must\s+(?:have|do|be)",
            r"(?i)requirement",
        ]
        self._low_importance_patterns = [
            r"(?i)^(ok|okay|thanks|thank you|got it)\.?$",
            r"(?i)^(yes|no|sure)\.?$",
        ]

    def score(self, message: Message, context: list[Message]) -> float:
        """
        Score message importance from 0.0 to 1.0.

        Higher scores indicate messages that should be retained
        longer during compaction.
        """
        score = 0.5  # Base score

        # Boost for high-importance patterns
        for pattern in self._high_importance_patterns:
            if re.search(pattern, message.content):
                score += 0.2

        # Reduce for low-importance patterns
        for pattern in self._low_importance_patterns:
            if re.search(pattern, message.content):
                score -= 0.3

        # Boost for messages with tool calls (indicates action)
        if message.metadata.get("tool_calls"):
            score += 0.15

        # Boost for longer, substantive messages
        if message.token_count > 100:
            score += 0.1

        # Boost for messages referenced by later messages
        if self._is_referenced(message, context):
            score += 0.25

        return max(0.0, min(1.0, score))

    def _is_referenced(
        self, message: Message, context: list[Message]
    ) -> bool:
        """Check if later messages reference this one."""
        # Simplified check - production would use semantic similarity
        key_phrases = self._extract_key_phrases(message.content)
        for later_msg in context:
            if later_msg.timestamp <= message.timestamp:
                continue
            for phrase in key_phrases:
                if phrase.lower() in later_msg.content.lower():
                    return True
        return False

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases for reference checking."""
        # Simple extraction - use NLP library in production
        words = text.split()
        return [
            " ".join(words[i : i + 3]) for i in range(0, len(words) - 2, 3)
        ]

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import asyncio
import hashlib


@dataclass
class Checkpoint:
    """Represents a task checkpoint."""

    task_id: str
    step_id: str
    state: dict[str, Any]
    created_at: datetime
    checksum: str

    @classmethod
    def create(
        cls, task_id: str, step_id: str, state: dict[str, Any]
    ) -> "Checkpoint":
        state_json = json.dumps(state, sort_keys=True)
        checksum = hashlib.sha256(state_json.encode()).hexdigest()[:16]
        return cls(
            task_id=task_id,
            step_id=step_id,
            state=state,
            created_at=datetime.now(timezone.utc),
            checksum=checksum,
        )

    def verify(self) -> bool:
        """Verify checkpoint integrity."""
        state_json = json.dumps(self.state, sort_keys=True)
        expected = hashlib.sha256(state_json.encode()).hexdigest()[:16]
        return expected == self.checksum


class TaskCheckpointer:
    """
    Manages checkpoints for resumable long-running tasks.

    Provides automatic checkpointing with configurable frequency,
    integrity verification, and resume capability.
    """

    def __init__(
        self,
        task_store: PostgresTaskStore,
        checkpoint_interval: int = 30,  # seconds
        max_checkpoints_per_task: int = 10,
        max_pending: int = 1000,
    ):
        self._store = task_store
        self._interval = checkpoint_interval
        self._max_checkpoints = max_checkpoints_per_task
        # Cap on how many checkpoints _flush_all writes in one batch.
        # Excess entries stay queued and flush in the next iteration,
        # so a stalled backend cannot let memory grow without bound.
        self._max_pending = max_pending
        self._pending_checkpoints: dict[str, Checkpoint] = {}
        self._checkpoint_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start background checkpoint flushing."""
        self._checkpoint_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Stop checkpointing and flush pending."""
        if self._checkpoint_task:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush_all()

    @asynccontextmanager
    async def checkpointed_task(
        self, task_id: str, task_type: str, payload: dict[str, Any]
    ) -> AsyncGenerator["CheckpointContext", None]:
        """
        Context manager for checkpointed task execution.

        Usage:
            async with checkpointer.checkpointed_task(
                task_id, task_type, payload
            ) as ctx:
                for item in items:
                    result = await process(item)
                    await ctx.checkpoint({"processed": item, "result": result})
        """
        # Check for existing task and resume point
        existing = await self._store.get_task(task_id)

        if existing and existing.checkpoint:
            if not Checkpoint(
                task_id=existing.task_id,
                step_id="resume",
                state=existing.checkpoint,
                created_at=existing.updated_at,
                checksum=existing.checkpoint.get("_checksum", ""),
            ).verify():
                raise StateCorruptionError(
                    f"Checkpoint verification failed for task {task_id}"
                )
            resume_state = existing.checkpoint
        else:
            if not existing:
                await self._store.create_task(task_id, task_type, payload)
            resume_state = None

        await self._store.transition_status(task_id, TaskStatus.RUNNING)

        ctx = CheckpointContext(
            task_id=task_id, checkpointer=self, resume_state=resume_state
        )

        try:
            yield ctx
            await self._store.transition_status(task_id, TaskStatus.COMPLETED)
        except Exception as e:
            await self._flush_task(task_id)  # Ensure last checkpoint saved
            await self._store.transition_status(
                task_id, TaskStatus.FAILED, error_message=str(e)
            )
            raise

    async def checkpoint(
        self, task_id: str, step_id: str, state: dict[str, Any]
    ) -> None:
        """
        Record a checkpoint for a task.

        Checkpoints are buffered and flushed periodically to reduce
        database writes while maintaining reasonable recovery points.
        """
        checkpoint = Checkpoint.create(task_id, step_id, state)
        # Include checksum in state for later verification
        checkpoint.state["_checksum"] = checkpoint.checksum
        checkpoint.state["_step_id"] = step_id
        self._pending_checkpoints[task_id] = checkpoint

    async def get_resume_point(
        self, task_id: str
    ) -> Optional[dict[str, Any]]:
        """Get the last checkpoint for resuming a task."""
        task = await self._store.get_task(task_id)
        if task and task.checkpoint:
            return task.checkpoint
        return None

    async def _flush_loop(self) -> None:
        """Background loop to flush pending checkpoints.

        Catches per-flush exceptions so one failure doesn't silently
        kill the loop and stop all subsequent checkpointing. Each
        failure is logged with a counter; the loop continues until
        the task is cancelled.
        """
        _log = __import__("logging").getLogger(__name__)
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._flush_all()
            except asyncio.CancelledError:
                # Cooperative shutdown: re-raise so the caller's
                # ``stop()`` semantics still work.
                raise
            except Exception as exc:  # noqa: BLE001
                _log.exception(
                    "Checkpoint flush failed; continuing loop: %s", exc
                )

    async def _flush_all(self) -> None:
        """Flush all pending checkpoints to storage.

        Caps the in-flight batch at ``_max_pending`` so a long backend
        stall cannot let pending checkpoints grow without bound between
        flushes. Excess entries are kept for the next iteration (FIFO
        by insertion order in the underlying dict).
        """
        # Take a bounded slice; leave excess for the next flush.
        pending_items = list(self._pending_checkpoints.items())
        batch = pending_items[: self._max_pending]
        # Remove only what we are about to flush; entries that arrive
        # mid-flush stay queued for the next interval.
        for task_id, _ in batch:
            self._pending_checkpoints.pop(task_id, None)

        for task_id, checkpoint in batch:
            await self._store.update_checkpoint(task_id, checkpoint.state)

    async def _flush_task(self, task_id: str) -> None:
        """Flush checkpoint for specific task."""
        if task_id in self._pending_checkpoints:
            checkpoint = self._pending_checkpoints.pop(task_id)
            await self._store.update_checkpoint(task_id, checkpoint.state)


@dataclass
class CheckpointContext:
    """Context for checkpointed task execution."""

    task_id: str
    checkpointer: TaskCheckpointer
    resume_state: Optional[dict[str, Any]]

    async def checkpoint(
        self, state: dict[str, Any], step_id: Optional[str] = None
    ) -> None:
        """Save a checkpoint with current state."""
        await self.checkpointer.checkpoint(
            self.task_id,
            step_id or f"step_{datetime.now(timezone.utc).timestamp()}",
            state,
        )

    @property
    def is_resume(self) -> bool:
        """Check if this is a resumed execution."""
        return self.resume_state is not None

    def get_resume_value(self, key: str, default: Any = None) -> Any:
        """Get a value from resume state."""
        if self.resume_state:
            return self.resume_state.get(key, default)
        return default


class StateCorruptionError(Exception):
    """Raised when state integrity check fails."""

    pass

# ============================================================================
# Block 7 (chapter listing #7)
# ============================================================================

class CheckpointStrategy(ABC):
    """Base class for checkpoint strategies."""

    @abstractmethod
    def should_checkpoint(
        self,
        last_checkpoint: Optional[datetime],
        items_processed: int,
        current_state: dict[str, Any],
    ) -> bool:
        pass


class TimeBasedStrategy(CheckpointStrategy):
    """Checkpoint at fixed time intervals."""

    def __init__(self, interval_seconds: int = 60):
        self._interval = timedelta(seconds=interval_seconds)

    def should_checkpoint(
        self,
        last_checkpoint: Optional[datetime],
        items_processed: int,
        current_state: dict[str, Any],
    ) -> bool:
        if last_checkpoint is None:
            return True
        return datetime.now(timezone.utc) - last_checkpoint >= self._interval


class CountBasedStrategy(CheckpointStrategy):
    """Checkpoint after processing N items."""

    def __init__(self, item_count: int = 100):
        self._count = item_count
        self._since_last = 0

    def should_checkpoint(
        self,
        last_checkpoint: Optional[datetime],
        items_processed: int,
        current_state: dict[str, Any],
    ) -> bool:
        self._since_last += 1
        if self._since_last >= self._count:
            self._since_last = 0
            return True
        return False


class AdaptiveStrategy(CheckpointStrategy):
    """
    Adaptive checkpointing based on processing cost.

    Checkpoints more frequently when operations are expensive
    (to minimize re-work on failure) and less frequently when
    operations are cheap (to minimize checkpoint overhead).
    """

    def __init__(
        self,
        min_interval: int = 10,
        max_interval: int = 300,
        cost_threshold: float = 1.0,
    ):
        self._min = timedelta(seconds=min_interval)
        self._max = timedelta(seconds=max_interval)
        self._threshold = cost_threshold
        # Bounded deque keeps the last 100 costs without manual trimming.
        self._recent_costs: deque[float] = deque(maxlen=100)

    def record_operation_cost(self, cost: float) -> None:
        """Record the cost of a recent operation."""
        self._recent_costs.append(cost)

    def should_checkpoint(
        self,
        last_checkpoint: Optional[datetime],
        items_processed: int,
        current_state: dict[str, Any],
    ) -> bool:
        if last_checkpoint is None:
            return True

        elapsed = datetime.now(timezone.utc) - last_checkpoint

        if elapsed < self._min:
            return False
        if elapsed >= self._max:
            return True

        # Calculate adaptive interval based on recent costs
        if self._recent_costs:
            avg_cost = sum(self._recent_costs) / len(self._recent_costs)
            if avg_cost > self._threshold:
                # High cost - checkpoint more frequently
                target = self._min + (self._max - self._min) * 0.25
            else:
                # Low cost - checkpoint less frequently
                target = self._min + (self._max - self._min) * 0.75
            return elapsed >= target

        return elapsed >= (self._min + self._max) / 2

# ============================================================================
# Block 8 (chapter listing #8)
# ============================================================================

@dataclass
class RecoveryResult:
    """Result of a recovery operation."""

    task_id: str
    recovered: bool
    resume_point: Optional[dict[str, Any]]
    data_loss: bool
    error: Optional[str]


class RecoveryCoordinator:
    """
    Coordinates state recovery after system failures.

    Handles detection of incomplete tasks, validation of
    recovery points, and orchestration of resume operations.
    """

    def __init__(
        self,
        task_store: PostgresTaskStore,
        conversation_manager: ConversationManager,
        max_recovery_attempts: int = 3,
    ):
        self._tasks = task_store
        self._conversations = conversation_manager
        self._max_attempts = max_recovery_attempts

    async def recover_interrupted_tasks(
        self, worker_id: str, task_types: Optional[list[str]] = None
    ) -> list[RecoveryResult]:
        """
        Recover all interrupted tasks for a worker.

        Called during worker startup to resume any tasks that
        were interrupted by the previous shutdown or crash.
        """
        results = []

        # Find tasks in running state that need recovery
        interrupted = await self._find_interrupted_tasks(task_types)

        for task in interrupted:
            result = await self._recover_task(task)
            results.append(result)

        return results

    async def _find_interrupted_tasks(
        self, task_types: Optional[list[str]]
    ) -> list[TaskState]:
        """Find tasks that were interrupted mid-execution."""
        async with self._tasks._pool.acquire() as conn:
            if task_types:
                rows = await conn.fetch(
                    """
                    SELECT * FROM task_state
                    WHERE status = 'running'
                    AND task_type = ANY($1)
                    AND updated_at < NOW() - INTERVAL '5 minutes'
                    ORDER BY updated_at
                    """,
                    task_types,
                )
            else:
                rows = await conn.fetch("""
                    SELECT * FROM task_state
                    WHERE status = 'running'
                    AND updated_at < NOW() - INTERVAL '5 minutes'
                    ORDER BY updated_at
                    """)
            return [self._tasks._row_to_task(r) for r in rows]

    async def _recover_task(self, task: TaskState) -> RecoveryResult:
        """Attempt to recover a single interrupted task."""
        # Check attempt count
        if task.attempt_count >= self._max_attempts:
            await self._tasks.transition_status(
                task.task_id,
                TaskStatus.FAILED,
                f"Max recovery attempts ({self._max_attempts}) exceeded",
            )
            return RecoveryResult(
                task_id=task.task_id,
                recovered=False,
                resume_point=None,
                data_loss=True,
                error="Max recovery attempts exceeded",
            )

        # Validate checkpoint if present
        if task.checkpoint:
            checkpoint = Checkpoint(
                task_id=task.task_id,
                step_id=task.checkpoint.get("_step_id", "unknown"),
                state=task.checkpoint,
                created_at=task.updated_at,
                checksum=task.checkpoint.get("_checksum", ""),
            )

            if not checkpoint.verify():
                # Checkpoint corrupted - try to recover from backup
                backup = await self._find_backup_checkpoint(task.task_id)
                if backup:
                    task.checkpoint = backup
                else:
                    return RecoveryResult(
                        task_id=task.task_id,
                        recovered=False,
                        resume_point=None,
                        data_loss=True,
                        error="Checkpoint corruption, no backup available",
                    )

        # Reset to pending for re-processing
        await self._tasks.transition_status(task.task_id, TaskStatus.PENDING)

        return RecoveryResult(
            task_id=task.task_id,
            recovered=True,
            resume_point=task.checkpoint,
            data_loss=False,
            error=None,
        )

    async def _find_backup_checkpoint(
        self, task_id: str
    ) -> Optional[dict[str, Any]]:
        """
        Attempt to find a backup checkpoint.

        In production, this might check a separate backup table,
        a different storage backend, or transaction logs.
        """
        # This is a simplified implementation
        # Production would have more sophisticated backup mechanisms
        return None

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

class CorruptionHandler:
    """
    Handles detection and recovery from state corruption.

    Implements multiple recovery strategies with fallback chains.
    """

    def __init__(
        self,
        primary_store: StateStore,
        backup_store: Optional[StateStore] = None,
        metrics_client: Optional[Any] = None,
    ):
        self._primary = primary_store
        self._backup = backup_store
        self._metrics = metrics_client

    async def get_with_validation(
        self, key: str, validator: Callable[[dict], bool]
    ) -> Optional[dict[str, Any]]:
        """
        Get state with validation, falling back to backup if invalid.
        """
        # Try primary store
        state = await self._primary.get(key)

        if state and validator(state):
            return state

        if state:
            self._record_corruption(key, "validation_failed")

        # Try backup if available
        if self._backup:
            backup_state = await self._backup.get(key)
            if backup_state and validator(backup_state):
                # Restore to primary
                await self._primary.set(key, backup_state)
                self._record_corruption(key, "recovered_from_backup")
                return backup_state

        return None

    async def repair_state(
        self,
        key: str,
        current_state: dict[str, Any],
        repair_strategy: str = "reconstruct",
    ) -> Optional[dict[str, Any]]:
        """
        Attempt to repair corrupted state.

        Strategies:
        - reconstruct: Rebuild from source data
        - rollback: Use previous version
        - partial: Salvage valid portions
        """
        if repair_strategy == "rollback":
            return await self._rollback_state(key)
        elif repair_strategy == "partial":
            return await self._partial_recovery(key, current_state)
        elif repair_strategy == "reconstruct":
            return await self._reconstruct_state(key)
        else:
            raise ValueError(f"Unknown repair strategy: {repair_strategy}")

    async def _rollback_state(self, key: str) -> Optional[dict[str, Any]]:
        """Roll back to previous version if available.

        This is a placeholder: real rollback requires a version-history
        table (or a versioned store) that is out of scope for this
        chapter's primary KV backend. Subclasses backed by such a
        store should override.
        """
        metadata = await self._primary.get_metadata(key)
        if metadata and metadata.version > 1:
            raise NotImplementedError(
                "Version-aware rollback requires a history-backed store; "
                "override _rollback_state in a subclass."
            )
        return None

    async def _partial_recovery(
        self, key: str, corrupted: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        """Salvage valid portions of corrupted state."""
        recovered = {}

        for field_name, value in corrupted.items():
            if self._is_valid_field(field_name, value):
                recovered[field_name] = value

        if recovered:
            # Mark as partially recovered
            recovered["_recovery_mode"] = "partial"
            recovered["_recovery_timestamp"] = datetime.now(
                timezone.utc
            ).isoformat()
            return recovered

        return None

    async def _reconstruct_state(self, key: str) -> Optional[dict[str, Any]]:
        """
        Reconstruct state from source data.

        This is application-specific and would typically involve
        replaying events or re-fetching from external sources.
        """
        # Implementation depends on application architecture
        return None

    def _is_valid_field(self, field: str, value: Any) -> bool:
        """Check if a single field value appears valid."""
        # Skip internal fields
        if field.startswith("_"):
            return True

        # Check for common corruption patterns
        if value is None:
            return True  # None might be valid

        if isinstance(value, str):
            # Check for encoding issues
            try:
                value.encode("utf-8")
            except UnicodeError:
                return False

        return True

    def _record_corruption(self, key: str, corruption_type: str) -> None:
        """Record corruption incident for monitoring."""
        if self._metrics:
            self._metrics.increment(
                "state.corruption",
                tags={
                    "key_prefix": key.split(":")[0],
                    "type": corruption_type,
                },
            )

# ============================================================================
# Block 10 (chapter listing #10)
# ============================================================================

import random
import uuid
from contextlib import asynccontextmanager


@dataclass
class Lock:
    """Represents a distributed lock."""

    resource_id: str
    owner_id: str
    acquired_at: datetime
    expires_at: datetime

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at


class SharedStateCoordinator:
    """
    Coordinates shared state access among multiple agents.

    Provides distributed locking, atomic operations, and
    conflict resolution for multi-agent scenarios.
    """

    def __init__(
        self,
        state_store: RedisStateStore,
        lock_timeout: int = 30,
        retry_delay: float = 0.1,
    ):
        self._store = state_store
        self._lock_timeout = lock_timeout
        self._retry_delay = retry_delay
        self._agent_id = str(uuid.uuid4())

    @asynccontextmanager
    async def locked(
        self, resource_id: str, timeout: Optional[int] = None
    ) -> AsyncGenerator[None, None]:
        """
        Context manager for exclusive access to a resource.

        Usage:
            async with coordinator.locked("resource:123"):
                # Exclusive access to resource
                state = await coordinator.read("resource:123")
                state["count"] += 1
                await coordinator.write("resource:123", state)
        """
        lock = await self.acquire_lock(resource_id, timeout)
        try:
            yield
        finally:
            await self.release_lock(resource_id, lock.owner_id)

    async def acquire_lock(
        self,
        resource_id: str,
        timeout: Optional[int] = None,
        max_wait: float = 10.0,
    ) -> Lock:
        """
        Acquire exclusive lock on a resource.

        Implements exponential backoff with jitter for contention handling.
        """
        timeout = timeout or self._lock_timeout
        lock_key = f"lock:{resource_id}"
        owner_id = f"{self._agent_id}:{uuid.uuid4()}"

        start_time = datetime.now(timezone.utc)
        wait_time = self._retry_delay

        while True:
            # Try to acquire lock with NX (only if not exists)
            now = datetime.now(timezone.utc)
            expires = now + timedelta(seconds=timeout)

            lock_value = json.dumps(
                {
                    "owner_id": owner_id,
                    "acquired_at": now.isoformat(),
                    "expires_at": expires.isoformat(),
                }
            )

            acquired = await self._store.set_raw(
                lock_key, lock_value, nx=True, ex=timeout
            )

            if acquired:
                return Lock(
                    resource_id=resource_id,
                    owner_id=owner_id,
                    acquired_at=now,
                    expires_at=expires,
                )

            # Check if existing lock is expired
            existing = await self._store.get_raw(lock_key)
            if existing:
                lock_data = json.loads(existing)
                existing_expires = datetime.fromisoformat(
                    lock_data["expires_at"]
                )
                if datetime.now(timezone.utc) > existing_expires:
                    # Lock expired, try to take over
                    await self._store.delete_raw(lock_key)
                    continue

            # Check timeout
            elapsed = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds()
            if elapsed >= max_wait:
                raise LockTimeout(
                    f"Failed to acquire lock on {resource_id} "
                    f"after {max_wait}s"
                )

            # Exponential backoff with jitter
            await asyncio.sleep(wait_time + random.uniform(0, wait_time))
            wait_time = min(wait_time * 2, 1.0)

    async def release_lock(self, resource_id: str, owner_id: str) -> bool:
        """
        Release a lock, but only if we own it.

        Uses Lua script for atomic check-and-delete.
        """
        lock_key = f"lock:{resource_id}"

        # Lua script ensures atomic check and delete
        script = """
        local lock_data = redis.call('GET', KEYS[1])
        if lock_data then
            local data = cjson.decode(lock_data)
            if data.owner_id == ARGV[1] then
                return redis.call('DEL', KEYS[1])
            end
        end
        return 0
        """

        result = await self._store._redis.eval(script, 1, lock_key, owner_id)
        released = result == 1
        if not released:
            # Either the lock had already expired, or another owner is
            # holding it. Either way, surface it on the structured
            # logger so ops can see the unexpected release attempt.
            import logging
            logging.getLogger(__name__).warning(
                "release_lock no-op for resource=%s owner=%s "
                "(lock expired or owned by someone else)",
                resource_id, owner_id,
            )
        return released

    async def read(self, resource_id: str) -> Optional[dict[str, Any]]:
        """Read shared state."""
        return await self._store.get(f"shared:{resource_id}")

    async def write(self, resource_id: str, state: dict[str, Any]) -> None:
        """Write shared state."""
        await self._store.set(f"shared:{resource_id}", state)

    async def atomic_update(
        self, resource_id: str, update_fn: Callable[[Optional[dict]], dict]
    ) -> dict[str, Any]:
        """
        Perform an atomic read-modify-write operation.

        The update function receives the current state (or None)
        and returns the new state. The entire operation is protected
        by a lock.
        """
        async with self.locked(resource_id):
            current = await self.read(resource_id)
            new_state = update_fn(current)
            await self.write(resource_id, new_state)
            return new_state

    async def watch_changes(
        self,
        resource_id: str,
        callback: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        """
        Watch for changes to a shared resource.

        Uses Redis pub/sub for real-time notifications.
        """
        channel = f"changes:{resource_id}"
        pubsub = self._store._redis.pubsub()
        await pubsub.subscribe(channel)

        # Cancellable listen loop: ``get_message`` with a timeout
        # yields control regularly so the surrounding asyncio task can
        # be cancelled if Redis drops the connection or the caller
        # tears down the watcher. ``async for ... listen()`` would
        # block indefinitely on a half-open socket.
        try:
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if message is None:
                    continue
                if message.get("type") == "message":
                    data = json.loads(message["data"])
                    await callback(data)
        except asyncio.CancelledError:
            raise
        finally:
            await pubsub.unsubscribe(channel)

    async def publish_change(
        self, resource_id: str, state: dict[str, Any]
    ) -> None:
        """Publish a state change notification."""
        channel = f"changes:{resource_id}"
        await self._store._redis.publish(channel, json.dumps(state))


class LockTimeout(Exception):
    """Raised when lock acquisition times out."""

    pass


# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

class ConflictResolver:
    """
    Resolves conflicts in shared state updates.

    Implements various conflict resolution strategies based on
    application requirements.
    """

    @staticmethod
    def last_write_wins(
        state_a: dict[str, Any], state_b: dict[str, Any]
    ) -> dict[str, Any]:
        """Simple last-write-wins based on timestamp."""
        time_a = datetime.fromisoformat(
            state_a.get("_updated_at", "1970-01-01T00:00:00")
        )
        time_b = datetime.fromisoformat(
            state_b.get("_updated_at", "1970-01-01T00:00:00")
        )
        return state_b if time_b >= time_a else state_a

    @staticmethod
    def merge_additive(
        state_a: dict[str, Any], state_b: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Merge states by combining values additively.

        Suitable for counters, lists, and other additive structures.
        """
        merged = {}
        all_keys = set(state_a.keys()) | set(state_b.keys())

        for key in all_keys:
            if key.startswith("_"):
                # Use latest metadata
                merged[key] = state_b.get(key) or state_a.get(key)
            elif key in state_a and key in state_b:
                val_a, val_b = state_a[key], state_b[key]

                if isinstance(val_a, (int, float)) and isinstance(
                    val_b, (int, float)
                ):
                    # Sum numeric values
                    merged[key] = val_a + val_b
                elif isinstance(val_a, list) and isinstance(val_b, list):
                    # Concatenate and deduplicate lists
                    merged[key] = list(dict.fromkeys(val_a + val_b))
                elif isinstance(val_a, dict) and isinstance(val_b, dict):
                    # Recursively merge dicts
                    merged[key] = ConflictResolver.merge_additive(
                        val_a, val_b
                    )
                else:
                    # Fall back to last-write-wins
                    merged[key] = val_b
            else:
                merged[key] = state_a.get(key) or state_b.get(key)

        return merged

    @staticmethod
    def vector_clock_merge(
        state_a: dict[str, Any], state_b: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """
        Merge using vector clocks to detect true conflicts.

        Returns (merged_state, had_conflict) tuple.
        """
        clock_a = state_a.get("_vclock", {})
        clock_b = state_b.get("_vclock", {})

        # Check if one dominates the other
        a_dominates = all(clock_a.get(k, 0) >= v for k, v in clock_b.items())
        b_dominates = all(clock_b.get(k, 0) >= v for k, v in clock_a.items())

        # Identical clocks: same causal version. Returning either
        # (without merging) is correct; merging would double-count
        # additive fields such as loyalty points.
        if a_dominates and b_dominates:
            return state_a, False
        if a_dominates:
            return state_a, False
        if b_dominates:
            return state_b, False

        # True concurrent updates - need merge
        merged = ConflictResolver.merge_additive(state_a, state_b)

        # Merge vector clocks
        merged_clock = {}
        for k in set(clock_a.keys()) | set(clock_b.keys()):
            merged_clock[k] = max(clock_a.get(k, 0), clock_b.get(k, 0))
        merged["_vclock"] = merged_clock

        return merged, True

# ============================================================================
# Block 12 (chapter listing #12)
# ============================================================================

from abc import ABC, abstractmethod


@dataclass
class StateVersion:
    """Tracks state schema version."""

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"

    @classmethod
    def parse(cls, version_str: str) -> "StateVersion":
        major, minor = version_str.split(".")
        return cls(int(major), int(minor))

    def __lt__(self, other: "StateVersion") -> bool:
        return (self.major, self.minor) < (other.major, other.minor)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateVersion):
            return False
        return self.major == other.major and self.minor == other.minor


class StateMigration(ABC):
    """Base class for state migrations."""

    @property
    @abstractmethod
    def from_version(self) -> StateVersion:
        """Version this migration upgrades from."""
        pass

    @property
    @abstractmethod
    def to_version(self) -> StateVersion:
        """Version this migration upgrades to."""
        pass

    @abstractmethod
    async def migrate(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Migrate state from from_version to to_version.

        Must be idempotent - applying the same migration twice
        should produce the same result.
        """
        pass

    @abstractmethod
    async def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Rollback migration, reverting to from_version.

        May not be possible for all migrations (data loss).
        """
        pass


class StateMigrator:
    """
    Manages state schema migrations.

    Automatically detects state version and applies necessary
    migrations to bring state up to current schema version.
    """

    def __init__(self, current_version: StateVersion):
        self._current_version = current_version
        self._migrations: list[StateMigration] = []

    def register_migration(self, migration: StateMigration) -> None:
        """Register a migration."""
        self._migrations.append(migration)
        # Keep migrations sorted by from_version
        self._migrations.sort(key=lambda m: m.from_version)

    async def migrate(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Migrate state to current version.

        Applies all necessary migrations in sequence.
        """
        version_str = state.get("_schema_version", "1.0")
        current = StateVersion.parse(version_str)

        if current == self._current_version:
            return state

        if current > self._current_version:
            raise MigrationError(
                f"State version {current} is newer than "
                f"current schema {self._current_version}"
            )

        # Find migration path
        path = self._find_migration_path(current, self._current_version)

        if not path:
            raise MigrationError(
                f"No migration path from {current} to {self._current_version}"
            )

        # Apply migrations
        migrated = state.copy()
        for migration in path:
            migrated = await migration.migrate(migrated)
            migrated["_schema_version"] = str(migration.to_version)
            migrated["_migrated_at"] = datetime.now(timezone.utc).isoformat()

        return migrated

    def _find_migration_path(
        self, from_version: StateVersion, to_version: StateVersion
    ) -> Optional[list[StateMigration]]:
        """Find sequence of migrations from one version to another."""
        path = []
        current = from_version

        while current < to_version:
            # Find migration from current version
            migration = next(
                (m for m in self._migrations if m.from_version == current),
                None,
            )
            if migration is None:
                return None
            path.append(migration)
            current = migration.to_version

        return path if current == to_version else None

    async def migrate_batch(
        self, store: StateStore, key_pattern: str, batch_size: int = 100
    ) -> dict[str, int]:
        """
        Migrate all states matching pattern.

        Returns statistics about the migration.
        """
        stats = {"migrated": 0, "already_current": 0, "failed": 0}

        # This would iterate through all matching keys
        # Implementation depends on store capabilities
        # Here's a conceptual example:

        # async for key in store.scan(key_pattern):
        #     state = await store.get(key)
        #     if state:
        #         try:
        #             migrated = await self.migrate(state)
        #             if migrated != state:
        #                 await store.set(key, migrated)
        #                 stats["migrated"] += 1
        #             else:
        #                 stats["already_current"] += 1
        #         except Exception:
        #             stats["failed"] += 1

        return stats


class MigrationError(Exception):
    """Raised when migration fails."""

    pass


class AddUserPreferencesFieldMigration(StateMigration):
    """Migration that adds user_preferences field."""

    @property
    def from_version(self) -> StateVersion:
        return StateVersion(1, 0)

    @property
    def to_version(self) -> StateVersion:
        return StateVersion(1, 1)

    async def migrate(self, state: dict[str, Any]) -> dict[str, Any]:
        migrated = state.copy()
        if "user_preferences" not in migrated:
            migrated["user_preferences"] = {
                "language": "en",
                "timezone": "UTC",
                "notifications_enabled": True,
            }
        return migrated

    async def rollback(self, state: dict[str, Any]) -> dict[str, Any]:
        # This migration is safe to rollback
        migrated = state.copy()
        migrated.pop("user_preferences", None)
        return migrated

# ============================================================================
# Block 13 (chapter block #13) — non-python listing (YAML)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_13_listing = r"""
Timeline:
09:14 - Redis cluster initiates planned failover
09:15 - New primary begins accepting writes
09:17 - First corrupted state detected by integrity check
09:23 - Alert fires: corruption rate exceeds threshold
09:31 - Engineering investigation begins
09:45 - Root cause identified
09:52 - Rollback deployed for serialization fix
10:15 - Recovery process begins for corrupted states
12:30 - Recovery complete, all affected users restored
"""

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

# Pattern 1: Truncated JSON
corrupted_1 = '{"messages": [{"role": "user", "content": "Hello'

# Pattern 2: Invalid Unicode sequences
corrupted_2 = '{"content": "Test \\ud800 message"}'

# Pattern 3: Duplicate keys (from retry logic)
corrupted_3 = '{"id": "abc", "id": "def", "content": "text"}'

# ============================================================================
# Block 15 (chapter listing #15)
# ============================================================================

from typing import Protocol, runtime_checkable


@runtime_checkable
class _LLMClient(Protocol):
    """Structural type for the recovery LLM call site.

    Recovery uses ``complete`` to ask the model to suggest a state
    reconstruction from the audit log. Wrap your provider client
    (Anthropic, OpenAI, etc.) in a thin adapter that exposes this.
    """

    async def complete(self, messages: list[dict], **kwargs: Any) -> Any: ...


class ConversationRecoveryPipeline:
    """
    Production recovery pipeline used to restore corrupted conversations.

    This code represents the actual recovery logic deployed during
    the incident, with details abstracted.
    """

    def __init__(
        self,
        primary_store: RedisStateStore,
        backup_store: RedisStateStore,
        audit_log: PostgresTaskStore,
        llm_client: _LLMClient,
    ):
        self._primary = primary_store
        self._backup = backup_store
        self._audit = audit_log
        self._llm = llm_client

    async def recover_conversation(
        self, conversation_id: str
    ) -> RecoveryResult:
        """
        Attempt to recover a single corrupted conversation.

        Recovery phases:
        1. Try parsing primary store data
        2. Attempt repair of partial corruption
        3. Fall back to backup store
        4. Reconstruct from audit log
        5. Use LLM to regenerate summaries if needed
        """
        key = f"conversation:{conversation_id}"

        # Phase 1: Check if actually corrupted
        try:
            primary_data = await self._primary.get(key)
            if primary_data and self._validate_conversation(primary_data):
                return RecoveryResult(
                    task_id=conversation_id,
                    recovered=True,
                    resume_point=primary_data,
                    data_loss=False,
                    error=None,
                )
        except json.JSONDecodeError as e:
            # Log corruption details for analysis
            await self._log_corruption(conversation_id, "json_decode", str(e))

        # Phase 2: Attempt to repair corrupted JSON
        raw_data = await self._get_raw_data(key)
        if raw_data:
            repaired = self._attempt_json_repair(raw_data)
            if repaired and self._validate_conversation(repaired):
                await self._primary.set(key, repaired)
                return RecoveryResult(
                    task_id=conversation_id,
                    recovered=True,
                    resume_point=repaired,
                    data_loss=False,
                    error=None,
                )

        # Phase 3: Try backup store
        backup_data = await self._backup.get(key)
        if backup_data and self._validate_conversation(backup_data):
            await self._primary.set(key, backup_data)
            await self._log_recovery(conversation_id, "backup_restore")
            return RecoveryResult(
                task_id=conversation_id,
                recovered=True,
                resume_point=backup_data,
                data_loss=self._calculate_data_loss(backup_data),
                error=None,
            )

        # Phase 4: Reconstruct from audit log
        reconstructed = await self._reconstruct_from_audit(conversation_id)
        if reconstructed:
            await self._primary.set(key, reconstructed)
            return RecoveryResult(
                task_id=conversation_id,
                recovered=True,
                resume_point=reconstructed,
                data_loss=True,  # Summaries lost
                error=None,
            )

        # Phase 5: Unrecoverable
        return RecoveryResult(
            task_id=conversation_id,
            recovered=False,
            resume_point=None,
            data_loss=True,
            error="Unable to recover conversation from any source",
        )

    def _attempt_json_repair(self, raw: bytes) -> Optional[dict]:
        """
        Attempt to repair common JSON corruption patterns.
        """
        text = raw.decode("utf-8", errors="replace")

        # Remove invalid Unicode surrogates
        text = self._fix_unicode_surrogates(text)

        # Try to fix truncation
        if not text.rstrip().endswith("}"):
            text = self._complete_json_structure(text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _fix_unicode_surrogates(self, text: str) -> str:
        """Remove or replace invalid Unicode surrogates."""
        import re

        # Match unpaired surrogates
        surrogate_pattern = re.compile(
            r"\\u[dD][89aAbB][0-9a-fA-F]{2}"
            r"(?!\\u[dD][cCdDeEfF][0-9a-fA-F]{2})"
        )
        return surrogate_pattern.sub("", text)

    def _complete_json_structure(self, text: str) -> str:
        """Attempt to complete truncated JSON structure."""
        # Count open braces/brackets
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        # Check if we're inside a string
        in_string = text.count('"') % 2 == 1

        if in_string:
            text += '"'

        text += "]" * open_brackets
        text += "}" * open_braces

        return text

    def _validate_conversation(self, data: dict) -> bool:
        """Validate conversation state structure."""
        required_fields = ["messages"]
        if not all(f in data for f in required_fields):
            return False

        if not isinstance(data["messages"], list):
            return False

        for msg in data["messages"]:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False

        return True

    async def _query_audit_events(
        self, conversation_id: str
    ) -> list[dict]:
        """Return ordered audit events for a conversation.

        Default implementation: look up events on the optional
        ``audit_store`` the pipeline was constructed with. Real
        deployments inject a store backed by their audit log
        (Postgres, OpenSearch, etc.); the in-memory fallback below
        keeps the example runnable without that wiring.
        """
        store = getattr(self, "audit_store", None)
        if store is not None and hasattr(store, "events_for"):
            events = await store.events_for(conversation_id)
            return list(events) if events else []
        return []

    async def _reconstruct_from_audit(
        self, conversation_id: str
    ) -> Optional[dict]:
        """Reconstruct conversation from audit log entries."""
        # Query audit log for all events related to this conversation
        # This is a simplified representation
        events = await self._query_audit_events(conversation_id)

        if not events:
            return None

        messages = []
        for event in events:
            if event["type"] == "message_added":
                messages.append(
                    {
                        "role": event["data"]["role"],
                        "content": event["data"]["content"],
                        "timestamp": event["timestamp"],
                    }
                )

        return {
            "messages": messages,
            "summary": None,  # Lost - would need regeneration
            "total_tokens": sum(
                len(m["content"].split()) * 1.3 for m in messages
            ),
            "_recovered": True,
            "_recovered_at": datetime.now(timezone.utc).isoformat(),
        }
