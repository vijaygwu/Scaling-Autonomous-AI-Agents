"""
Scaling Agent Systems

Code listings from Chapter 01, Book 3:
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

from __future__ import annotations


# ============================================================================
# Block 1 (chapter listing #1)
# ============================================================================

from collections import OrderedDict, deque
from typing import Any
import threading
import time

# What went wrong: stateful agent design
class CustomerAgent:
    def __init__(
        self, max_messages_per_user: int = 100, max_users: int = 10_000
    ) -> None:
        if max_messages_per_user < 1:
            raise ValueError("max_messages_per_user must be >= 1")
        if max_users < 1:
            raise ValueError("max_users must be >= 1")
        # Still the wrong scaling model because state is local to this
        # process, but bounded so the anti-pattern cannot OOM a demo.
        self.max_messages_per_user = max_messages_per_user
        self.max_users = max_users
        self.conversation_history: OrderedDict[str, deque[dict]] = (
            OrderedDict()
        )
        self._history_lock = threading.RLock()

    def handle_request(self, user_id: str, message: str) -> None:
        # State lost when this instance dies or scales
        with self._history_lock:
            if user_id not in self.conversation_history:
                if len(self.conversation_history) >= self.max_users:
                    self.conversation_history.popitem(last=False)
                self.conversation_history[user_id] = deque(
                    maxlen=self.max_messages_per_user
                )
            self.conversation_history.move_to_end(user_id)
            self.conversation_history[user_id].append(
                {"role": "user", "content": message}
            )
        # ...

# ============================================================================
# Block 2 (chapter listing #2)
# ============================================================================

try:  # Optional provider SDK; keep this module importable without it.
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dependency not required for examples
    anthropic = None  # type: ignore[assignment]

# Anti-pattern: Stateful agent design
class StatefulAgent:
    def __init__(
        self,
        llm_client: Any | None = None,
        max_messages_per_conversation: int = 100,
        max_conversations: int = 10_000,
        llm_max_retries: int = 3,
        llm_retry_backoff: float = 0.25,
        llm_model: str = "claude-sonnet-4-20250514",
        llm_max_tokens: int = 1024,
        llm_timeout_seconds: float = 30.0,
    ) -> None:
        if max_messages_per_conversation < 1:
            raise ValueError("max_messages_per_conversation must be >= 1")
        if max_conversations < 1:
            raise ValueError("max_conversations must be >= 1")
        if llm_max_retries < 1:
            raise ValueError("llm_max_retries must be >= 1")
        if llm_retry_backoff < 0:
            raise ValueError("llm_retry_backoff must be >= 0")
        if llm_max_tokens < 1:
            raise ValueError("llm_max_tokens must be >= 1")
        if llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be > 0")
        # Still an anti-pattern because state is local to one process, but
        # bound the demo so a load test cannot OOM the interpreter.
        self.llm_client = llm_client
        self.max_messages_per_conversation = max_messages_per_conversation
        self.max_conversations = max_conversations
        self.llm_max_retries = llm_max_retries
        self.llm_retry_backoff = llm_retry_backoff
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.llm_timeout_seconds = llm_timeout_seconds
        self.conversations: OrderedDict[str, deque[dict]] = OrderedDict()
        self._conversation_lock = threading.RLock()

    def process_message(self, conversation_id: str, message: str) -> str:
        with self._conversation_lock:
            if conversation_id not in self.conversations:
                if len(self.conversations) >= self.max_conversations:
                    self.conversations.popitem(last=False)
                self.conversations[conversation_id] = deque(
                    maxlen=self.max_messages_per_conversation
                )
            self.conversations.move_to_end(conversation_id)

            self.conversations[conversation_id].append(
                {"role": "user", "content": message}
            )

            response = self._call_llm(self.conversations[conversation_id])

            self.conversations[conversation_id].append(
                {"role": "assistant", "content": response}
            )

            return response

    def _call_llm(self, messages: deque[dict]) -> str:
        """Minimal LLM call stub so the anti-pattern example is runnable."""
        if self.llm_client is None:
            return "stub response"
        retryable_errors = [TimeoutError, ConnectionError, OSError]
        provider = anthropic
        if provider is not None:
            for name in ("APITimeoutError", "APIConnectionError", "RateLimitError"):
                err = getattr(provider, name, None)
                if err is not None:
                    retryable_errors.append(err)

        for attempt in range(self.llm_max_retries):
            try:
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=self.llm_max_tokens,
                    messages=list(messages),
                    timeout=self.llm_timeout_seconds,
                )
                return response.content[0].text
            except tuple(retryable_errors):
                if attempt == self.llm_max_retries - 1:
                    raise
                time.sleep(self.llm_retry_backoff * (2**attempt))
        raise RuntimeError("LLM call exhausted retries")

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

import json
try:
    import redis
    from redis.exceptions import RedisError
except ImportError:
    redis = None  # type: ignore[assignment]

    class RedisError(Exception):
        pass
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol
import hashlib


@dataclass
class ConversationState:
    """Represents externalized conversation state."""

    conversation_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "conversation_id": self.conversation_id,
                "messages": self.messages,
                "metadata": self.metadata,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "ConversationState":
        parsed = json.loads(data)
        return cls(**parsed)


class _MessageClient(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _LLMClient(Protocol):
    messages: _MessageClient


class StateManager:
    """Manages externalized agent state in Redis."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        state_ttl: int = 86400,
        max_retries: int = 3,
        retry_backoff: float = 0.05,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        max_connections: int = 50,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if redis is None:
            raise ImportError("redis is required for StateManager")
        # Explicit timeouts + health-check + bounded pool. The default
        # ``redis.from_url(...)`` produces a client that blocks
        # indefinitely on a hung server; in a worker pool that
        # serializes every state op behind the slowest TCP timeout.
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            max_connections=max_connections,
        )
        self.state_ttl = state_ttl
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _redis_call(self, operation: Callable[..., Any], *args: Any) -> Any:
        """Run a Redis operation with bounded retry/backoff.

        On final failure we re-raise the original exception so that
        callers can discriminate by error class (e.g., RedisError vs
        ConnectionError vs TimeoutError) for retry pipelines, rather
        than catching a generic RuntimeError and inspecting __cause__.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args)
            except (RedisError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))
        # Preserve the original exception class so callers can pattern-match.
        assert last_error is not None
        raise last_error

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Retrieve conversation state from Redis."""
        key = f"conversation:{conversation_id}"
        data = self._redis_call(self.redis.get, key)

        if data is None:
            return None

        return ConversationState.from_json(data)

    def save_state(self, state: ConversationState) -> None:
        """Persist conversation state to Redis."""
        state.updated_at = time.time()
        if state.created_at == 0.0:
            state.created_at = state.updated_at

        key = f"conversation:{state.conversation_id}"
        version_key = f"conversation:{state.conversation_id}:version"

        def write_state() -> Any:
            pipe = self.redis.pipeline()
            pipe.setex(key, self.state_ttl, state.to_json())
            pipe.incr(version_key)
            pipe.expire(version_key, self.state_ttl)
            return pipe.execute()

        self._redis_call(write_state)

    def get_state_with_version(
        self, conversation_id: str
    ) -> tuple[Optional[ConversationState], int]:
        """Retrieve conversation state along with its version."""
        key = f"conversation:{conversation_id}"
        version_key = f"conversation:{conversation_id}:version"

        def read_versioned_state() -> tuple[Optional[str], Optional[str]]:
            pipe = self.redis.pipeline()
            pipe.get(key)
            pipe.get(version_key)
            return pipe.execute()

        data, version = self._redis_call(read_versioned_state)

        if data is None:
            return None, 0

        return ConversationState.from_json(data), int(version or 0)

    def save_state_with_version(
        self, state: ConversationState, expected_version: int
    ) -> bool:
        """Persist state only if Redis still holds the expected version."""
        state.updated_at = time.time()
        if state.created_at == 0.0:
            state.created_at = state.updated_at

        key = f"conversation:{state.conversation_id}"
        version_key = f"conversation:{state.conversation_id}:version"
        lua_script = """
        local current_version = redis.call('GET', KEYS[2])
        if current_version == false then current_version = 0 end

        if tonumber(current_version) ~= tonumber(ARGV[2]) then
            return 0
        end

        redis.call('SETEX', KEYS[1], ARGV[3], ARGV[1])
        redis.call('SETEX', KEYS[2], ARGV[3], tonumber(ARGV[2]) + 1)
        return 1
        """

        result = self._redis_call(
            self.redis.eval,
            lua_script,
            2,
            key,
            version_key,
            state.to_json(),
            expected_version,
            self.state_ttl,
        )

        return result == 1

    def delete_state(self, conversation_id: str) -> None:
        """Remove conversation state."""
        key = f"conversation:{conversation_id}"
        version_key = f"conversation:{conversation_id}:version"

        def delete_keys() -> Any:
            pipe = self.redis.pipeline()
            pipe.delete(key)
            pipe.delete(version_key)
            return pipe.execute()

        self._redis_call(delete_keys)

    def extend_ttl(self, conversation_id: str) -> None:
        """Extend the TTL for an active conversation."""
        key = f"conversation:{conversation_id}"
        version_key = f"conversation:{conversation_id}:version"

        def extend_keys() -> Any:
            pipe = self.redis.pipeline()
            pipe.expire(key, self.state_ttl)
            pipe.expire(version_key, self.state_ttl)
            return pipe.execute()

        self._redis_call(extend_keys)


class VersionedStateStore(Protocol):
    def get_state_with_version(
        self, conversation_id: str
    ) -> tuple[Optional[ConversationState], int]: ...

    def save_state_with_version(
        self, state: ConversationState, expected_version: int
    ) -> bool: ...


class StateConflictError(Exception):
    """Raised when state changes faster than bounded CAS retries can merge."""


class StatelessAgent:
    """Agent designed for horizontal scaling with externalized state."""

    def __init__(
        self,
        state_manager: VersionedStateStore,
        llm_client: _LLMClient,
        max_messages_per_conversation: int = 100,
        llm_max_retries: int = 3,
        llm_retry_backoff: float = 0.25,
        llm_model: str = "claude-sonnet-4-20250514",
        llm_max_tokens: int = 4096,
        llm_timeout_seconds: float = 30.0,
        state_write_max_retries: int = 5,
        state_write_retry_backoff: float = 0.02,
    ) -> None:
        if max_messages_per_conversation < 1:
            raise ValueError("max_messages_per_conversation must be >= 1")
        if llm_max_retries < 1:
            raise ValueError("llm_max_retries must be >= 1")
        if llm_max_tokens < 1:
            raise ValueError("llm_max_tokens must be >= 1")
        if llm_timeout_seconds <= 0:
            raise ValueError("llm_timeout_seconds must be > 0")
        if state_write_max_retries < 1:
            raise ValueError("state_write_max_retries must be >= 1")
        if state_write_retry_backoff < 0:
            raise ValueError("state_write_retry_backoff must be >= 0")
        self.state_manager = state_manager
        self.llm_client = llm_client
        self.max_messages_per_conversation = max_messages_per_conversation
        self.llm_max_retries = llm_max_retries
        self.llm_retry_backoff = llm_retry_backoff
        self.llm_model = llm_model
        self.llm_max_tokens = llm_max_tokens
        self.llm_timeout_seconds = llm_timeout_seconds
        self.state_write_max_retries = state_write_max_retries
        self.state_write_retry_backoff = state_write_retry_backoff

    def _trim_messages(self, messages: list[dict]) -> list[dict]:
        """Apply the context-window cap before any LLM call."""
        return messages[-self.max_messages_per_conversation :]

    def process_message(self, conversation_id: str, message: str) -> str:
        for attempt in range(self.state_write_max_retries):
            # Load state and version together so concurrent workers cannot
            # overwrite each other's appended messages.
            state, version = self.state_manager.get_state_with_version(
                conversation_id
            )

            if state is None:
                state = ConversationState(conversation_id=conversation_id)

            messages = self._trim_messages(
                list(state.messages)
                + [{"role": "user", "content": message}]
            )

            # Generate response against the version we plan to save.
            response = self._call_llm(list(messages))

            state.messages = self._trim_messages(
                messages + [{"role": "assistant", "content": response}]
            )

            if self.state_manager.save_state_with_version(state, version):
                return response

            if attempt < self.state_write_max_retries - 1:
                time.sleep(self.state_write_retry_backoff * (2**attempt))

        raise StateConflictError(
            f"Conversation {conversation_id} changed during write retries"
        )

    def _call_llm(self, messages: list[dict]) -> str:
        # Anthropic-SDK clients accept ``timeout`` and ``max_retries``
        # at construction time; surface them here for readers porting
        # this code so a stalled provider does not block the worker
        # indefinitely. Production callers should additionally wrap
        # this in the @with_retry/@with_timeout decorators introduced
        # in ch05 (customer service) to cap end-to-end latency budgets.
        retryable_errors = [TimeoutError, ConnectionError, OSError]
        provider = anthropic
        if provider is not None:
            for name in ("APITimeoutError", "APIConnectionError", "RateLimitError"):
                err = getattr(provider, name, None)
                if err is not None:
                    retryable_errors.append(err)

        for attempt in range(self.llm_max_retries):
            try:
                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=self.llm_max_tokens,
                    messages=messages,
                    timeout=self.llm_timeout_seconds,
                )
                return response.content[0].text
            except tuple(retryable_errors):
                if attempt == self.llm_max_retries - 1:
                    raise
                time.sleep(self.llm_retry_backoff * (2**attempt))
        raise RuntimeError("LLM call exhausted retries")

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

class VersionedStateManager:
    """State manager with optimistic locking for conflict detection."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        state_ttl: int = 86400,
        max_retries: int = 3,
        retry_backoff: float = 0.05,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        max_connections: int = 50,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if redis is None:
            raise ImportError("redis is required for VersionedStateManager")
        # Mirror StateManager's production timeouts/pool; a flaky Redis
        # would otherwise wedge every version check behind the default
        # unbounded socket timeout.
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            max_connections=max_connections,
        )
        self.state_ttl = state_ttl
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def _redis_call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (RedisError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))
        if last_error is not None:
            raise last_error
        raise RuntimeError("Versioned Redis operation failed after retries")

    def get_state_with_version(
        self, conversation_id: str
    ) -> tuple[Optional[ConversationState], int]:
        """Retrieve state along with its version number."""
        key = f"conversation:{conversation_id}"
        version_key = f"conversation:{conversation_id}:version"

        def read_versioned_state() -> tuple[Optional[str], Optional[str]]:
            pipe = self.redis.pipeline()
            pipe.get(key)
            pipe.get(version_key)
            return pipe.execute()

        data, version = self._redis_call(read_versioned_state)

        if data is None:
            return None, 0

        return ConversationState.from_json(data), int(version or 0)

    def save_state_with_version(
        self, state: ConversationState, expected_version: int
    ) -> bool:
        """
        Save state only if version matches expected.
        Returns True if save succeeded; False if conflict detected.
        """
        key = f"conversation:{state.conversation_id}"
        version_key = f"conversation:{state.conversation_id}:version"

        # Use Lua script for atomic check-and-set
        lua_script = """
        local current_version = redis.call('GET', KEYS[2])
        if current_version == false then current_version = 0 end
        
        if tonumber(current_version) ~= tonumber(ARGV[2]) then
            return 0
        end
        
        redis.call('SET', KEYS[1], ARGV[1])
        redis.call('INCR', KEYS[2])
        redis.call('EXPIRE', KEYS[1], ARGV[3])
        redis.call('EXPIRE', KEYS[2], ARGV[3])
        return 1
        """

        result = self._redis_call(
            self.redis.eval,
            lua_script,
            2,
            key,
            version_key,
            state.to_json(),
            expected_version,
            self.state_ttl,
        )

        return result == 1

# ============================================================================
# Block 5 (chapter block #5) - Python fragment (incomplete, depends on surrounding context)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_5_listing = r"""
Request 1 (2s) --> Worker A
Request 2 (45s) --> Worker B
Request 3 (3s) --> Worker A
Request 4 (60s) --> Worker B
Request 5 (2s) --> Worker A

Worker A: Handles requests 1, 3, 5 (7 seconds total)
Worker B: Still processing request 2, request 4 waiting
"""

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
import time
from typing import Optional
import random


@dataclass
class WorkerInfo:
    """Tracks information about a worker node."""

    worker_id: str
    address: str
    active_connections: int = 0
    total_requests: int = 0
    last_health_check: float = 0.0
    is_healthy: bool = True
    weight: float = 1.0


class LeastConnectionsBalancer:
    """Load balancer using least connections algorithm."""

    def __init__(self, max_workers: int = 10_000) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self.max_workers = max_workers
        self.workers: dict[str, WorkerInfo] = {}
        self.lock = threading.Lock()

    def register_worker(
        self, worker_id: str, address: str, weight: float = 1.0
    ) -> None:
        """Add a worker to the pool."""
        with self.lock:
            if worker_id not in self.workers and len(self.workers) >= self.max_workers:
                raise RuntimeError("worker registry is at capacity")
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                address=address,
                weight=weight,
                last_health_check=time.time(),
            )

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker from the pool."""
        with self.lock:
            self.workers.pop(worker_id, None)

    def select_worker(self) -> Optional[WorkerInfo]:
        """
        Select the worker with the fewest active connections,
        weighted by worker capacity.
        """
        with self.lock:
            healthy_workers = [
                w for w in self.workers.values() if w.is_healthy
            ]

            if not healthy_workers:
                return None

            # Calculate weighted connection count
            # Lower is better: connections / weight. ``max(w.weight, 1e-9)``
            # guards against a caller registering a worker with weight=0,
            # which would otherwise ZeroDivisionError mid-request.
            return min(
                healthy_workers,
                key=lambda w: w.active_connections / max(w.weight, 1e-9),
            )

    def mark_request_start(self, worker_id: str) -> None:
        """Increment active connection count."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].active_connections += 1
                self.workers[worker_id].total_requests += 1

    def mark_request_complete(self, worker_id: str) -> None:
        """Decrement active connection count."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].active_connections = max(
                    0, self.workers[worker_id].active_connections - 1
                )

    def mark_worker_unhealthy(self, worker_id: str) -> None:
        """Mark a worker as unhealthy."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].is_healthy = False

    def mark_worker_healthy(self, worker_id: str) -> None:
        """Mark a worker as healthy."""
        with self.lock:
            if worker_id in self.workers:
                self.workers[worker_id].is_healthy = True
                self.workers[worker_id].last_health_check = time.time()

# ============================================================================
# Block 7 (chapter listing #7)
# ============================================================================

class WeightedBalancer:
    """
    Load balancer that considers worker capacity and
    current performance metrics.
    """

    def __init__(
        self, max_workers: int = 10_000, history_window: int = 100
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if history_window < 1:
            raise ValueError("history_window must be >= 1")
        self.max_workers = max_workers
        self.workers: dict[str, WorkerInfo] = {}
        self.history_window = history_window
        # ``deque(maxlen=...)`` makes the bound automatic and avoids an
        # O(n) slice copy on every overflow append.
        self.latency_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.history_window)
        )
        # RLock (reentrant) because ``select_worker`` holds the lock and
        # then calls ``get_average_latency``, which also takes it. A
        # plain ``Lock`` would deadlock on that second acquire.
        self.lock = threading.RLock()

    def register_worker(
        self, worker_id: str, address: str, weight: float = 1.0
    ) -> None:
        """Add a worker to the weighted pool."""
        with self.lock:
            if worker_id not in self.workers and len(self.workers) >= self.max_workers:
                raise RuntimeError("worker registry is at capacity")
            self.workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                address=address,
                weight=weight,
                last_health_check=time.time(),
            )
            self.latency_history.setdefault(
                worker_id, deque(maxlen=self.history_window)
            )

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a worker and its latency history."""
        with self.lock:
            self.workers.pop(worker_id, None)
            self.latency_history.pop(worker_id, None)

    def record_latency(self, worker_id: str, latency_ms: float) -> None:
        """Record request latency for a worker."""
        with self.lock:
            history = self.latency_history.get(worker_id)
            if history is None or worker_id not in self.workers:
                raise KeyError(f"unknown worker_id: {worker_id}")
            history.append(latency_ms)

    def get_average_latency(self, worker_id: str) -> float:
        """Get average latency for a worker."""
        with self.lock:
            history = self.latency_history.get(worker_id)
            if not history:
                return 0.0
            return sum(history) / len(history)

    def select_worker(self) -> Optional[WorkerInfo]:
        """
        Select worker using weighted algorithm that considers:
        - Static weight (machine capacity)
        - Active connections
        - Historical latency
        """
        with self.lock:
            healthy_workers = [
                w for w in self.workers.values() if w.is_healthy
            ]

            if not healthy_workers:
                return None

            def calculate_score(worker: WorkerInfo) -> float:
                # Base score from weight and connections
                connection_score = (
                    worker.active_connections + 1
                ) / max(worker.weight, 1e-9)

                # Latency penalty
                avg_latency = self.get_average_latency(worker.worker_id)
                latency_factor = 1.0 + (
                    avg_latency / 1000.0
                )  # Normalize to seconds

                return connection_score * latency_factor

            # Lower score is better
            return min(healthy_workers, key=calculate_score)

# ============================================================================
# Block 8 (chapter listing #8)
# ============================================================================

import json
try:
    import redis
    from redis.exceptions import RedisError
except ImportError:
    redis = None  # type: ignore[assignment]

    class RedisError(Exception):
        pass
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Callable, Any
from enum import Enum
import logging
import threading


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class AgentTask:
    """Represents a task for agent processing."""

    task_id: str
    conversation_id: str
    message: str
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0
    max_attempts: int = 3
    result: Optional[str] = None
    error: Optional[str] = None

    def to_json(self) -> str:
        data = asdict(self)
        data["status"] = self.status.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, data: str) -> "AgentTask":
        parsed = json.loads(data)
        parsed["status"] = TaskStatus(parsed["status"])
        return cls(**parsed)


class RedisTaskQueue:
    """Redis-based task queue with retry, DLQ, and visibility-timeout recovery.

    Production deployments typically construct this from environment
    variables rather than hardcoded args. A from_env() factory keeps
    capacity-planning knobs (visibility timeout, body TTL, queue
    cap) out of the deployment image so they can be tuned without a
    code change. Example::

        @classmethod
        def from_env(cls) -> "RedisTaskQueue":
            import os
            return cls(
                redis_url=os.environ.get(
                    "REDIS_URL", "redis://localhost:6379"
                ),
                queue_name=os.environ.get(
                    "AGENT_QUEUE_NAME", "agent_tasks"
                ),
                visibility_timeout=int(
                    os.environ.get("AGENT_QUEUE_VISIBILITY_TIMEOUT", "300")
                ),
                body_ttl_seconds=int(
                    os.environ.get(
                        "AGENT_QUEUE_BODY_TTL_SECONDS", str(7 * 24 * 3600)
                    )
                ),
                max_queue_size=int(
                    os.environ.get("AGENT_QUEUE_MAX_SIZE", "100000")
                ),
                max_retries=int(
                    os.environ.get("AGENT_QUEUE_MAX_RETRIES", "3")
                ),
                retry_backoff=float(
                    os.environ.get("AGENT_QUEUE_RETRY_BACKOFF", "0.05")
                ),
            )

    The same pattern applies to ``WorkerPool`` (min/max workers, idle
    timeout) and ``AutoScalerConfig`` (thresholds, cooldowns); keep
    the env-var names namespaced so multiple agent pools can coexist.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_name: str = "agent_tasks",
        visibility_timeout: int = 300,  # 5 minutes
        body_ttl_seconds: int = 7 * 24 * 3600,  # 7 days
        max_queue_size: int = 100_000,
        max_retries: int = 3,
        retry_backoff: float = 0.05,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        max_connections: int = 50,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if redis is None:
            raise ImportError("redis is required for RedisTaskQueue")
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            max_connections=max_connections,
        )
        self.queue_name = queue_name
        self.processing_queue = f"{queue_name}:processing"
        self.dead_letter_queue = f"{queue_name}:dlq"
        self.visibility_index = f"{queue_name}:visibility"
        self.visibility_timeout = visibility_timeout
        self.max_queue_size = max_queue_size
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Body TTL must comfortably exceed the worst-case queue
        # residence (queue depth x visibility timeout + retries).
        # A 24-hour TTL on the body but multi-day queue residence
        # causes the id to outlive its body and the dequeue path
        # returns None silently (work is lost). 7 days is a safer
        # default; callers with a different SLA should override it.
        self.body_ttl_seconds = body_ttl_seconds
        self._enqueue_script = """
        if redis.call('LLEN', KEYS[1]) >= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('SETEX', KEYS[2], tonumber(ARGV[2]), ARGV[3])
        redis.call('LPUSH', KEYS[1], ARGV[4])
        return 1
        """
        self._return_to_head_script = """
        local removed = redis.call('LREM', KEYS[2], 1, ARGV[1])
        if removed == 0 then
            return 0
        end
        redis.call('SETEX', KEYS[1], tonumber(ARGV[2]), ARGV[3])
        redis.call('DEL', KEYS[3])
        redis.call('RPUSH', KEYS[4], ARGV[1])
        return 1
        """
        self._retry_requeue_script = """
        redis.call('SETEX', KEYS[1], tonumber(ARGV[2]), ARGV[3])
        local removed = redis.call('LREM', KEYS[2], 1, ARGV[1])
        if removed == 0 then
            return 0
        end
        redis.call('DEL', KEYS[4])
        redis.call('ZREM', KEYS[5], ARGV[1])
        redis.call('LPUSH', KEYS[3], ARGV[1])
        return 1
        """
        self._recover_stale_script = """
        local removed = redis.call('LREM', KEYS[1], 1, ARGV[1])
        if removed == 0 then
            redis.call('ZREM', KEYS[3], ARGV[1])
            return 0
        end
        redis.call('LPUSH', KEYS[2], ARGV[1])
        redis.call('ZREM', KEYS[3], ARGV[1])
        redis.call('DEL', KEYS[4])
        return 1
        """
        self.logger = logging.getLogger(__name__)

    def _redis_call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run Redis I/O with bounded retry for transient failures.

        On final failure we re-raise the original exception so that
        callers can discriminate by error class (e.g., RedisError vs
        ConnectionError vs TimeoutError) rather than catching a generic
        RuntimeError and inspecting ``__cause__``. Mirrors the contract
        used by StateManager._redis_call earlier in this chapter.
        """
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (RedisError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))
        # Preserve the original exception class so callers can pattern-match.
        assert last_error is not None
        raise last_error

    def enqueue(self, task: AgentTask) -> str:
        """Add a task to the queue."""
        task.created_at = time.time()

        # Store task details
        task_key = f"task:{task.task_id}"
        inserted = self._redis_call(
            self.redis.eval,
            self._enqueue_script,
            2,
            self.queue_name,
            task_key,
            self.max_queue_size,
            self.body_ttl_seconds,
            task.to_json(),
            task.task_id,
        )
        if not inserted:
            raise RuntimeError(f"Queue {self.queue_name} is at capacity")

        self.logger.info(
            "enqueued task",
            extra={
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
                "queue": self.queue_name,
            },
        )
        return task.task_id

    def dequeue(self, timeout: int = 0) -> Optional[AgentTask]:
        """
        Retrieve a task from the queue.
        Uses BLMOVE for reliable processing (Redis 6.2+).
        """
        # Move task from main queue to processing queue atomically
        task_id = self._redis_call(
            self.redis.blmove,
            self.queue_name,
            self.processing_queue,
            timeout=timeout,
            src="RIGHT",
            dest="LEFT",
        )

        if task_id is None:
            return None

        # Retrieve task details
        task_key = f"task:{task_id}"
        task_data = self._redis_call(self.redis.get, task_key)

        if task_data is None:
            # Task body TTL expired between submit and claim. Remove
            # the orphan ID from the processing queue so it does not
            # block the cursor. We intentionally drop rather than DLQ:
            # the body is gone, so we cannot recover the work. In a
            # production deployment you would emit a counter here
            # (e.g., ``metrics.increment("queue.orphan_drop")``) and
            # tune ``body_ttl_seconds`` upward so this path is
            # diagnostic rather than data-loss.
            self._redis_call(self.redis.lrem, self.processing_queue, 1, task_id)
            return None

        task = AgentTask.from_json(task_data)
        task.status = TaskStatus.PROCESSING
        # ``attempt`` increments at dequeue, not at completion. The
        # contract is "at most ``max_attempts`` claim cycles", which
        # is what the visibility-timeout retry path counts; this is
        # not the same as "at most ``max_attempts`` real failures".
        # Producers that need stricter failure-only counting should
        # subtract one in fail()/complete() or move the increment
        # into AgentWorker._process_task. The BackpressureQueueConsumer
        # below rolls back this bump to avoid burning attempts on
        # shaping-only dequeues.
        task.attempt += 1

        # Update task status
        self._redis_call(
            self.redis.setex, task_key, self.body_ttl_seconds, task.to_json()
        )

        # Set visibility timeout
        self._set_visibility_timeout(task_id)

        return task

    def complete(self, task: AgentTask, result: str) -> None:
        """Mark a task as successfully completed."""
        task.status = TaskStatus.COMPLETED
        task.result = result

        task_key = f"task:{task.task_id}"
        self._redis_call(
            self.redis.setex, task_key, self.body_ttl_seconds, task.to_json()
        )

        # Remove from processing queue
        self._redis_call(self.redis.lrem, self.processing_queue, 1, task.task_id)
        self._clear_visibility_timeout(task.task_id)

        self.logger.info(
            "completed task",
            extra={
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
                "queue": self.queue_name,
            },
        )

    def fail(self, task: AgentTask, error: str) -> None:
        """Mark a task as failed, potentially requeueing for retry."""
        task.error = error

        if task.attempt < task.max_attempts:
            # Requeue for retry
            task.status = TaskStatus.RETRYING
            task_key = f"task:{task.task_id}"
            timeout_key = f"task:{task.task_id}:timeout"
            requeued = self._redis_call(
                self.redis.eval,
                self._retry_requeue_script,
                5,
                task_key,
                self.processing_queue,
                self.queue_name,
                timeout_key,
                self.visibility_index,
                task.task_id,
                self.body_ttl_seconds,
                task.to_json(),
            )
            if not requeued:
                raise RuntimeError(
                    f"Task {task.task_id} was not in processing queue"
                )

            self.logger.warning(
                "task failed; requeueing",
                extra={
                    "task_id": task.task_id,
                    "conversation_id": task.conversation_id,
                    "attempt": task.attempt,
                    "max_attempts": task.max_attempts,
                    "queue": self.queue_name,
                },
            )
        else:
            # Move to dead letter queue
            task.status = TaskStatus.FAILED
            task_key = f"task:{task.task_id}"
            self._redis_call(
                self.redis.setex, task_key, self.body_ttl_seconds, task.to_json()
            )

            self._redis_call(self.redis.lrem, self.processing_queue, 1, task.task_id)
            self._redis_call(self.redis.lpush, self.dead_letter_queue, task.task_id)

            self.logger.error(
                "task permanently failed",
                extra={
                    "task_id": task.task_id,
                    "conversation_id": task.conversation_id,
                    "attempt": task.attempt,
                    "max_attempts": task.max_attempts,
                    "queue": self.dead_letter_queue,
                },
            )

        self._clear_visibility_timeout(task.task_id)

    def get_queue_depth(self) -> dict[str, int]:
        """Get the current depth of all queues."""
        return {
            "pending": self._redis_call(self.redis.llen, self.queue_name),
            "processing": self._redis_call(self.redis.llen, self.processing_queue),
            "dead_letter": self._redis_call(self.redis.llen, self.dead_letter_queue),
        }

    def _set_visibility_timeout(self, task_id: str) -> None:
        """Set a timeout for task processing."""
        timeout_key = f"task:{task_id}:timeout"
        deadline = time.time() + self.visibility_timeout
        self._redis_call(
            self.redis.setex,
            timeout_key, self.visibility_timeout, str(deadline)
        )
        self._redis_call(
            self.redis.zadd,
            self.visibility_index,
            {task_id: deadline},
        )

    def _clear_visibility_timeout(self, task_id: str) -> None:
        """Clear the visibility timeout."""
        timeout_key = f"task:{task_id}:timeout"
        self._redis_call(self.redis.delete, timeout_key)
        self._redis_call(self.redis.zrem, self.visibility_index, task_id)

    def return_to_head_after_backpressure(self, task: AgentTask) -> None:
        """Atomically undo a shaping-only dequeue and make it next to claim."""
        if task.attempt > 0:
            task.attempt -= 1
        task.status = TaskStatus.PENDING
        task_key = f"task:{task.task_id}"
        timeout_key = f"task:{task.task_id}:timeout"
        returned = self._redis_call(
            self.redis.eval,
            self._return_to_head_script,
            4,
            task_key,
            self.processing_queue,
            timeout_key,
            self.queue_name,
            task.task_id,
            self.body_ttl_seconds,
            task.to_json(),
        )
        if not returned:
            raise RuntimeError(
                f"Task {task.task_id} was not in processing queue"
            )

    def recover_stale_tasks(self, batch_size: int = 100) -> int:
        """
        Recover tasks that have exceeded visibility timeout.
        Should be called periodically by a maintenance process.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        recovered = 0
        processing_tasks = self._redis_call(
            self.redis.zrangebyscore,
            self.visibility_index,
            "-inf",
            time.time(),
            start=0,
            num=batch_size,
        )

        for task_id in processing_tasks:
            timeout_key = f"task:{task_id}:timeout"
            recovered_one = self._redis_call(
                self.redis.eval,
                self._recover_stale_script,
                4,
                self.processing_queue,
                self.queue_name,
                self.visibility_index,
                timeout_key,
                task_id,
            )
            if recovered_one:
                recovered += 1
                self.logger.warning(
                    "recovered stale task",
                    extra={"task_id": task_id, "queue": self.queue_name},
                )

        return recovered

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

try:
    import pika
except ImportError:
    pika = None  # type: ignore[assignment]
import json
from typing import Any, Callable, Optional
import logging
from dataclasses import dataclass


@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ connection."""

    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    heartbeat: int = 600
    connection_timeout: int = 30
    message_ttl_ms: int = 3_600_000
    publish_max_retries: int = 3
    publish_retry_backoff: float = 0.25

    def __post_init__(self) -> None:
        if self.publish_max_retries < 1:
            raise ValueError("publish_max_retries must be >= 1")
        if self.publish_retry_backoff < 0:
            raise ValueError("publish_retry_backoff must be >= 0")


class RabbitMQTaskQueue:
    """RabbitMQ-based task queue with advanced features."""

    def __init__(
        self,
        config: RabbitMQConfig,
        exchange_name: str = "agent_exchange",
        queue_name: str = "agent_tasks",
    ) -> None:
        if pika is None:
            raise ImportError("pika is required for RabbitMQTaskQueue")
        self.config = config
        self.exchange_name = exchange_name
        self.queue_name = queue_name
        self.dead_letter_exchange = f"{exchange_name}_dlx"
        self.dead_letter_queue = f"{queue_name}_dlq"
        self.logger = logging.getLogger(__name__)

        self.connection: Optional[Any] = None
        self.channel: Optional[Any] = None

    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        credentials = pika.PlainCredentials(
            self.config.username, self.config.password
        )

        parameters = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            virtual_host=self.config.virtual_host,
            credentials=credentials,
            heartbeat=self.config.heartbeat,
            connection_attempts=3,
            retry_delay=5,
            socket_timeout=self.config.connection_timeout,
            stack_timeout=self.config.connection_timeout,
            blocked_connection_timeout=self.config.connection_timeout,
        )

        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()

        self._setup_exchanges_and_queues()

    def _setup_exchanges_and_queues(self) -> None:
        """Declare exchanges and queues with dead letter routing."""
        # Dead letter exchange and queue
        self.channel.exchange_declare(
            exchange=self.dead_letter_exchange,
            exchange_type="direct",
            durable=True,
        )

        self.channel.queue_declare(queue=self.dead_letter_queue, durable=True)

        self.channel.queue_bind(
            queue=self.dead_letter_queue,
            exchange=self.dead_letter_exchange,
            routing_key=self.queue_name,
        )

        # Main exchange and queue
        self.channel.exchange_declare(
            exchange=self.exchange_name, exchange_type="direct", durable=True
        )

        # Queue with dead letter configuration
        self.channel.queue_declare(
            queue=self.queue_name,
            durable=True,
            arguments={
                "x-dead-letter-exchange": self.dead_letter_exchange,
                "x-dead-letter-routing-key": self.queue_name,
                "x-message-ttl": self.config.message_ttl_ms,
            },
        )

        self.channel.queue_bind(
            queue=self.queue_name,
            exchange=self.exchange_name,
            routing_key=self.queue_name,
        )

    def publish(self, task: AgentTask, priority: int = 5) -> None:
        """Publish a task to the queue."""
        message = task.to_json()

        last_error: Optional[BaseException] = None
        success = False
        for attempt in range(self.config.publish_max_retries):
            try:
                if self.channel is None or self.channel.is_closed:
                    self.connect()
                assert self.channel is not None
                self.channel.basic_publish(
                    exchange=self.exchange_name,
                    routing_key=self.queue_name,
                    body=message,
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # Persistent
                        priority=priority,
                        content_type="application/json",
                        message_id=task.task_id,
                    ),
                    mandatory=True,
                )
                success = True
                break
            except pika.exceptions.AMQPError as exc:
                last_error = exc
                if attempt == self.config.publish_max_retries - 1:
                    break
                time.sleep(self.config.publish_retry_backoff * (2**attempt))
        if not success:
            raise RuntimeError("RabbitMQ publish failed") from last_error

        self.logger.info(f"Published task {task.task_id}")

    def consume(
        self, callback: Callable[[AgentTask], bool], prefetch_count: int = 1
    ) -> None:
        """
        Start consuming tasks from the queue.
        Callback should return True on success, False on failure.
        """
        if self.channel is None or self.channel.is_closed:
            self.connect()
        assert self.channel is not None
        self.channel.basic_qos(prefetch_count=prefetch_count)

        def on_message(
            channel: Any, method: Any, properties: Any, body: bytes
        ) -> None:
            try:
                task = AgentTask.from_json(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
                self.logger.warning("Invalid task payload sent to DLQ: %s", exc)
                channel.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
                )
                return

            try:
                success = callback(task)
            except (TimeoutError, RuntimeError, OSError) as exc:
                self.logger.exception("Task callback failed; sending to DLQ: %s", exc)
                channel.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
                )
                return

            if success:
                channel.basic_ack(delivery_tag=method.delivery_tag)
            else:
                # Requeue with delay by rejecting
                channel.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=True
                )

        self.channel.basic_consume(
            queue=self.queue_name, on_message_callback=on_message
        )

        self.logger.info("Starting to consume messages")
        self.channel.start_consuming()

    def close(self) -> None:
        """Close the connection."""
        if self.connection and self.connection.is_open:
            self.connection.close()

# ============================================================================
# Block 10 (chapter listing #10)
# ============================================================================

try:
    import boto3
    from botocore.config import Config
except ImportError:
    boto3 = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]
import json
from typing import Optional, Generator
import logging


class SQSTaskQueue:
    """AWS SQS-based task queue for cloud deployments."""

    def __init__(
        self,
        queue_url: str,
        dead_letter_queue_url: Optional[str] = None,
        region_name: str = "us-east-1",
        visibility_timeout: int = 300,
        max_receive_count: int = 3,
    ) -> None:
        if boto3 is None or Config is None:
            raise ImportError("boto3 and botocore are required for SQSTaskQueue")
        self.sqs = boto3.client(
            "sqs",
            region_name=region_name,
            config=Config(
                connect_timeout=2,
                read_timeout=10,
                max_pool_connections=50,
                retries={"max_attempts": 4, "mode": "adaptive"},
            ),
        )
        self.queue_url = queue_url
        self.dlq_url = dead_letter_queue_url
        self.visibility_timeout = visibility_timeout
        self.max_receive_count = max_receive_count
        self.logger = logging.getLogger(__name__)

    def send_task(self, task: AgentTask, delay_seconds: int = 0) -> str:
        """Send a task to the queue."""
        response = self.sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=task.to_json(),
            DelaySeconds=delay_seconds,
            MessageAttributes={
                "TaskId": {"StringValue": task.task_id, "DataType": "String"},
                "ConversationId": {
                    "StringValue": task.conversation_id,
                    "DataType": "String",
                },
            },
        )

        message_id = response["MessageId"]
        self.logger.info(f"Sent task {task.task_id} as message {message_id}")
        return message_id

    def receive_tasks(
        self, max_messages: int = 10, wait_time_seconds: int = 20
    ) -> Generator[tuple[AgentTask, str], None, None]:
        """
        Receive tasks from the queue.
        Yields (task, receipt_handle) tuples.
        """
        response = self.sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=min(max_messages, 10),
            WaitTimeSeconds=wait_time_seconds,
            VisibilityTimeout=self.visibility_timeout,
            MessageAttributeNames=["All"],
        )

        messages = response.get("Messages", [])

        for message in messages:
            task = AgentTask.from_json(message["Body"])
            yield task, message["ReceiptHandle"]

    def delete_task(self, receipt_handle: str) -> None:
        """Delete a successfully processed task."""
        self.sqs.delete_message(
            QueueUrl=self.queue_url, ReceiptHandle=receipt_handle
        )

    def extend_visibility(
        self, receipt_handle: str, additional_seconds: int
    ) -> None:
        """Extend visibility timeout for long-running tasks."""
        self.sqs.change_message_visibility(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=additional_seconds,
        )

    def get_queue_attributes(self) -> dict[str, int]:
        """Get queue metrics for monitoring."""
        response = self.sqs.get_queue_attributes(
            QueueUrl=self.queue_url,
            AttributeNames=[
                "ApproximateNumberOfMessages",
                "ApproximateNumberOfMessagesNotVisible",
                "ApproximateNumberOfMessagesDelayed",
            ],
        )

        attrs = response["Attributes"]
        return {
            "pending": int(attrs["ApproximateNumberOfMessages"]),
            "processing": int(attrs["ApproximateNumberOfMessagesNotVisible"]),
            "delayed": int(attrs["ApproximateNumberOfMessagesDelayed"]),
        }

# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

import asyncio
import signal
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor
import threading


class WorkerState(Enum):
    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"


@dataclass
class WorkerStats:
    """Statistics for a single worker."""

    worker_id: str
    state: WorkerState = WorkerState.IDLE
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_processing_time: float = 0.0
    current_task_start: Optional[float] = None
    last_heartbeat: float = field(default_factory=time.time)

    @property
    def average_processing_time(self) -> float:
        if self.tasks_completed == 0:
            return 0.0
        return self.total_processing_time / self.tasks_completed


class AgentProcessingError(Exception):
    """Expected agent/provider failure that should fail a task, not the worker."""


class AgentWorker:
    """Individual worker that processes agent tasks."""

    def __init__(
        self,
        worker_id: str,
        task_queue: RedisTaskQueue,
        agent: StatelessAgent,
        heartbeat_interval: int = 30,
        error_backoff_base: float = 1.0,
        error_backoff_max: float = 30.0,
        rate_limiter: Optional[Any] = None,
        agent_error_types: Optional[tuple[type[BaseException], ...]] = None,
    ) -> None:
        if error_backoff_base <= 0:
            raise ValueError("error_backoff_base must be > 0")
        if error_backoff_max < error_backoff_base:
            raise ValueError(
                "error_backoff_max must be >= error_backoff_base"
            )
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.agent = agent
        self.heartbeat_interval = heartbeat_interval
        self.error_backoff_base = error_backoff_base
        self.error_backoff_max = error_backoff_max
        self.rate_limiter = rate_limiter
        self.agent_error_types = (
            RedisError,
            TimeoutError,
            ConnectionError,
            OSError,
            AgentProcessingError,
            StateConflictError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            AttributeError,
            *(agent_error_types or ()),
        )
        self.stats = WorkerStats(worker_id=worker_id)
        self.logger = logging.getLogger(f"worker.{worker_id}")
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Main worker loop."""
        self.logger.info(f"Worker {self.worker_id} starting")
        consecutive_errors = 0

        while not self._stop_event.is_set():
            try:
                # Dequeue with timeout to allow checking stop event
                task = self.task_queue.dequeue(timeout=5)

                if task is None:
                    self._update_heartbeat()
                    consecutive_errors = 0
                    continue

                if self.rate_limiter is not None:
                    if not self.rate_limiter.acquire(timeout=5.0):
                        self.task_queue.return_to_head_after_backpressure(task)
                        self._update_heartbeat()
                        continue

                start_time = time.time()
                self._process_task(task)
                if self.rate_limiter is not None:
                    latency_ms = (time.time() - start_time) * 1000
                    self.rate_limiter.record_latency(latency_ms)
                consecutive_errors = 0

            except (RedisError, TimeoutError, ConnectionError, OSError) as e:
                self.logger.error(
                    "worker loop error",
                    extra={
                        "worker_id": self.worker_id,
                        "error_type": type(e).__name__,
                        "consecutive_errors": consecutive_errors,
                    },
                    exc_info=True,
                )
                delay = min(
                    self.error_backoff_max,
                    self.error_backoff_base
                    * (2 ** min(consecutive_errors, 10)),
                )
                consecutive_errors += 1
                time.sleep(delay)

        self.stats.state = WorkerState.STOPPED
        self.logger.info("worker stopped", extra={"worker_id": self.worker_id})

    def _process_task(self, task: AgentTask) -> None:
        """Process a single task."""
        self.stats.state = WorkerState.BUSY
        self.stats.current_task_start = time.time()

        self.logger.info(
            "processing task",
            extra={
                "worker_id": self.worker_id,
                "task_id": task.task_id,
                "conversation_id": task.conversation_id,
            },
        )

        try:
            result = self.agent.process_message(
                task.conversation_id, task.message
            )

            self.task_queue.complete(task, result)

            processing_time = time.time() - self.stats.current_task_start
            self.stats.tasks_completed += 1
            self.stats.total_processing_time += processing_time

            self.logger.info(
                "completed task",
                extra={
                    "worker_id": self.worker_id,
                    "task_id": task.task_id,
                    "conversation_id": task.conversation_id,
                    "processing_time_seconds": processing_time,
                },
            )

        except self.agent_error_types as e:
            self.logger.error(
                "task failed",
                extra={
                    "worker_id": self.worker_id,
                    "task_id": task.task_id,
                    "conversation_id": task.conversation_id,
                    "error_type": type(e).__name__,
                },
                exc_info=True,
            )
            self.task_queue.fail(task, str(e))
            self.stats.tasks_failed += 1

        finally:
            self.stats.state = WorkerState.IDLE
            self.stats.current_task_start = None
            self._update_heartbeat()

    def _update_heartbeat(self) -> None:
        """Update the worker's heartbeat timestamp."""
        self.stats.last_heartbeat = time.time()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self.stats.state = WorkerState.STOPPING
        self._stop_event.set()


class WorkerPool:
    """Manages a pool of agent workers."""

    def __init__(
        self,
        task_queue: RedisTaskQueue,
        agent_factory: Callable[[], StatelessAgent],
        min_workers: int = 2,
        max_workers: int = 10,
        worker_idle_timeout: int = 300,
        stale_heartbeat_seconds: int = 120,
        rate_limiter: Optional[Any] = None,
    ) -> None:
        if min_workers < 1:
            raise ValueError("min_workers must be >= 1")
        if max_workers < min_workers:
            raise ValueError("max_workers must be >= min_workers")
        if worker_idle_timeout <= 0:
            raise ValueError("worker_idle_timeout must be > 0")
        if stale_heartbeat_seconds < 1:
            raise ValueError("stale_heartbeat_seconds must be >= 1")
        self.task_queue = task_queue
        self.agent_factory = agent_factory
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_idle_timeout = worker_idle_timeout
        self.stale_heartbeat_seconds = stale_heartbeat_seconds
        self.rate_limiter = rate_limiter

        self.workers: dict[str, AgentWorker] = {}
        self.threads: dict[str, threading.Thread] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger("worker_pool")

        self._shutdown_event = threading.Event()
        self._maintenance_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the worker pool with minimum workers."""
        self.logger.info(
            f"Starting worker pool with {self.min_workers} workers"
        )

        for i in range(self.min_workers):
            self._add_worker()

        # Start maintenance thread
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop, daemon=True
        )
        self._maintenance_thread.start()

    def _add_worker(self) -> Optional[str]:
        """Add a new worker to the pool."""
        with self.lock:
            if len(self.workers) >= self.max_workers:
                return None

            # uuid4 (not ``len(workers)``) so removed-then-re-added
            # workers do not collide with surviving ones.
            worker_id = f"worker-{uuid.uuid4().hex[:8]}"
            agent = self.agent_factory()

            worker = AgentWorker(
                worker_id=worker_id,
                task_queue=self.task_queue,
                agent=agent,
                rate_limiter=self.rate_limiter,
            )

            thread = threading.Thread(
                target=worker.run, name=worker_id, daemon=True
            )

            self.workers[worker_id] = worker
            self.threads[worker_id] = thread

            thread.start()

            self.logger.info(f"Added worker {worker_id}")
            return worker_id

    def set_rate_limiter(self, rate_limiter: Optional[Any]) -> None:
        """Attach a shared rate limiter to existing and future workers."""
        with self.lock:
            self.rate_limiter = rate_limiter
            for worker in self.workers.values():
                worker.rate_limiter = rate_limiter

    def _remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the pool.

        We snapshot worker/thread refs and signal stop under the lock,
        then release the lock before joining. ``thread.join`` can take
        up to the configured timeout (default 60s) while a worker
        finishes its current task; holding the registry lock for that
        long would block scale-up, health checks, and stats queries.
        """
        with self.lock:
            if worker_id not in self.workers:
                return

            if len(self.workers) <= self.min_workers:
                return

            worker = self.workers.pop(worker_id)
            thread = self.threads.pop(worker_id)
            worker.stop()

        # Wait for worker to finish current task OUTSIDE the lock so
        # other pool operations are not blocked for up to a minute.
        thread.join(timeout=60)

        self.logger.info(f"Removed worker {worker_id}")

    def scale_to(self, target_workers: int) -> None:
        """Scale the pool to the target number of workers."""
        target = max(self.min_workers, min(target_workers, self.max_workers))
        current = len(self.workers)

        if target > current:
            for _ in range(target - current):
                self._add_worker()
        elif target < current:
            # Remove idle workers first
            idle_workers = [
                w
                for w in self.workers.values()
                if w.stats.state == WorkerState.IDLE
            ]

            for worker in idle_workers[: current - target]:
                self._remove_worker(worker.worker_id)

    def _maintenance_loop(self) -> None:
        """Periodic maintenance: health checks, cleanup.

        The narrow ``except`` below covers the expected transient
        Redis/network/OS failures, but an unanticipated error class
        would otherwise terminate this daemon thread silently and
        leave the pool without health-check coverage. The broad
        fallback catch logs the unexpected exception and waits a
        bounded interval before retrying so a supervisor can still see
        the failures while the loop stays alive.
        """
        while not self._shutdown_event.is_set():
            try:
                self._check_worker_health()
                self._recover_stale_tasks()
            except (RedisError, TimeoutError, ConnectionError, OSError) as e:
                self.logger.exception("Maintenance error: %s", e)
            except Exception as exc:  # noqa: BLE001 - daemon must stay alive
                self.logger.exception(
                    "Unexpected maintenance error; loop continuing: %s", exc
                )
                # Brief sleep to avoid a tight error loop if the same
                # unexpected exception fires on every iteration.
                time.sleep(1.0)

            self._shutdown_event.wait(timeout=30)

    def _check_worker_health(self) -> None:
        """Check worker health and restart unhealthy workers.

        Stale-heartbeat detection only flags; it does not yet restart
        the worker (operators usually want to investigate why a worker
        wedged before recycling it). We emit a counter metric so an
        external alert can fire, and log at WARNING so the line is
        searchable.
        """
        now = time.time()
        stale_seconds = self.stale_heartbeat_seconds

        with self.lock:
            for worker_id, worker in list(self.workers.items()):
                # Check if worker is alive
                thread = self.threads.get(worker_id)
                if thread and not thread.is_alive():
                    self.logger.warning(f"Worker {worker_id} died, removing")
                    self._emit_metric(
                        "worker_pool.worker_died", 1, worker_id=worker_id
                    )
                    del self.workers[worker_id]
                    del self.threads[worker_id]
                    continue

                # Check heartbeat
                heartbeat_age = now - worker.stats.last_heartbeat
                if heartbeat_age > stale_seconds:
                    self.logger.warning(
                        "Worker %s heartbeat stale (age=%.1fs, threshold=%ss)",
                        worker_id,
                        heartbeat_age,
                        stale_seconds,
                    )
                    self._emit_metric(
                        "worker_pool.heartbeat_stale",
                        1,
                        worker_id=worker_id,
                        age_seconds=heartbeat_age,
                    )

    def _emit_metric(self, name: str, value: float, **tags: Any) -> None:
        """Emit a counter metric.

        Default implementation is a structured log line so that the
        chapter examples stay framework-agnostic; production callers
        should override this (or inject a metrics client) to push to
        Prometheus, statsd, CloudWatch, etc.
        """
        self.logger.info(
            "metric name=%s value=%s tags=%s", name, value, tags
        )

    def _recover_stale_tasks(self) -> None:
        """Recover tasks that have timed out."""
        recovered = self.task_queue.recover_stale_tasks()
        if recovered > 0:
            self.logger.info(f"Recovered {recovered} stale tasks")

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            worker_count = len(self.workers)
            worker_stats = {
                wid: {
                    "state": w.stats.state.value,
                    "tasks_completed": w.stats.tasks_completed,
                    "tasks_failed": w.stats.tasks_failed,
                    "avg_processing_time": w.stats.average_processing_time,
                }
                for wid, w in self.workers.items()
            }

        queue_depth = self.task_queue.get_queue_depth()

        return {
            "worker_count": worker_count,
            "workers": worker_stats,
            "queue": queue_depth,
        }

    def shutdown(self, timeout: int = 60) -> None:
        """Gracefully shutdown the worker pool."""
        self.logger.info("Shutting down worker pool")
        self._shutdown_event.set()

        # Join the maintenance thread explicitly so its scaling and
        # health-check ticks cannot race the worker teardown below.
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=timeout)

        # Snapshot under the lock; worker health checks and scaling can
        # mutate these dictionaries while shutdown is running.
        with self.lock:
            workers = list(self.workers.values())
            threads = list(self.threads.values())

        # Stop all workers outside the lock so worker.stop() callbacks
        # cannot deadlock against maintenance paths.
        for worker in workers:
            worker.stop()

        # Wait for threads to finish without extending the total timeout
        # by ``timeout * thread_count``.
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

        self.logger.info("Worker pool shutdown complete")

# ============================================================================
# Block 12 (chapter listing #12)
# ============================================================================

import time
import threading
from dataclasses import dataclass
from typing import Optional
import logging
from collections import deque


@dataclass
class BackpressureConfig:
    """Configuration for backpressure control."""

    target_latency_ms: float = 1000.0  # Target processing latency
    max_latency_ms: float = 5000.0  # Maximum acceptable latency
    min_rate: float = 1.0  # Minimum tasks per second
    max_rate: float = 100.0  # Maximum tasks per second
    window_size: int = 100  # Sliding window for metrics


class AdaptiveRateLimiter:
    """Rate limiter that adjusts based on system performance."""

    def __init__(self, config: BackpressureConfig) -> None:
        self.config = config
        self.current_rate = config.max_rate / 2  # Start at midpoint
        self.latencies: deque[float] = deque(maxlen=config.window_size)
        self.last_adjustment = time.time()
        self.adjustment_interval = 5.0  # Seconds between adjustments
        self.lock = threading.Lock()
        self.logger = logging.getLogger("rate_limiter")

        # Token bucket state
        self.tokens = self.current_rate
        self.last_token_time = time.time()

    def record_latency(self, latency_ms: float) -> None:
        """Record a task's processing latency."""
        with self.lock:
            self.latencies.append(latency_ms)
            self._maybe_adjust_rate()

    def _maybe_adjust_rate(self) -> None:
        """Adjust rate based on recent latencies."""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return

        if len(self.latencies) < 10:
            return

        self.last_adjustment = now

        # Calculate percentile latencies
        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]

        # Adjust rate based on p95 latency
        if p95 > self.config.max_latency_ms:
            # Decrease rate aggressively
            self.current_rate = max(
                self.config.min_rate, self.current_rate * 0.5
            )
            self.logger.warning(
                f"High latency ({p95:.0f}ms), reducing rate to "
                f"{self.current_rate:.1f}/s"
            )
        elif p95 > self.config.target_latency_ms:
            # Decrease rate gradually
            self.current_rate = max(
                self.config.min_rate, self.current_rate * 0.9
            )
        elif p50 < self.config.target_latency_ms * 0.5:
            # Increase rate if we have headroom
            self.current_rate = min(
                self.config.max_rate, self.current_rate * 1.1
            )

    def acquire(self, timeout: float = 30.0) -> bool:
        """
        Acquire permission to process a task.
        Uses token bucket algorithm.
        """
        start = time.time()

        while time.time() - start < timeout:
            with self.lock:
                # Refill tokens based on elapsed time
                now = time.time()
                elapsed = now - self.last_token_time
                self.tokens = min(
                    self.current_rate,
                    self.tokens + elapsed * self.current_rate,
                )
                self.last_token_time = now

                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return True

                # Wait at least until the next token would be ready.
                # Sleeping outside the lock so other threads can refill
                # in parallel; capped by the remaining timeout.
                if self.current_rate > 0:
                    next_token_in = (1.0 - self.tokens) / self.current_rate
                else:
                    next_token_in = 0.05
            wait = max(0.001, min(next_token_in, timeout - (time.time() - start)))
            time.sleep(wait)

        return False

    def get_current_rate(self) -> float:
        """Get the current allowed rate."""
        with self.lock:
            return self.current_rate


class BackpressureQueueConsumer:
    """Queue consumer with adaptive backpressure control.

    Intentionally a stub: this class shapes inbound load only --- see
    ``RedisTaskQueue`` (the producer/queue) and ``AgentWorker`` inside
    ``WorkerPool`` for the actual dispatch loop. The shared
    ``AdaptiveRateLimiter`` is enforced inside ``AgentWorker.process``
    just before processing begins, so this consumer only observes queue
    depth and updates the rate-limiter signal; it does not pop tasks.
    """

    def __init__(
        self,
        task_queue: RedisTaskQueue,
        worker_pool: WorkerPool,
        backpressure_config: Optional[BackpressureConfig] = None,
    ) -> None:
        self.task_queue = task_queue
        self.worker_pool = worker_pool
        self.rate_limiter = AdaptiveRateLimiter(
            backpressure_config or BackpressureConfig()
        )
        self.worker_pool.set_rate_limiter(self.rate_limiter)
        self.logger = logging.getLogger("consumer")
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Run the backpressure observer loop.

        This is intentionally not a dispatch loop: workers in
        ``WorkerPool`` self-pull from ``RedisTaskQueue`` and consult the
        shared ``AdaptiveRateLimiter`` before processing. This method
        only watches queue depth and lets the rate limiter react.
        """
        self.logger.info("Starting backpressure-controlled consumer")

        while not self._stop_event.is_set():
            # Check queue depth
            queue_depth = self.task_queue.get_queue_depth()

            if queue_depth["pending"] == 0:
                # No tasks, wait before checking again
                self._stop_event.wait(timeout=1.0)
                continue

            # The shared limiter is enforced inside AgentWorker, just
            # before processing begins. Keeping the token check there
            # bounds all workers that self-pull from the queue.
            self.logger.debug(
                "Backpressure active: pending=%s current_rate=%.2f/s",
                queue_depth["pending"],
                self.rate_limiter.get_current_rate(),
            )
            self._stop_event.wait(timeout=1.0)

    def _dispatch_task(self, task: AgentTask) -> None:
        """Return a dequeued task to the head of the queue.

        Compensate for the side effects of the dequeue itself:
        ``RedisTaskQueue.dequeue`` increments ``task.attempt`` and
        installs a visibility-timeout marker, both of which are
        accounting that should only happen on real processing
        attempts. Without rolling these back, every pass through the
        rate limiter burns one retry from the task's retry budget
        and could push the task into the dead-letter queue without
        a worker ever touching it.
        """
        self.task_queue.return_to_head_after_backpressure(task)
        self.logger.debug(
            "Backpressure dispatch: returned task %s to head of queue",
            task.task_id,
        )

    def stop(self) -> None:
        """Stop the consumer."""
        self._stop_event.set()

# ============================================================================
# Block 13 (chapter listing #13)
# ============================================================================

import time
import threading
from dataclasses import dataclass, field
from typing import Optional
import logging
from collections import deque
import math


@dataclass
class AutoScalerConfig:
    """Configuration for auto-scaling behavior."""

    min_workers: int = 2
    max_workers: int = 20

    # Scale-up thresholds
    scale_up_queue_threshold: int = 50  # Queue depth to trigger scale up
    scale_up_latency_threshold: float = 2000.0  # ms

    # Scale-down thresholds
    scale_down_queue_threshold: int = 10
    scale_down_idle_time: float = 300.0  # seconds
    scale_down_idle_buffer: int = 1

    # Cooldown periods
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 300.0  # seconds

    # Per-decision limits
    scale_up_max_step: int = 5
    scale_down_max_step: int = 2
    queue_drain_target_minutes: float = 5.0

    # Evaluation interval
    evaluation_interval: float = 30.0  # seconds
    metrics_retention_seconds: float = 30 * 60  # keep 30 minutes

    def __post_init__(self) -> None:
        if self.min_workers < 1:
            raise ValueError("min_workers must be >= 1")
        if self.max_workers < self.min_workers:
            raise ValueError("max_workers must be >= min_workers")
        if self.evaluation_interval <= 0:
            raise ValueError("evaluation_interval must be > 0")
        if self.metrics_retention_seconds <= 0:
            raise ValueError("metrics_retention_seconds must be > 0")
        if self.scale_up_max_step < 1:
            raise ValueError("scale_up_max_step must be >= 1")
        if self.scale_down_max_step < 1:
            raise ValueError("scale_down_max_step must be >= 1")
        if self.scale_down_idle_buffer < 0:
            raise ValueError("scale_down_idle_buffer must be >= 0")
        if self.queue_drain_target_minutes <= 0:
            raise ValueError("queue_drain_target_minutes must be > 0")


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""

    queue_depth: int = 0
    processing_count: int = 0
    avg_latency_ms: float = 0.0
    worker_count: int = 0
    idle_workers: int = 0
    timestamp: float = field(default_factory=time.time)


class AutoScaler:
    """Auto-scales worker pool based on queue depth and latency."""

    def __init__(
        self,
        worker_pool: WorkerPool,
        task_queue: RedisTaskQueue,
        config: Optional[AutoScalerConfig] = None,
    ) -> None:
        self.worker_pool = worker_pool
        self.task_queue = task_queue
        self.config = config or AutoScalerConfig()

        history_len = max(
            1,
            math.ceil(
                self.config.metrics_retention_seconds
                / self.config.evaluation_interval
            ),
        )
        self.metrics_history: deque[ScalingMetrics] = deque(
            maxlen=history_len
        )
        self.last_scale_up = 0.0
        self.last_scale_down = 0.0

        self.logger = logging.getLogger("autoscaler")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the auto-scaler background thread."""
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="autoscaler"
        )
        self._thread.start()
        self.logger.info("Auto-scaler started")

    def _run(self) -> None:
        """Main auto-scaler loop."""
        while not self._stop_event.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)

                decision = self._evaluate_scaling(metrics)

                if decision > 0:
                    self._scale_up(decision)
                elif decision < 0:
                    self._scale_down(abs(decision))

            except (
                TimeoutError,
                ConnectionError,
                OSError,
                RuntimeError,
                ValueError,
            ) as e:
                self.logger.exception("Auto-scaler error: %s", e)
            except Exception as exc:
                # Generic fallback so the auto-scaler daemon thread
                # cannot die on an unanticipated error class. The loop
                # then sleeps and tries again on the next interval.
                self.logger.exception(
                    "Auto-scaler unexpected error: %s", exc
                )

            self._stop_event.wait(timeout=self.config.evaluation_interval)

    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""
        pool_stats = self.worker_pool.get_stats()
        queue_depth = pool_stats["queue"]

        # Calculate average latency from worker stats
        latencies = []
        idle_count = 0

        for worker_stats in pool_stats["workers"].values():
            if worker_stats["avg_processing_time"] > 0:
                latencies.append(worker_stats["avg_processing_time"] * 1000)
            if worker_stats["state"] == "idle":
                idle_count += 1

        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return ScalingMetrics(
            queue_depth=queue_depth["pending"],
            processing_count=queue_depth["processing"],
            avg_latency_ms=avg_latency,
            worker_count=pool_stats["worker_count"],
            idle_workers=idle_count,
        )

    def _evaluate_scaling(self, metrics: ScalingMetrics) -> int:
        """
        Evaluate whether to scale.
        Returns: positive for scale up, negative for scale down, 0 for no change
        """
        now = time.time()

        # Check for scale-up conditions
        should_scale_up = False
        scale_up_reason = ""

        if metrics.queue_depth > self.config.scale_up_queue_threshold:
            should_scale_up = True
            scale_up_reason = f"queue_depth={metrics.queue_depth}"

        if metrics.avg_latency_ms > self.config.scale_up_latency_threshold:
            should_scale_up = True
            scale_up_reason = f"latency={metrics.avg_latency_ms:.0f}ms"

        if should_scale_up:
            if now - self.last_scale_up < self.config.scale_up_cooldown:
                self.logger.debug("Scale-up needed but in cooldown")
                return 0

            if metrics.worker_count >= self.config.max_workers:
                self.logger.warning("Scale-up needed but at max workers")
                return 0

            # Calculate how many workers to add
            workers_needed = self._calculate_workers_needed(metrics)
            workers_to_add = min(
                workers_needed - metrics.worker_count,
                self.config.max_workers - metrics.worker_count,
                self.config.scale_up_max_step,
            )

            if workers_to_add > 0:
                self.logger.info(
                    f"Scaling up by {workers_to_add}: {scale_up_reason}"
                )
                return workers_to_add

        # Check for scale-down conditions
        if metrics.worker_count > self.config.min_workers:
            if (
                metrics.queue_depth < self.config.scale_down_queue_threshold
                and metrics.idle_workers > self.config.scale_down_idle_buffer
            ):

                if (
                    now - self.last_scale_down
                    < self.config.scale_down_cooldown
                ):
                    return 0

                # Scale down conservatively
                workers_to_remove = min(
                    metrics.idle_workers - self.config.scale_down_idle_buffer,
                    metrics.worker_count - self.config.min_workers,
                    self.config.scale_down_max_step,
                )

                if workers_to_remove > 0:
                    self.logger.info(
                        f"Scaling down by {workers_to_remove}: "
                        f"idle_workers={metrics.idle_workers}"
                    )
                    return -workers_to_remove

        return 0

    def _calculate_workers_needed(self, metrics: ScalingMetrics) -> int:
        """Calculate ideal worker count based on current metrics."""
        if metrics.avg_latency_ms <= 0:
            # No latency data, estimate based on queue
            return metrics.worker_count + (metrics.queue_depth // 10)

        # Estimate throughput per worker (tasks per minute)
        tasks_per_minute = 60000.0 / metrics.avg_latency_ms

        # Calculate workers needed to drain the queue within the configured
        # window while handling incoming work.
        incoming_rate = self._estimate_incoming_rate()
        target_minutes = self.config.queue_drain_target_minutes
        total_work = metrics.queue_depth + (incoming_rate * target_minutes)

        workers_needed = int(total_work / (tasks_per_minute * target_minutes)) + 1

        return workers_needed

    def _estimate_incoming_rate(self) -> float:
        """Estimate incoming task rate from recent history."""
        if len(self.metrics_history) < 2:
            return 0.0

        # Look at queue growth over recent intervals
        recent = list(self.metrics_history)[-10:]
        if len(recent) < 2:
            return 0.0

        # Simple rate estimation
        queue_changes = [
            recent[i + 1].queue_depth - recent[i].queue_depth
            for i in range(len(recent) - 1)
        ]

        avg_change = sum(queue_changes) / len(queue_changes)
        return max(0.0, avg_change / self.config.evaluation_interval * 60)

    def _scale_up(self, count: int) -> None:
        """Execute scale-up."""
        target = self.worker_pool.get_stats()["worker_count"] + count
        self.worker_pool.scale_to(target)
        self.last_scale_up = time.time()

    def _scale_down(self, count: int) -> None:
        """Execute scale-down."""
        target = self.worker_pool.get_stats()["worker_count"] - count
        self.worker_pool.scale_to(target)
        self.last_scale_down = time.time()

    def stop(self) -> None:
        """Stop the auto-scaler."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

import hashlib
from typing import Optional, Callable, Any
try:
    import redis
    from redis.exceptions import RedisError
except ImportError:
    redis = None  # type: ignore[assignment]

    class RedisError(Exception):
        pass
from dataclasses import dataclass
import time


@dataclass
class ShardConfig:
    """Configuration for a database shard."""

    shard_id: int
    redis_url: str
    postgres_url: Optional[str] = None


class ShardedStateManager:
    """State manager that distributes data across multiple shards."""

    def __init__(
        self,
        shards: list[ShardConfig],
        state_ttl: int = 86400,
        max_retries: int = 3,
        retry_backoff: float = 0.05,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        max_connections: int = 50,
    ) -> None:
        if not shards:
            raise ValueError("ShardedStateManager requires at least one shard")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")

        self.shards = {s.shard_id: s for s in shards}
        if len(self.shards) != len(shards):
            raise ValueError("Shard IDs must be unique")
        self.shard_ids = sorted(self.shards)
        self.shard_count = len(self.shard_ids)
        self.state_ttl = state_ttl
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Initialize connections to each shard
        if redis is None:
            raise ImportError("redis is required for ShardedStateManager")
        self.redis_clients: dict[int, Any] = {}
        for shard in shards:
            # Same production timeouts/pool as StateManager. With
            # potentially dozens of shards, the default unbounded
            # client would multiply hang risk across the cluster.
            self.redis_clients[shard.shard_id] = redis.from_url(
                shard.redis_url,
                decode_responses=True,
                socket_timeout=socket_timeout,
                socket_connect_timeout=socket_connect_timeout,
                health_check_interval=health_check_interval,
                max_connections=max_connections,
            )

    def _get_shard_id(self, key: str) -> int:
        """
        Determine which shard a key belongs to.

        This uses simple modulo hashing for clarity. For production systems
        where shards may be added/removed, use consistent hashing
        (hash ring with virtual nodes) to minimize data migration.
        Libraries: uhashring, hash_ring, or ketama.
        """
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        shard_index = hash_value % self.shard_count
        return self.shard_ids[shard_index]

    def _get_client(self, conversation_id: str) -> Any:
        """Get the Redis client for a conversation."""
        shard_id = self._get_shard_id(conversation_id)
        return self.redis_clients[shard_id]

    def _redis_call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run shard Redis I/O with bounded retry for transient failures."""
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (RedisError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError("Shard Redis operation failed after retries") from last_error

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Retrieve conversation state from the appropriate shard."""
        client = self._get_client(conversation_id)
        key = f"conversation:{conversation_id}"

        data = self._redis_call(client.get, key)
        if data is None:
            return None

        return ConversationState.from_json(data)

    def save_state(self, state: ConversationState) -> None:
        """Save conversation state to the appropriate shard."""
        client = self._get_client(state.conversation_id)
        key = f"conversation:{state.conversation_id}"

        state.updated_at = time.time()
        if state.created_at == 0.0:
            state.created_at = state.updated_at

        self._redis_call(client.setex, key, self.state_ttl, state.to_json())

    def get_shard_stats(self) -> dict[int, dict[str, Any]]:
        """Get statistics from each shard."""
        stats: dict[int, dict[str, Any]] = {}
        for shard_id, client in self.redis_clients.items():
            info = self._redis_call(client.info)
            stats[shard_id] = {
                "used_memory": info["used_memory_human"],
                "connected_clients": info["connected_clients"],
                "keys": self._redis_call(client.dbsize),
            }
        return stats

# ============================================================================
# Block 15 (chapter listing #15)
# ============================================================================

from contextlib import contextmanager
from typing import Callable, Generator, Generic, Optional, TypeVar
import threading
import time
from queue import Queue, Empty, Full


_ConnectionT = TypeVar("_ConnectionT")


class ConnectionPool(Generic[_ConnectionT]):
    """Generic connection pool for database connections."""

    def __init__(
        self,
        connection_factory: Callable[[], _ConnectionT],
        min_connections: int = 5,
        max_connections: int = 20,
        connection_timeout: float = 30.0,
        connection_error_types: tuple[type[BaseException], ...] = (
            ConnectionError,
            OSError,
            RuntimeError,
        ),
    ) -> None:
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if min_connections < 0:
            raise ValueError("min_connections must be >= 0")
        if min_connections > max_connections:
            raise ValueError("min_connections must be <= max_connections")
        if connection_timeout <= 0:
            raise ValueError("connection_timeout must be > 0")
        self.connection_factory = connection_factory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.connection_error_types = connection_error_types

        self.pool: Queue = Queue(maxsize=max_connections)
        self.size = 0
        self.lock = threading.Lock()

        # Initialize minimum connections
        for _ in range(min_connections):
            conn = self._create_connection()
            if conn:
                self.pool.put(conn)

    def _create_connection(self) -> Optional[_ConnectionT]:
        """Create a new connection."""
        with self.lock:
            if self.size >= self.max_connections:
                return None
            # Reserve the slot before the factory call so the cap check
            # above remains exact under concurrent callers; the
            # try/except below releases the slot if the factory raises
            # so a flaky backend cannot leak the pool's size accounting.
            self.size += 1
        try:
            return self.connection_factory()
        except BaseException:
            with self.lock:
                self.size -= 1
            raise

    @contextmanager
    def get_connection(self) -> Generator[_ConnectionT, None, None]:
        """Get a connection from the pool."""
        conn: Optional[_ConnectionT] = None

        try:
            # Try to get existing connection
            conn = self.pool.get(timeout=self.connection_timeout)
        except Empty:
            # Pool exhausted, try to create new connection
            conn = self._create_connection()
            if conn is None:
                raise RuntimeError("Connection pool exhausted")

        broken = False
        try:
            yield conn
        except self.connection_error_types:
            # Caller hit an error using this connection; assume it is
            # in an indeterminate state. Close and replace instead of
            # returning to the pool.
            broken = True
            raise
        finally:
            if conn is None:
                pass
            elif broken:
                try:
                    conn.close()
                finally:
                    with self.lock:
                        self.size -= 1
            else:
                # Caller exited cleanly; safe to recycle. Bounded
                # timeout matches the get path: an indefinitely
                # blocked put would mask backend regressions.
                try:
                    self.pool.put(conn, timeout=self.connection_timeout)
                except Full:
                    # Pool already at capacity (shouldn't happen with
                    # max_connections, but defensive): drop the connection.
                    try:
                        conn.close()
                    finally:
                        with self.lock:
                            self.size -= 1

    def close_all(self) -> None:
        """Close all connections in the pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except Empty:
                break

        with self.lock:
            self.size = 0

# ============================================================================
# Block 16 (chapter listing #16)
# ============================================================================

import os

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - optional example dependency
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class FastAPI:
        def on_event(self, event_name: str) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        def post(self, path: str) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

try:
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional example dependency
    class _AnthropicStub:
        class Anthropic:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError(
                    "anthropic package is required for the direct chat API"
                )

    anthropic = _AnthropicStub  # type: ignore[assignment,misc]

app = FastAPI()
client: Optional[anthropic.Anthropic] = None
state_manager: Optional[StateManager] = None
agent: Optional[StatelessAgent] = None
task_queue: Optional[RedisTaskQueue] = None
def _int_env(name: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` on bad values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        return default


def _float_env(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` on bad values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        return default


DIRECT_CHAT_MAX_WORKERS = _int_env("DIRECT_CHAT_MAX_WORKERS", 16)
DIRECT_CHAT_MAX_IN_FLIGHT = _int_env("DIRECT_CHAT_MAX_IN_FLIGHT", 32)
if DIRECT_CHAT_MAX_WORKERS < 1:
    raise ValueError("DIRECT_CHAT_MAX_WORKERS must be >= 1")
if DIRECT_CHAT_MAX_IN_FLIGHT < 1:
    raise ValueError("DIRECT_CHAT_MAX_IN_FLIGHT must be >= 1")
direct_executor = ThreadPoolExecutor(
    max_workers=DIRECT_CHAT_MAX_WORKERS,
    thread_name_prefix="direct-chat",
)
direct_executor_capacity = asyncio.Semaphore(
    DIRECT_CHAT_MAX_IN_FLIGHT
)
QUEUED_CHAT_RESULT_TIMEOUT_SECONDS = _float_env(
    "QUEUED_CHAT_RESULT_TIMEOUT_SECONDS", 60.0
)
if QUEUED_CHAT_RESULT_TIMEOUT_SECONDS <= 0:
    raise ValueError("QUEUED_CHAT_RESULT_TIMEOUT_SECONDS must be > 0")


DIRECT_CHAT_SHUTDOWN_GRACE_SECONDS = _float_env(
    "DIRECT_CHAT_SHUTDOWN_GRACE_SECONDS", 30.0
)
if DIRECT_CHAT_SHUTDOWN_GRACE_SECONDS <= 0:
    raise ValueError("DIRECT_CHAT_SHUTDOWN_GRACE_SECONDS must be > 0")


@app.on_event("shutdown")
def shutdown_direct_executor() -> None:
    """Drain in-flight direct-chat work, bounded by a grace period.

    ``ThreadPoolExecutor.shutdown`` has no native timeout, so the
    drain happens in a helper thread that we join with a deadline.
    If anything is still pending after the grace period we log a
    warning so ops can see the loss instead of silently abandoning
    work as the previous ``cancel_futures=True`` path did.
    """
    drain = threading.Thread(
        target=lambda: direct_executor.shutdown(
            wait=True, cancel_futures=False
        ),
        name="direct-chat-drain",
        daemon=True,
    )
    drain.start()
    drain.join(timeout=DIRECT_CHAT_SHUTDOWN_GRACE_SECONDS)
    if drain.is_alive():
        logging.getLogger(__name__).warning(
            "direct_executor still draining after %.1fs grace; "
            "some in-flight chat work may not have completed",
            DIRECT_CHAT_SHUTDOWN_GRACE_SECONDS,
        )


def get_redis_url() -> str:
    """Read Redis location from deployment configuration."""
    return os.getenv("REDIS_URL", "redis://localhost:6379")


def get_agent() -> StatelessAgent:
    """Lazily construct clients so imports do not perform network setup."""
    global client, state_manager, agent
    if agent is None:
        client = anthropic.Anthropic()
        state_manager = StateManager(get_redis_url())
        agent = StatelessAgent(state_manager, client)
    return agent


def get_task_queue() -> RedisTaskQueue:
    """Lazily construct the queue client outside module import."""
    global task_queue
    if task_queue is None:
        task_queue = RedisTaskQueue(get_redis_url())
    return task_queue


@app.post("/chat/direct")
async def chat_direct(conversation_id: str, message: str) -> dict[str, str]:
    # ``agent.process_message`` is synchronous (it issues blocking
    # Redis + Anthropic SDK calls). Isolate it in an explicit bounded
    # executor instead of the event loop's shared default pool, and
    # fail fast when the direct path is saturated.
    try:
        await asyncio.wait_for(direct_executor_capacity.acquire(), 0.05)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503, detail="direct chat capacity exhausted"
        ) from exc
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            direct_executor,
            get_agent().process_message,
            conversation_id,
            message,
        )
    finally:
        direct_executor_capacity.release()
    return {"response": response}

# ============================================================================
# Block 17 (chapter listing #17)
# ============================================================================

async def wait_for_result(
    task_id: str, timeout: float = 60.0, poll_interval: float = 0.5
) -> Optional[str]:
    """Poll Redis for a completed task result with bounded waiting.

    Returns the task's ``result`` once it lands or None on timeout.
    In production prefer pub/sub or webhooks (one round-trip), but
    polling is the simplest pattern that demonstrates the contract.
    """
    deadline = time.time() + timeout
    queue = get_task_queue()
    while time.time() < deadline:
        raw = await asyncio.to_thread(
            queue._redis_call, queue.redis.get, f"task:{task_id}"
        )
        if raw:
            task = AgentTask.from_json(raw)
            if task.status == TaskStatus.COMPLETED:
                return task.result
            if task.status == TaskStatus.FAILED:
                return None
        await asyncio.sleep(poll_interval)
    return None


@app.post("/chat/queued")
async def chat_queued(
    conversation_id: str, message: str
) -> dict[str, Optional[str]]:
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        message=message,
        created_at=time.time(),
    )

    get_task_queue().enqueue(task)

    # Poll for result (or use websockets/webhooks)
    result = await wait_for_result(
        task.task_id, timeout=QUEUED_CHAT_RESULT_TIMEOUT_SECONDS
    )
    return {"response": result}

# ============================================================================
# Block 18 (chapter listing #18)
# ============================================================================

# Configuration for 2000 req/min target
config = AutoScalerConfig(
    min_workers=6,
    max_workers=30,
    scale_up_queue_threshold=100,
    scale_up_latency_threshold=5000.0,
    scale_down_queue_threshold=20,
    evaluation_interval=15.0,
)

# ============================================================================
# Block 19 (chapter listing #19)
# ============================================================================

class PriorityTaskQueue:
    """
    Task queue with priority levels.

    Dequeue uses a Redis Lua script so the "find highest-priority
    non-empty queue and pop one task" operation is atomic across
    concurrent workers.
    """

    def __init__(
        self,
        redis_url: str,
        task_ttl_seconds: int = 86400,
        max_retries: int = 3,
        retry_backoff: float = 0.05,
        max_queue_depth: int = 100_000,
        visibility_timeout: int = 300,
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 2.0,
        health_check_interval: int = 30,
        max_connections: int = 50,
    ) -> None:
        if task_ttl_seconds < 1:
            raise ValueError("task_ttl_seconds must be >= 1")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")
        if visibility_timeout < 1:
            raise ValueError("visibility_timeout must be >= 1")
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if redis is None:
            raise ImportError("redis is required for PriorityTaskQueue")
        # Same production timeouts as RedisTaskQueue; a flaky Redis
        # would otherwise hang priority dequeue indefinitely.
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            health_check_interval=health_check_interval,
            max_connections=max_connections,
        )
        self.queues = {
            "critical": "tasks:critical",
            "high": "tasks:high",
            "normal": "tasks:normal",
            "low": "tasks:low",
        }
        self.processing_queue = "tasks:priority:processing"
        self.dead_letter_queue = "tasks:priority:dlq"
        self.visibility_index = "tasks:priority:visibility"
        self.task_ttl_seconds = task_ttl_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_queue_depth = max_queue_depth
        self.visibility_timeout = visibility_timeout
        self._priority_order = ["critical", "high", "normal", "low"]
        self._enqueue_script = """
        if redis.call('LLEN', KEYS[1]) >= tonumber(ARGV[1]) then
            return 0
        end
        redis.call('SETEX', KEYS[2], tonumber(ARGV[2]), ARGV[3])
        redis.call('LPUSH', KEYS[1], ARGV[4])
        return 1
        """
        self._dequeue_script = """
        local processing = KEYS[#KEYS - 1]
        local visibility = KEYS[#KEYS]
        for i = 1, #KEYS - 2 do
            while true do
                local task_id = redis.call('RPOP', KEYS[i])
                if not task_id then
                    break
                end

                local task_data = redis.call('GET', ARGV[1] .. task_id)
                if task_data then
                    redis.call('LPUSH', processing, task_id)
                    redis.call('ZADD', visibility, tonumber(ARGV[2]), task_id)
                    return task_data
                end
            end
        end
        return nil
        """
        self._complete_script = """
        redis.call('SETEX', KEYS[1], tonumber(ARGV[2]), ARGV[3])
        redis.call('LREM', KEYS[2], 1, ARGV[1])
        redis.call('ZREM', KEYS[3], ARGV[1])
        return 1
        """
        self._fail_script = """
        redis.call('SETEX', KEYS[1], tonumber(ARGV[3]), ARGV[4])
        redis.call('LREM', KEYS[2], 1, ARGV[1])
        redis.call('ZREM', KEYS[5], ARGV[1])
        if ARGV[2] == 'retry' then
            redis.call('LPUSH', KEYS[3], ARGV[1])
        else
            redis.call('LPUSH', KEYS[4], ARGV[1])
        end
        return 1
        """
        self._recover_script = """
        local removed = redis.call('LREM', KEYS[1], 1, ARGV[1])
        redis.call('ZREM', KEYS[3], ARGV[1])
        if removed == 0 then
            return 0
        end
        redis.call('LPUSH', KEYS[2], ARGV[1])
        return 1
        """

    def _redis_call(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        last_error: Optional[BaseException] = None
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except (RedisError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt == self.max_retries - 1:
                    break
                time.sleep(self.retry_backoff * (2**attempt))
        raise RuntimeError("Priority queue Redis operation failed after retries") from last_error

    def enqueue(
        self, task: AgentTask, priority: str = "normal"
    ) -> None:
        queue_name = self.queues.get(
            priority, self.queues["normal"]
        )
        task_key = f"task:{task.task_id}"

        inserted = self._redis_call(
            self.redis.eval,
            self._enqueue_script,
            2,
            queue_name,
            task_key,
            self.max_queue_depth,
            self.task_ttl_seconds,
            task.to_json(),
            task.task_id,
        )
        if not inserted:
            raise RuntimeError(f"Priority queue {priority} is full")

    def dequeue(self, timeout: int = 5) -> Optional[AgentTask]:
        queues = [self.queues[p] for p in self._priority_order]
        wait_deadline = time.monotonic() + max(timeout, 0)

        while True:
            visibility_deadline = time.time() + self.visibility_timeout
            data = self._redis_call(
                self.redis.eval,
                self._dequeue_script,
                len(queues) + 2,
                *queues,
                self.processing_queue,
                self.visibility_index,
                "task:",
                visibility_deadline,
            )
            if data:
                task = AgentTask.from_json(data)
                task.status = TaskStatus.PROCESSING
                task.attempt += 1
                self._redis_call(
                    self.redis.setex,
                    f"task:{task.task_id}",
                    self.task_ttl_seconds,
                    task.to_json(),
                )
                return task

            if timeout <= 0 or time.monotonic() >= wait_deadline:
                return None

            remaining = wait_deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.1, remaining))

    def complete(self, task: AgentTask, result: str) -> None:
        task.status = TaskStatus.COMPLETED
        task.result = result
        self._redis_call(
            self.redis.eval,
            self._complete_script,
            3,
            f"task:{task.task_id}",
            self.processing_queue,
            self.visibility_index,
            task.task_id,
            self.task_ttl_seconds,
            task.to_json(),
        )

    def fail(
        self, task: AgentTask, error: str, priority: str = "normal"
    ) -> None:
        task.error = error
        retry = task.attempt < min(task.max_attempts, self.max_retries)
        task.status = TaskStatus.RETRYING if retry else TaskStatus.FAILED
        target_queue = self.queues.get(priority, self.queues["normal"])
        self._redis_call(
            self.redis.eval,
            self._fail_script,
            5,
            f"task:{task.task_id}",
            self.processing_queue,
            target_queue,
            self.dead_letter_queue,
            self.visibility_index,
            task.task_id,
            "retry" if retry else "dlq",
            self.task_ttl_seconds,
            task.to_json(),
        )

    def recover_stale_tasks(self, batch_size: int = 100) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        task_ids = self._redis_call(
            self.redis.zrangebyscore,
            self.visibility_index,
            "-inf",
            time.time(),
            start=0,
            num=batch_size,
        )
        recovered = 0
        for task_id in task_ids:
            if self._redis_call(
                self.redis.eval,
                self._recover_script,
                3,
                self.processing_queue,
                self.queues["normal"],
                self.visibility_index,
                task_id,
            ):
                recovered += 1
        return recovered
