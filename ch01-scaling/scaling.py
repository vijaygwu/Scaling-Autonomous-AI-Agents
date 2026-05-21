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


# ============================================================================
# Block 1 (chapter listing #1)
# ============================================================================

# What went wrong: stateful agent design
class CustomerAgent:
    def __init__(self):
        self.conversation_history = {}  # In-memory state

    def handle_request(self, user_id, message):
        # State lost when this instance dies or scales
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        # ...

# ============================================================================
# Block 2 (chapter listing #2)
# ============================================================================

# Anti-pattern: Stateful agent design
class StatefulAgent:
    def __init__(self):
        self.conversations: dict[str, list[dict]] = {}

    def process_message(self, conversation_id: str, message: str) -> str:
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []

        self.conversations[conversation_id].append(
            {"role": "user", "content": message}
        )

        response = self._call_llm(self.conversations[conversation_id])

        self.conversations[conversation_id].append(
            {"role": "assistant", "content": response}
        )

        return response

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

import json
import redis
import time
from dataclasses import dataclass, field
from typing import Optional
import hashlib


@dataclass
class ConversationState:
    """Represents externalized conversation state."""

    conversation_id: str
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
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


class StateManager:
    """Manages externalized agent state in Redis."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        # Explicit timeouts + health-check + bounded pool. The default
        # ``redis.from_url(...)`` produces a client that blocks
        # indefinitely on a hung server; in a worker pool that
        # serializes every state op behind the slowest TCP timeout.
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=2,
            health_check_interval=30,
            max_connections=50,
        )
        self.state_ttl = 86400  # 24 hours default

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Retrieve conversation state from Redis."""
        key = f"conversation:{conversation_id}"
        data = self.redis.get(key)

        if data is None:
            return None

        return ConversationState.from_json(data)

    def save_state(self, state: ConversationState) -> None:
        """Persist conversation state to Redis."""
        state.updated_at = time.time()
        if state.created_at == 0.0:
            state.created_at = state.updated_at

        key = f"conversation:{state.conversation_id}"
        self.redis.setex(key, self.state_ttl, state.to_json())

    def delete_state(self, conversation_id: str) -> None:
        """Remove conversation state."""
        key = f"conversation:{conversation_id}"
        self.redis.delete(key)

    def extend_ttl(self, conversation_id: str) -> None:
        """Extend the TTL for an active conversation."""
        key = f"conversation:{conversation_id}"
        self.redis.expire(key, self.state_ttl)


class StatelessAgent:
    """Agent designed for horizontal scaling with externalized state."""

    def __init__(self, state_manager: StateManager, llm_client):
        self.state_manager = state_manager
        self.llm_client = llm_client

    def process_message(self, conversation_id: str, message: str) -> str:
        # Load state from external storage
        state = self.state_manager.get_state(conversation_id)

        if state is None:
            state = ConversationState(conversation_id=conversation_id)

        # Add the new message
        state.messages.append({"role": "user", "content": message})

        # Generate response
        response = self._call_llm(state.messages)

        # Update state
        state.messages.append({"role": "assistant", "content": response})

        # Persist state back to external storage
        self.state_manager.save_state(state)

        return response

    def _call_llm(self, messages: list[dict]) -> str:
        # Anthropic-SDK clients accept ``timeout`` and ``max_retries``
        # at construction time; surface them here for readers porting
        # this code so a stalled provider does not block the worker
        # indefinitely. Production callers should additionally wrap
        # this in the @with_retry/@with_timeout decorators introduced
        # in ch05 (customer service) to cap end-to-end latency budgets.
        response = self.llm_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=messages,
            timeout=30.0,  # seconds; aligns with default budget
        )
        return response.content[0].text

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

class VersionedStateManager:
    """State manager with optimistic locking for conflict detection."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        # Mirror StateManager's production timeouts/pool; a flaky Redis
        # would otherwise wedge every version check behind the default
        # unbounded socket timeout.
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=2,
            health_check_interval=30,
            max_connections=50,
        )

    def get_state_with_version(
        self, conversation_id: str
    ) -> tuple[Optional[ConversationState], int]:
        """Retrieve state along with its version number."""
        key = f"conversation:{conversation_id}"
        version_key = f"conversation:{conversation_id}:version"

        pipe = self.redis.pipeline()
        pipe.get(key)
        pipe.get(version_key)
        data, version = pipe.execute()

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
        redis.call('EXPIRE', KEYS[1], 86400)
        redis.call('EXPIRE', KEYS[2], 86400)
        return 1
        """

        result = self.redis.eval(
            lua_script, 2, key, version_key, state.to_json(), expected_version
        )

        return result == 1

# ============================================================================
# Block 5 (chapter block #5) — Python fragment (incomplete, depends on surrounding context)
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

    def __init__(self):
        self.workers: dict[str, WorkerInfo] = {}
        self.lock = threading.Lock()

    def register_worker(
        self, worker_id: str, address: str, weight: float = 1.0
    ) -> None:
        """Add a worker to the pool."""
        with self.lock:
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

    def __init__(self):
        self.workers: dict[str, WorkerInfo] = {}
        self.history_window = 100  # Keep last 100 latencies
        # ``deque(maxlen=...)`` makes the bound automatic and avoids an
        # O(n) slice copy on every overflow append.
        self.latency_history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.history_window)
        )
        # RLock (reentrant) because ``select_worker`` holds the lock and
        # then calls ``get_average_latency``, which also takes it. A
        # plain ``Lock`` would deadlock on that second acquire.
        self.lock = threading.RLock()

    def record_latency(self, worker_id: str, latency_ms: float) -> None:
        """Record request latency for a worker."""
        with self.lock:
            self.latency_history[worker_id].append(latency_ms)

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
                ) / worker.weight

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
import redis
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
    """Production-ready Redis-based task queue for agent systems."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        queue_name: str = "agent_tasks",
        visibility_timeout: int = 300,  # 5 minutes
        body_ttl_seconds: int = 7 * 24 * 3600,  # 7 days
    ):
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=2,
            health_check_interval=30,
            max_connections=50,
        )
        self.queue_name = queue_name
        self.processing_queue = f"{queue_name}:processing"
        self.dead_letter_queue = f"{queue_name}:dlq"
        self.visibility_timeout = visibility_timeout
        # Body TTL must comfortably exceed the worst-case queue
        # residence (queue depth × visibility timeout + retries).
        # A 24-hour TTL on the body but multi-day queue residence
        # causes the id to outlive its body and the dequeue path
        # returns None silently (work is lost). 7 days is a safer
        # default; callers with a different SLA should override it.
        self.body_ttl_seconds = body_ttl_seconds
        self.logger = logging.getLogger(__name__)

    def enqueue(self, task: AgentTask) -> str:
        """Add a task to the queue."""
        task.created_at = time.time()

        # Store task details
        task_key = f"task:{task.task_id}"
        self.redis.setex(task_key, self.body_ttl_seconds, task.to_json())

        # Add to queue
        self.redis.lpush(self.queue_name, task.task_id)

        self.logger.info(f"Enqueued task {task.task_id}")
        return task.task_id

    def dequeue(self, timeout: int = 0) -> Optional[AgentTask]:
        """
        Retrieve a task from the queue.
        Uses BLMOVE for reliable processing (Redis 6.2+).
        """
        # Move task from main queue to processing queue atomically
        task_id = self.redis.blmove(
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
        task_data = self.redis.get(task_key)

        if task_data is None:
            # Task expired, remove from processing queue
            self.redis.lrem(self.processing_queue, 1, task_id)
            return None

        task = AgentTask.from_json(task_data)
        task.status = TaskStatus.PROCESSING
        task.attempt += 1

        # Update task status
        self.redis.setex(task_key, self.body_ttl_seconds, task.to_json())

        # Set visibility timeout
        self._set_visibility_timeout(task_id)

        return task

    def complete(self, task: AgentTask, result: str) -> None:
        """Mark a task as successfully completed."""
        task.status = TaskStatus.COMPLETED
        task.result = result

        task_key = f"task:{task.task_id}"
        self.redis.setex(task_key, self.body_ttl_seconds, task.to_json())

        # Remove from processing queue
        self.redis.lrem(self.processing_queue, 1, task.task_id)
        self._clear_visibility_timeout(task.task_id)

        self.logger.info(f"Completed task {task.task_id}")

    def fail(self, task: AgentTask, error: str) -> None:
        """Mark a task as failed, potentially requeueing for retry."""
        task.error = error

        if task.attempt < task.max_attempts:
            # Requeue for retry
            task.status = TaskStatus.RETRYING
            task_key = f"task:{task.task_id}"
            self.redis.setex(task_key, self.body_ttl_seconds, task.to_json())

            # Remove from processing and re-enqueue. The submit path uses
            # lpush + a right-side dequeue (BLMOVE from RIGHT), giving FIFO
            # order with the oldest task at the right end. Match that
            # convention here so retries are treated like fresh arrivals
            # rather than jumping the queue.
            self.redis.lrem(self.processing_queue, 1, task.task_id)
            self.redis.lpush(self.queue_name, task.task_id)

            self.logger.warning(
                f"Task {task.task_id} failed, requeueing "
                f"(attempt {task.attempt}/{task.max_attempts})"
            )
        else:
            # Move to dead letter queue
            task.status = TaskStatus.FAILED
            task_key = f"task:{task.task_id}"
            self.redis.setex(task_key, self.body_ttl_seconds, task.to_json())

            self.redis.lrem(self.processing_queue, 1, task.task_id)
            self.redis.lpush(self.dead_letter_queue, task.task_id)

            self.logger.error(
                f"Task {task.task_id} permanently failed after "
                f"{task.max_attempts} attempts"
            )

        self._clear_visibility_timeout(task.task_id)

    def get_queue_depth(self) -> dict[str, int]:
        """Get the current depth of all queues."""
        return {
            "pending": self.redis.llen(self.queue_name),
            "processing": self.redis.llen(self.processing_queue),
            "dead_letter": self.redis.llen(self.dead_letter_queue),
        }

    def _set_visibility_timeout(self, task_id: str) -> None:
        """Set a timeout for task processing."""
        timeout_key = f"task:{task_id}:timeout"
        self.redis.setex(
            timeout_key, self.visibility_timeout, str(time.time())
        )

    def _clear_visibility_timeout(self, task_id: str) -> None:
        """Clear the visibility timeout."""
        timeout_key = f"task:{task_id}:timeout"
        self.redis.delete(timeout_key)

    def recover_stale_tasks(self) -> int:
        """
        Recover tasks that have exceeded visibility timeout.
        Should be called periodically by a maintenance process.
        """
        recovered = 0
        processing_tasks = self.redis.lrange(self.processing_queue, 0, -1)

        for task_id in processing_tasks:
            timeout_key = f"task:{task_id}:timeout"
            if not self.redis.exists(timeout_key):
                # Timeout expired, requeue the task
                self.redis.lrem(self.processing_queue, 1, task_id)
                self.redis.lpush(self.queue_name, task_id)
                recovered += 1
                self.logger.warning(f"Recovered stale task {task_id}")

        return recovered

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

import pika
import json
from typing import Callable, Optional
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


class RabbitMQTaskQueue:
    """RabbitMQ-based task queue with advanced features."""

    def __init__(
        self,
        config: RabbitMQConfig,
        exchange_name: str = "agent_exchange",
        queue_name: str = "agent_tasks",
    ):
        self.config = config
        self.exchange_name = exchange_name
        self.queue_name = queue_name
        self.dead_letter_exchange = f"{exchange_name}_dlx"
        self.dead_letter_queue = f"{queue_name}_dlq"
        self.logger = logging.getLogger(__name__)

        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None

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
                "x-message-ttl": 3600000,  # 1 hour TTL
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
        )

        self.logger.info(f"Published task {task.task_id}")

    def consume(
        self, callback: Callable[[AgentTask], bool], prefetch_count: int = 1
    ) -> None:
        """
        Start consuming tasks from the queue.
        Callback should return True on success, False on failure.
        """
        self.channel.basic_qos(prefetch_count=prefetch_count)

        def on_message(channel, method, properties, body):
            task = AgentTask.from_json(body.decode("utf-8"))

            try:
                success = callback(task)

                if success:
                    channel.basic_ack(delivery_tag=method.delivery_tag)
                else:
                    # Requeue with delay by rejecting
                    channel.basic_nack(
                        delivery_tag=method.delivery_tag, requeue=True
                    )
            except Exception as e:
                self.logger.error(f"Error processing task: {e}")
                # Don't requeue on exception, send to DLQ
                channel.basic_nack(
                    delivery_tag=method.delivery_tag, requeue=False
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

import boto3
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
    ):
        self.sqs = boto3.client("sqs", region_name=region_name)
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

    def get_queue_attributes(self) -> dict:
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


class AgentWorker:
    """Individual worker that processes agent tasks."""

    def __init__(
        self,
        worker_id: str,
        task_queue: RedisTaskQueue,
        agent: StatelessAgent,
        heartbeat_interval: int = 30,
    ):
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.agent = agent
        self.heartbeat_interval = heartbeat_interval
        self.stats = WorkerStats(worker_id=worker_id)
        self.logger = logging.getLogger(f"worker.{worker_id}")
        self._stop_event = threading.Event()

    def run(self) -> None:
        """Main worker loop."""
        self.logger.info(f"Worker {self.worker_id} starting")

        while not self._stop_event.is_set():
            try:
                # Dequeue with timeout to allow checking stop event
                task = self.task_queue.dequeue(timeout=5)

                if task is None:
                    self._update_heartbeat()
                    continue

                self._process_task(task)

            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
                time.sleep(1)  # Back off on errors

        self.stats.state = WorkerState.STOPPED
        self.logger.info(f"Worker {self.worker_id} stopped")

    def _process_task(self, task: AgentTask) -> None:
        """Process a single task."""
        self.stats.state = WorkerState.BUSY
        self.stats.current_task_start = time.time()

        self.logger.info(f"Processing task {task.task_id}")

        try:
            result = self.agent.process_message(
                task.conversation_id, task.message
            )

            self.task_queue.complete(task, result)

            processing_time = time.time() - self.stats.current_task_start
            self.stats.tasks_completed += 1
            self.stats.total_processing_time += processing_time

            self.logger.info(
                f"Completed task {task.task_id} in {processing_time:.2f}s"
            )

        except Exception as e:
            self.logger.error(f"Task {task.task_id} failed: {e}")
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
    ):
        self.task_queue = task_queue
        self.agent_factory = agent_factory
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_idle_timeout = worker_idle_timeout

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
                worker_id=worker_id, task_queue=self.task_queue, agent=agent
            )

            thread = threading.Thread(
                target=worker.run, name=worker_id, daemon=True
            )

            self.workers[worker_id] = worker
            self.threads[worker_id] = thread

            thread.start()

            self.logger.info(f"Added worker {worker_id}")
            return worker_id

    def _remove_worker(self, worker_id: str) -> None:
        """Remove a worker from the pool."""
        with self.lock:
            if worker_id not in self.workers:
                return

            if len(self.workers) <= self.min_workers:
                return

            worker = self.workers[worker_id]
            worker.stop()

            # Wait for worker to finish current task
            thread = self.threads[worker_id]
            thread.join(timeout=60)

            del self.workers[worker_id]
            del self.threads[worker_id]

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
        """Periodic maintenance: health checks, cleanup."""
        while not self._shutdown_event.is_set():
            try:
                self._check_worker_health()
                self._recover_stale_tasks()
            except Exception as e:
                self.logger.error(f"Maintenance error: {e}")

            self._shutdown_event.wait(timeout=30)

    def _check_worker_health(self) -> None:
        """Check worker health and restart unhealthy workers."""
        now = time.time()

        with self.lock:
            for worker_id, worker in list(self.workers.items()):
                # Check if worker is alive
                thread = self.threads.get(worker_id)
                if thread and not thread.is_alive():
                    self.logger.warning(f"Worker {worker_id} died, removing")
                    del self.workers[worker_id]
                    del self.threads[worker_id]
                    continue

                # Check heartbeat
                if now - worker.stats.last_heartbeat > 120:
                    self.logger.warning(f"Worker {worker_id} heartbeat stale")

    def _recover_stale_tasks(self) -> None:
        """Recover tasks that have timed out."""
        recovered = self.task_queue.recover_stale_tasks()
        if recovered > 0:
            self.logger.info(f"Recovered {recovered} stale tasks")

    def get_stats(self) -> dict:
        """Get pool statistics."""
        with self.lock:
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
            "worker_count": len(self.workers),
            "workers": worker_stats,
            "queue": queue_depth,
        }

    def shutdown(self, timeout: int = 60) -> None:
        """Gracefully shutdown the worker pool."""
        self.logger.info("Shutting down worker pool")
        self._shutdown_event.set()

        # Stop all workers
        for worker in self.workers.values():
            worker.stop()

        # Wait for threads to finish
        for thread in self.threads.values():
            thread.join(timeout=timeout)

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

    def __init__(self, config: BackpressureConfig):
        self.config = config
        self.current_rate = config.max_rate / 2  # Start at midpoint
        self.latencies: deque = deque(maxlen=config.window_size)
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
    """Queue consumer with adaptive backpressure control."""

    def __init__(
        self,
        task_queue: RedisTaskQueue,
        worker_pool: WorkerPool,
        backpressure_config: Optional[BackpressureConfig] = None,
    ):
        self.task_queue = task_queue
        self.worker_pool = worker_pool
        self.rate_limiter = AdaptiveRateLimiter(
            backpressure_config or BackpressureConfig()
        )
        self.logger = logging.getLogger("consumer")
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start consuming tasks with backpressure control."""
        self.logger.info("Starting backpressure-controlled consumer")

        while not self._stop_event.is_set():
            # Check queue depth
            queue_depth = self.task_queue.get_queue_depth()

            if queue_depth["pending"] == 0:
                # No tasks, wait before checking again
                self._stop_event.wait(timeout=1.0)
                continue

            # Apply backpressure
            if not self.rate_limiter.acquire(timeout=5.0):
                self.logger.debug("Rate limited, waiting")
                continue

            # Dispatch task to worker pool
            task = self.task_queue.dequeue(timeout=1)
            if task:
                start_time = time.time()
                self._dispatch_task(task)
                latency_ms = (time.time() - start_time) * 1000
                self.rate_limiter.record_latency(latency_ms)

    def _dispatch_task(self, task: AgentTask) -> None:
        """Re-enqueue the rate-limited task for the worker pool.

        ``WorkerPool`` workers self-pull from ``task_queue``; this
        consumer's role is to apply backpressure (rate limiting) on
        top of that. We dequeued the task to check it through the
        limiter; now we put it back at the head so the next worker
        picks it up immediately.

        Compensate for the side effects of the dequeue itself:
        ``RedisTaskQueue.dequeue`` increments ``task.attempt`` and
        installs a visibility-timeout marker, both of which are
        accounting that should only happen on real processing
        attempts. Without rolling these back, every pass through the
        rate limiter burns one retry from the task's retry budget
        and could push the task into the dead-letter queue without
        a worker ever touching it.
        """
        q = self.task_queue
        # Roll back the attempt bump that dequeue applied (this was
        # a shaping pass, not a processing attempt) and persist.
        if task.attempt > 0:
            task.attempt -= 1
            q.redis.setex(
                f"task:{task.task_id}", q.body_ttl_seconds, task.to_json()
            )
        # Remove from processing list (one occurrence) and clear the
        # visibility timer.
        q.redis.lrem(q.processing_queue, 1, task.task_id)
        q._clear_visibility_timeout(task.task_id)
        # Re-insert at the head to preserve the rate-limited task's
        # priority over newer enqueues.
        q.redis.lpush(q.queue_name, task.task_id)
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

    # Cooldown periods
    scale_up_cooldown: float = 60.0  # seconds
    scale_down_cooldown: float = 300.0  # seconds

    # Evaluation interval
    evaluation_interval: float = 30.0  # seconds


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
    ):
        self.worker_pool = worker_pool
        self.task_queue = task_queue
        self.config = config or AutoScalerConfig()

        self.metrics_history: deque = deque(
            maxlen=60
        )  # 30 min at 30s intervals
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

            except Exception as e:
                self.logger.error(f"Auto-scaler error: {e}")

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
                5,  # Add at most 5 workers at once
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
                and metrics.idle_workers > 1
            ):

                if (
                    now - self.last_scale_down
                    < self.config.scale_down_cooldown
                ):
                    return 0

                # Scale down conservatively
                workers_to_remove = min(
                    metrics.idle_workers - 1,  # Keep at least 1 idle
                    metrics.worker_count - self.config.min_workers,
                    2,  # Remove at most 2 workers at once
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

        # Calculate workers needed to drain queue in 5 minutes
        # while handling incoming work
        incoming_rate = self._estimate_incoming_rate()
        total_work = metrics.queue_depth + (incoming_rate * 5)

        workers_needed = int(total_work / (tasks_per_minute * 5)) + 1

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
from typing import Optional
import redis
from dataclasses import dataclass


@dataclass
class ShardConfig:
    """Configuration for a database shard."""

    shard_id: int
    redis_url: str
    postgres_url: Optional[str] = None


class ShardedStateManager:
    """State manager that distributes data across multiple shards."""

    def __init__(self, shards: list[ShardConfig]):
        self.shards = {s.shard_id: s for s in shards}
        self.shard_count = len(shards)

        # Initialize connections to each shard
        self.redis_clients: dict[int, redis.Redis] = {}
        for shard in shards:
            # Same production timeouts/pool as StateManager. With
            # potentially dozens of shards, the default unbounded
            # client would multiply hang risk across the cluster.
            self.redis_clients[shard.shard_id] = redis.from_url(
                shard.redis_url,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=2,
                health_check_interval=30,
                max_connections=50,
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
        return hash_value % self.shard_count

    def _get_client(self, conversation_id: str) -> redis.Redis:
        """Get the Redis client for a conversation."""
        shard_id = self._get_shard_id(conversation_id)
        return self.redis_clients[shard_id]

    def get_state(self, conversation_id: str) -> Optional[ConversationState]:
        """Retrieve conversation state from the appropriate shard."""
        client = self._get_client(conversation_id)
        key = f"conversation:{conversation_id}"

        data = client.get(key)
        if data is None:
            return None

        return ConversationState.from_json(data)

    def save_state(self, state: ConversationState) -> None:
        """Save conversation state to the appropriate shard."""
        client = self._get_client(state.conversation_id)
        key = f"conversation:{state.conversation_id}"

        import time

        state.updated_at = time.time()
        if state.created_at == 0.0:
            state.created_at = state.updated_at

        client.setex(key, 86400, state.to_json())

    def get_shard_stats(self) -> dict[int, dict]:
        """Get statistics from each shard."""
        stats = {}
        for shard_id, client in self.redis_clients.items():
            info = client.info()
            stats[shard_id] = {
                "used_memory": info["used_memory_human"],
                "connected_clients": info["connected_clients"],
                "keys": client.dbsize(),
            }
        return stats

# ============================================================================
# Block 15 (chapter listing #15)
# ============================================================================

from contextlib import contextmanager
from typing import Generator
import threading
import time
from queue import Queue, Empty, Full


class ConnectionPool:
    """Generic connection pool for database connections."""

    def __init__(
        self,
        connection_factory,
        min_connections: int = 5,
        max_connections: int = 20,
        connection_timeout: float = 30.0,
    ):
        self.connection_factory = connection_factory
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout

        self.pool: Queue = Queue(maxsize=max_connections)
        self.size = 0
        self.lock = threading.Lock()

        # Initialize minimum connections
        for _ in range(min_connections):
            conn = self._create_connection()
            if conn:
                self.pool.put(conn)

    def _create_connection(self):
        """Create a new connection."""
        with self.lock:
            if self.size >= self.max_connections:
                return None

            conn = self.connection_factory()
            self.size += 1
            return conn

    @contextmanager
    def get_connection(self) -> Generator:
        """Get a connection from the pool."""
        conn = None

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
        except Exception:
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

from fastapi import FastAPI
import anthropic

app = FastAPI()
client = anthropic.Anthropic()
state_manager = StateManager("redis://localhost:6379")
agent = StatelessAgent(state_manager, client)
# Module-level handle shared by the queue-backed ``/chat`` endpoint and
# ``wait_for_result`` below. Defined here so those references resolve
# at import time; in production wire this through your app's lifespan
# handler rather than a module global.
task_queue = RedisTaskQueue("redis://localhost:6379")


@app.post("/chat")
async def chat(conversation_id: str, message: str):
    # ``agent.process_message`` is synchronous (it issues blocking
    # Redis + Anthropic SDK calls). Calling it directly from an async
    # endpoint would block the event loop and serialize every concurrent
    # request through a single thread. ``asyncio.to_thread`` hands it
    # off to the default ThreadPoolExecutor so the loop can serve other
    # requests while this one waits on I/O.
    response = await asyncio.to_thread(
        agent.process_message, conversation_id, message
    )
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
    while time.time() < deadline:
        raw = await asyncio.to_thread(task_queue.redis.get, f"task:{task_id}")
        if raw:
            task = AgentTask.from_json(raw)
            if task.status == TaskStatus.COMPLETED:
                return task.result
            if task.status == TaskStatus.FAILED:
                return None
        await asyncio.sleep(poll_interval)
    return None


@app.post("/chat")
async def chat(conversation_id: str, message: str):
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
        message=message,
        created_at=time.time(),
    )

    task_queue.enqueue(task)

    # Poll for result (or use websockets/webhooks)
    result = await wait_for_result(task.task_id, timeout=60)
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

    Note: the dequeue loop below checks multiple queues and is
    not atomic. Under high concurrency, tasks may be processed
    slightly out of priority order. For strict priority guarantees,
    use Redis Streams with consumer groups or a Lua script to pop
    atomically from the highest-priority non-empty queue.
    """

    def __init__(self, redis_url: str):
        # Same production timeouts as RedisTaskQueue; a flaky Redis
        # would otherwise hang priority dequeue indefinitely.
        self.redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=2,
            health_check_interval=30,
            max_connections=50,
        )
        self.queues = {
            "critical": "tasks:critical",
            "high": "tasks:high",
            "normal": "tasks:normal",
            "low": "tasks:low",
        }

    def enqueue(
        self, task: AgentTask, priority: str = "normal"
    ) -> None:
        queue_name = self.queues.get(
            priority, self.queues["normal"]
        )
        self.redis.lpush(queue_name, task.task_id)
        task_key = f"task:{task.task_id}"
        self.redis.setex(task_key, 86400, task.to_json())

    def dequeue(self, timeout: int = 5) -> Optional[AgentTask]:
        # Check queues in priority order
        for priority in ["critical", "high", "normal", "low"]:
            task_id = self.redis.rpop(self.queues[priority])
            if task_id:
                data = self.redis.get(f"task:{task_id}")
                if data:
                    return AgentTask.from_json(data)

        # Fall back to blocking pop on normal queue
        queues = list(self.queues.values())
        result = self.redis.brpop(queues, timeout=timeout)

        if result:
            _, task_id = result
            data = self.redis.get(f"task:{task_id}")
            if data:
                return AgentTask.from_json(data)

        return None
