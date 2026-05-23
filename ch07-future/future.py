"""
Looking Ahead: The Future of Agentic AI

Code listings from Chapter 07, Book 3:
"Scaling Autonomous AI Agents: Engineering for Production at Real-World Scale"
by Dr. Vijay Raghavan

This file faithfully reproduces every code listing from the chapter, in book
order, with section banners showing the block number. Most listings are
runnable Python that builds incrementally; some are illustrative fragments
(log output, file trees, Dockerfile snippets, JSON examples) preserved as
docstrings so this file always remains valid Python.

To use a particular class or function, copy it into your own project and
provide the surrounding context (imports, dependencies) as needed.

Chapter 7 is forward-looking and intentionally light on concrete code. The
sketch below illustrates the multi-modal agent loop the chapter discusses
(see Section "Multi-Modal Agents"). It is a stub: the LLM call, vision
encoder, and audio encoder are abstract dependencies you supply.
"""

from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Callable, Optional, Protocol
import asyncio
import concurrent.futures
import inspect
import logging
import threading
import time


__all__ = ["MultiModalInput", "MultiModalAgent"]


@dataclass
class MultiModalInput:
    """A single multi-modal turn the agent receives.

    All fields are optional so the same dataclass works for text-only,
    voice-only, image+text, or fully multimodal interactions.
    """

    text: Optional[str] = None
    image_bytes: Optional[bytes] = None
    audio_bytes: Optional[bytes] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _EncoderProtocol(Protocol):
    def encode(self, payload: bytes) -> list[float]:  # pragma: no cover
        ...


class _CompleteClient(Protocol):
    def complete(self, **kwargs: Any) -> Any:  # pragma: no cover
        ...


class MultiModalAgent:
    """Skeleton multi-modal agent illustrating Chapter 7 patterns.

    A production implementation would wire in real encoders (Whisper for
    audio, a vision encoder for images), apply the policy/guard layers
    from Book 2, and emit observability events as covered in earlier
    chapters of this book. We deliberately keep dependencies abstract
    here so the example compiles in isolation.
    """

    # Size limits to prevent attacker-supplied payloads from OOMing
    # the encoder. Real deployments should tighten these against
    # observed traffic profiles.
    MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB
    MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MiB
    MAX_TEXT_CHARS = 20_000

    @staticmethod
    def _validate_retryable_errors(
        name: str, errors: tuple[type[BaseException], ...]
    ) -> tuple[type[BaseException], ...]:
        if any(
            err in (Exception, BaseException)
            or not isinstance(err, type)
            or not issubclass(err, BaseException)
            for err in errors
        ):
            raise ValueError(
                f"{name} must contain specific exception classes only"
            )
        return tuple(errors)

    def __init__(
        self,
        llm_client: _CompleteClient | Callable[[str], Any],
        vision_encoder: Optional[_EncoderProtocol] = None,
        audio_encoder: Optional[_EncoderProtocol] = None,
        timeout_seconds: float = 10.0,
        max_encoder_workers: int = 2,
        max_image_bytes: int = MAX_IMAGE_BYTES,
        max_audio_bytes: int = MAX_AUDIO_BYTES,
        max_text_chars: int = MAX_TEXT_CHARS,
        max_llm_retries: int = 3,
        llm_retry_backoff: float = 0.25,
        max_encoder_retries: int = 2,
        encoder_retry_backoff: float = 0.10,
        retryable_llm_errors: tuple[type[BaseException], ...] = (),
        retryable_encoder_errors: tuple[type[BaseException], ...] = (),
        max_encoder_pool_rotations: int = 2,
        max_llm_pool_rotations: int = 2,
        llm_circuit_breaker_failures: int = 3,
        llm_circuit_open_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if max_llm_retries < 1:
            raise ValueError("max_llm_retries must be >= 1")
        if max_encoder_retries < 1:
            raise ValueError("max_encoder_retries must be >= 1")
        if max_encoder_workers < 1:
            raise ValueError("max_encoder_workers must be >= 1")
        if max_image_bytes < 1:
            raise ValueError("max_image_bytes must be >= 1")
        if max_audio_bytes < 1:
            raise ValueError("max_audio_bytes must be >= 1")
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be >= 1")
        if llm_retry_backoff < 0:
            raise ValueError("llm_retry_backoff must be >= 0")
        if encoder_retry_backoff < 0:
            raise ValueError("encoder_retry_backoff must be >= 0")
        if max_encoder_pool_rotations < 0:
            raise ValueError("max_encoder_pool_rotations must be >= 0")
        if max_llm_pool_rotations < 0:
            raise ValueError("max_llm_pool_rotations must be >= 0")
        if llm_circuit_breaker_failures < 1:
            raise ValueError("llm_circuit_breaker_failures must be >= 1")
        if llm_circuit_open_seconds <= 0:
            raise ValueError("llm_circuit_open_seconds must be > 0")
        retryable_llm_errors = self._validate_retryable_errors(
            "retryable_llm_errors", retryable_llm_errors
        )
        retryable_encoder_errors = self._validate_retryable_errors(
            "retryable_encoder_errors", retryable_encoder_errors
        )
        # Fail fast on a misconfigured client at construction time rather
        # than after acquiring locks inside _submit_sync_llm_call.
        if not (hasattr(llm_client, "complete") or callable(llm_client)):
            raise TypeError(
                "llm_client must expose complete() or be callable"
            )
        self.llm_client = llm_client
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.timeout_seconds = timeout_seconds
        self.max_encoder_workers = max_encoder_workers
        self.max_image_bytes = max_image_bytes
        self.max_audio_bytes = max_audio_bytes
        self.max_text_chars = max_text_chars
        self.max_llm_retries = max_llm_retries
        self.llm_retry_backoff = llm_retry_backoff
        self.max_encoder_retries = max_encoder_retries
        self.encoder_retry_backoff = encoder_retry_backoff
        self.max_encoder_pool_rotations = max_encoder_pool_rotations
        self.max_llm_pool_rotations = max_llm_pool_rotations
        self.llm_circuit_breaker_failures = llm_circuit_breaker_failures
        self.llm_circuit_open_seconds = llm_circuit_open_seconds
        self._encoder_pool_rotations = 0
        self._llm_pool_rotations = 0
        self._abandoned_encoder_pools = 0
        self._abandoned_llm_pools = 0
        self._llm_failures = 0
        self._llm_circuit_open_until = 0.0
        self._closed = False
        # Set to True once we have exhausted pool rotations; subsequent
        # public-entry calls then raise RuntimeError so a supervisor
        # process can detect the degraded agent and recycle it.
        self._degraded = False
        provider_retryable: tuple[type[BaseException], ...] = ()
        try:
            from openai import (
                APIConnectionError,
                APIError,
                APITimeoutError,
                RateLimitError,
            )
        except ImportError:  # pragma: no cover
            pass
        else:
            provider_retryable += (
                APIConnectionError,
                APIError,
                APITimeoutError,
                RateLimitError,
            )
        try:
            import anthropic
        except ImportError:  # pragma: no cover
            pass
        else:
            provider_retryable += (
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
            )
        self.retryable_llm_errors = (
            TimeoutError,
            ConnectionError,
            OSError,
            *provider_retryable,
            *retryable_llm_errors,
        )
        self.retryable_encoder_errors = (
            TimeoutError,
            ConnectionError,
            OSError,
            *retryable_encoder_errors,
        )
        self.logger = logging.getLogger(__name__)
        self._lifecycle_lock = threading.RLock()
        self._encoder_pool_lock = threading.Lock()
        self._llm_pool_lock = threading.Lock()
        self._llm_circuit_lock = threading.Lock()
        self._encoder_semaphore = threading.BoundedSemaphore(
            max_encoder_workers
        )
        self._encoder_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_encoder_workers,
            thread_name_prefix="multimodal-encoder",
        )
        self._llm_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="multimodal-llm",
        )

    def close(self, wait: bool = True) -> None:
        """Release encoder and LLM worker threads.

        Default is graceful shutdown; pass ``wait=False`` only from a
        supervisor that can tolerate abandoned in-flight SDK calls.
        Once closed, subsequent calls to
        ``perceive``/``respond``/``respond_async`` raise ``RuntimeError``.
        Note: rotating a stuck thread pool is best-effort. Python has no
        kill primitive, so a stuck worker continues to hold its GIL slice
        and memory until it returns; production deployments that need
        hard-kill semantics should run encoders as OS-level subprocesses.
        """
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            with self._encoder_pool_lock:
                self._encoder_pool.shutdown(wait=wait, cancel_futures=True)
            with self._llm_pool_lock:
                self._llm_pool.shutdown(wait=wait, cancel_futures=True)

    def _check_open(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MultiModalAgent is closed")
            if self._degraded:
                raise RuntimeError(
                    "agent degraded \u2014 exceeded max pool rotations"
                )

    def _check_llm_circuit(self) -> None:
        with self._llm_circuit_lock:
            if time.monotonic() < self._llm_circuit_open_until:
                raise RuntimeError("LLM circuit breaker is open")

    def _record_llm_success(self) -> None:
        with self._llm_circuit_lock:
            self._llm_failures = 0
            self._llm_circuit_open_until = 0.0

    def _record_llm_failure(self) -> None:
        with self._llm_circuit_lock:
            self._llm_failures += 1
            if self._llm_failures >= self.llm_circuit_breaker_failures:
                self._llm_circuit_open_until = (
                    time.monotonic() + self.llm_circuit_open_seconds
                )

    def runtime_metrics(self) -> dict[str, Any]:
        """Expose pool-rotation and circuit-breaker counters for monitoring."""
        with self._encoder_pool_lock:
            encoder_pool_rotations = self._encoder_pool_rotations
            abandoned_encoder_pools = self._abandoned_encoder_pools
        with self._llm_pool_lock:
            llm_pool_rotations = self._llm_pool_rotations
            abandoned_llm_pools = self._abandoned_llm_pools
        with self._llm_circuit_lock:
            llm_failures = self._llm_failures
            circuit_open_for_seconds = max(
                0.0, self._llm_circuit_open_until - time.monotonic()
            )
        return {
            "encoder_pool_rotations": encoder_pool_rotations,
            "llm_pool_rotations": llm_pool_rotations,
            "abandoned_encoder_pools": abandoned_encoder_pools,
            "abandoned_llm_pools": abandoned_llm_pools,
            "llm_failures": llm_failures,
            "llm_circuit_open": circuit_open_for_seconds > 0,
            "llm_circuit_open_for_seconds": circuit_open_for_seconds,
        }

    def __enter__(self) -> "MultiModalAgent":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _encode_with_timeout(
        self, encoder: _EncoderProtocol, payload: bytes, modality: str
    ) -> list[float]:
        start = time.monotonic()
        if not self._encoder_semaphore.acquire(timeout=self.timeout_seconds):
            raise TimeoutError(f"{modality} encoder workers are saturated")
        last_error: BaseException | None = None
        try:
            for attempt in range(self.max_encoder_retries):
                with self._lifecycle_lock:
                    self._check_open()
                    with self._encoder_pool_lock:
                        future = self._encoder_pool.submit(
                            encoder.encode, payload
                        )
                try:
                    return future.result(timeout=self.timeout_seconds)
                except concurrent.futures.TimeoutError as exc:
                    last_error = exc
                    if not future.cancel() and not self._replace_encoder_pool():
                        self.logger.error(
                            "multimodal encoder rotation cap reached",
                            extra={"modality": modality, "bytes": len(payload)},
                        )
                    self.logger.warning(
                        "multimodal encoder timed out",
                        extra={"modality": modality, "bytes": len(payload)},
                    )
                    if attempt == self.max_encoder_retries - 1:
                        break
                    time.sleep(self.encoder_retry_backoff * (2**attempt))
                except self.retryable_encoder_errors as exc:
                    last_error = exc
                    if attempt == self.max_encoder_retries - 1:
                        break
                    self.logger.warning(
                        "multimodal encoder retryable failure",
                        extra={
                            "modality": modality,
                            "bytes": len(payload),
                            "attempt": attempt + 1,
                        },
                    )
                    time.sleep(self.encoder_retry_backoff * (2**attempt))
            self.logger.error(
                "multimodal encoder failed",
                extra={"modality": modality, "bytes": len(payload)},
            )
            if isinstance(last_error, concurrent.futures.TimeoutError):
                raise TimeoutError(
                    f"{modality} encoder timed out"
                ) from last_error
            raise RuntimeError(f"{modality} encoder failed") from last_error
        finally:
            self._encoder_semaphore.release()
            self.logger.info(
                "multimodal encoder completed",
                extra={
                    "modality": modality,
                    "bytes": len(payload),
                    "latency_ms": (time.monotonic() - start) * 1000,
                },
            )

    def _replace_encoder_pool(self) -> bool:
        """Rotate the pool after a stuck encoder exceeds the timeout."""
        with self._lifecycle_lock:
            if self._closed:
                return False
            with self._encoder_pool_lock:
                if self._encoder_pool_rotations >= self.max_encoder_pool_rotations:
                    # Exhausted our budget for rotating around a stuck
                    # encoder pool; mark the agent degraded so the next
                    # public-entry call raises rather than silently
                    # incrementing a counter.
                    self._degraded = True
                    return False
                old_pool = self._encoder_pool
                self._encoder_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.max_encoder_workers,
                    thread_name_prefix="multimodal-encoder",
                )
                self._encoder_pool_rotations += 1
                self._abandoned_encoder_pools += 1
        old_pool.shutdown(wait=False, cancel_futures=True)
        self.logger.warning(
            "rotated encoder pool; abandoned_pools=%d",
            self._abandoned_encoder_pools,
            extra=self.runtime_metrics(),
        )
        return True

    def _replace_llm_pool(self) -> bool:
        """Rotate the single-worker LLM pool after a stuck call times out."""
        with self._lifecycle_lock:
            if self._closed:
                return False
            with self._llm_pool_lock:
                if self._llm_pool_rotations >= self.max_llm_pool_rotations:
                    # Exhausted our budget for rotating around a stuck
                    # LLM pool; mark the agent degraded so the next
                    # public-entry call raises rather than silently
                    # incrementing a counter.
                    self._degraded = True
                    return False
                old_pool = self._llm_pool
                self._llm_pool = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="multimodal-llm",
                )
                self._llm_pool_rotations += 1
                self._abandoned_llm_pools += 1
        old_pool.shutdown(wait=False, cancel_futures=True)
        self.logger.warning(
            "rotated LLM pool; abandoned_pools=%d",
            self._abandoned_llm_pools,
            extra=self.runtime_metrics(),
        )
        return True

    def perceive(self, turn: MultiModalInput) -> dict[str, Any]:
        """Convert raw modalities into model-ready representations."""
        self._check_open()
        perception: dict[str, Any] = {}
        if turn.text:
            if len(turn.text) > self.max_text_chars:
                raise ValueError(
                    f"text exceeds max_text_chars "
                    f"({len(turn.text)} > {self.max_text_chars})"
                )
            perception["text"] = turn.text
        if turn.image_bytes and self.vision_encoder is not None:
            if len(turn.image_bytes) > self.max_image_bytes:
                raise ValueError(
                    f"image_bytes exceeds max_image_bytes "
                    f"({len(turn.image_bytes)} > {self.max_image_bytes})"
                )
            perception["image_embedding"] = self._encode_with_timeout(
                self.vision_encoder, turn.image_bytes, "image"
            )
        if turn.audio_bytes and self.audio_encoder is not None:
            if len(turn.audio_bytes) > self.max_audio_bytes:
                raise ValueError(
                    f"audio_bytes exceeds max_audio_bytes "
                    f"({len(turn.audio_bytes)} > {self.max_audio_bytes})"
                )
            perception["audio_embedding"] = self._encode_with_timeout(
                self.audio_encoder, turn.audio_bytes, "audio"
            )
        return perception

    async def _encode_with_timeout_async(
        self, encoder: _EncoderProtocol, payload: bytes, modality: str
    ) -> list[float]:
        return await asyncio.to_thread(
            self._encode_with_timeout, encoder, payload, modality
        )

    async def perceive_async(self, turn: MultiModalInput) -> dict[str, Any]:
        """Async perception path that keeps encoder waits off the event loop."""
        self._check_open()
        perception: dict[str, Any] = {}
        if turn.text:
            if len(turn.text) > self.max_text_chars:
                raise ValueError(
                    f"text exceeds max_text_chars "
                    f"({len(turn.text)} > {self.max_text_chars})"
                )
            perception["text"] = turn.text
        if turn.image_bytes and self.vision_encoder is not None:
            if len(turn.image_bytes) > self.max_image_bytes:
                raise ValueError(
                    f"image_bytes exceeds max_image_bytes "
                    f"({len(turn.image_bytes)} > {self.max_image_bytes})"
                )
            perception["image_embedding"] = await self._encode_with_timeout_async(
                self.vision_encoder, turn.image_bytes, "image"
            )
        if turn.audio_bytes and self.audio_encoder is not None:
            if len(turn.audio_bytes) > self.max_audio_bytes:
                raise ValueError(
                    f"audio_bytes exceeds max_audio_bytes "
                    f"({len(turn.audio_bytes)} > {self.max_audio_bytes})"
                )
            perception["audio_embedding"] = await self._encode_with_timeout_async(
                self.audio_encoder, turn.audio_bytes, "audio"
            )
        return perception

    def _build_prompt(self, perception: dict[str, Any]) -> str:
        lines = [
            "Respond to this multimodal user turn.",
            f"Available fields: {sorted(perception.keys())}",
            f"Text: {perception.get('text', '')}",
        ]
        for key in ("image_embedding", "audio_embedding"):
            if key in perception:
                embedding = perception[key]
                preview = ", ".join(f"{float(value):.4f}" for value in embedding[:8])
                lines.append(
                    f"{key}: dim={len(embedding)} preview=[{preview}]"
                )
        return "\n".join(lines)

    @staticmethod
    def _response_text(response: Any) -> str:
        if inspect.isawaitable(response):
            raise TypeError("async LLM response returned; use respond_async()")
        return getattr(response, "text", str(response))

    def _submit_sync_llm_call(
        self, prompt: str
    ) -> concurrent.futures.Future[Any]:
        # __init__ already rejected a client that lacks both complete()
        # and __call__, so we can branch unconditionally inside the
        # locked region without a redundant TypeError fallthrough.
        with self._lifecycle_lock:
            self._check_open()
            self._check_llm_circuit()
            with self._llm_pool_lock:
                pool = self._llm_pool
                if hasattr(self.llm_client, "complete"):
                    return pool.submit(
                        self.llm_client.complete,
                        prompt=prompt,
                        timeout=self.timeout_seconds,
                    )
                return pool.submit(self.llm_client, prompt)

    async def _call_llm_async(self, prompt: str) -> str:
        self._check_llm_circuit()
        if hasattr(self.llm_client, "complete"):
            complete = self.llm_client.complete
            if inspect.iscoroutinefunction(complete):
                result = complete(prompt=prompt, timeout=self.timeout_seconds)
            else:
                result = await self._run_sync_llm_call_async(
                    complete, prompt=prompt, timeout=self.timeout_seconds
                )
        elif callable(self.llm_client):
            if inspect.iscoroutinefunction(self.llm_client):
                result = self.llm_client(prompt)
            else:
                result = await self._run_sync_llm_call_async(
                    self.llm_client, prompt
                )
        else:
            raise TypeError("llm_client must expose complete() or be callable")

        if inspect.isawaitable(result):
            try:
                result = await asyncio.wait_for(
                    result, timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError("LLM call timed out") from exc
        self._record_llm_success()
        return getattr(result, "text", str(result))

    async def _run_sync_llm_call_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        loop = asyncio.get_running_loop()
        with self._lifecycle_lock:
            self._check_open()
            self._check_llm_circuit()
            with self._llm_pool_lock:
                pool = self._llm_pool
            submitted = loop.run_in_executor(
                pool, lambda: func(*args, **kwargs)
            )
        try:
            return await asyncio.wait_for(
                submitted,
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            if not self._replace_llm_pool():
                self.logger.error(
                    "multimodal LLM rotation cap reached",
                    extra=self.runtime_metrics(),
                )
            raise TimeoutError("LLM call timed out") from exc

    def respond(self, turn: MultiModalInput) -> str:
        """End-to-end perceive -> reason -> respond loop for sync clients."""
        self._check_open()
        perception = self.perceive(turn)
        prompt = self._build_prompt(perception)
        start = time.monotonic()
        try:
            for attempt in range(self.max_llm_retries):
                try:
                    future = self._submit_sync_llm_call(prompt)
                    try:
                        text = self._response_text(
                            future.result(timeout=self.timeout_seconds)
                        )
                        self._record_llm_success()
                        return text
                    except concurrent.futures.TimeoutError as exc:
                        if not future.cancel():
                            if not self._replace_llm_pool():
                                self.logger.error(
                                    "multimodal LLM rotation cap reached",
                                    extra=self.runtime_metrics(),
                                )
                        raise TimeoutError("LLM call timed out") from exc
                except self.retryable_llm_errors as exc:
                    self._record_llm_failure()
                    if attempt == self.max_llm_retries - 1:
                        raise
                    self.logger.warning("multimodal LLM call retrying: %s", exc)
                    time.sleep(self.llm_retry_backoff * (2**attempt))
            raise RuntimeError("multimodal LLM call exhausted retries")
        finally:
            self.logger.info(
                "multimodal response completed",
                extra={
                    "modalities": sorted(perception.keys()),
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "correlation_id": turn.metadata.get("correlation_id"),
                    "request_id": turn.metadata.get("request_id"),
                },
            )

    async def respond_async(self, turn: MultiModalInput) -> str:
        """Async equivalent of respond() for coroutine-based LLM clients."""
        self._check_open()
        perception = await self.perceive_async(turn)
        prompt = self._build_prompt(perception)
        start = time.monotonic()
        try:
            for attempt in range(self.max_llm_retries):
                try:
                    return await self._call_llm_async(prompt)
                except self.retryable_llm_errors as exc:
                    self._record_llm_failure()
                    if attempt == self.max_llm_retries - 1:
                        raise
                    self.logger.warning(
                        "multimodal async LLM call retrying: %s", exc
                    )
                    await asyncio.sleep(self.llm_retry_backoff * (2**attempt))
            raise RuntimeError("multimodal async LLM call exhausted retries")
        finally:
            self.logger.info(
                "multimodal async response completed",
                extra={
                    "modalities": sorted(perception.keys()),
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "correlation_id": turn.metadata.get("correlation_id"),
                    "request_id": turn.metadata.get("request_id"),
                },
            )
