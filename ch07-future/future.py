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
from typing import Any, Optional, Protocol
import concurrent.futures
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


class MultiModalAgent:
    """Skeleton multi-modal agent illustrating Chapter 7 patterns.

    A production implementation would wire in real encoders (Whisper for
    audio, a vision encoder for images), apply the policy/guard layers
    from Book 2, and emit observability events as covered in earlier
    chapters of this book. We deliberately keep dependencies abstract
    here so the example compiles in isolation.
    """

    def __init__(
        self,
        llm_client: Any,
        vision_encoder: Optional[_EncoderProtocol] = None,
        audio_encoder: Optional[_EncoderProtocol] = None,
        timeout_seconds: float = 10.0,
        max_encoder_workers: int = 2,
        max_image_bytes: int = 20 * 1024 * 1024,
        max_audio_bytes: int = 50 * 1024 * 1024,
        max_llm_retries: int = 3,
        llm_retry_backoff: float = 0.25,
        retryable_llm_errors: tuple[type[BaseException], ...] = (),
    ) -> None:
        if max_llm_retries < 1:
            raise ValueError("max_llm_retries must be >= 1")
        if max_encoder_workers < 1:
            raise ValueError("max_encoder_workers must be >= 1")
        self.llm_client = llm_client
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.timeout_seconds = timeout_seconds
        self.max_encoder_workers = max_encoder_workers
        self.max_image_bytes = max_image_bytes
        self.max_audio_bytes = max_audio_bytes
        self.max_llm_retries = max_llm_retries
        self.llm_retry_backoff = llm_retry_backoff
        self.retryable_llm_errors = (
            TimeoutError,
            ConnectionError,
            OSError,
            *retryable_llm_errors,
        )
        self.logger = logging.getLogger(__name__)
        self._encoder_pool_lock = threading.Lock()
        self._llm_pool_lock = threading.Lock()
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

    def close(self) -> None:
        """Release encoder worker threads."""
        with self._encoder_pool_lock:
            self._encoder_pool.shutdown(wait=False, cancel_futures=True)
        with self._llm_pool_lock:
            self._llm_pool.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "MultiModalAgent":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # Size limits to prevent attacker-supplied payloads from OOMing
    # the encoder. Real deployments should tighten these against
    # observed traffic profiles.
    MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB
    MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MiB

    def _encode_with_timeout(
        self, encoder: _EncoderProtocol, payload: bytes, modality: str
    ) -> list[float]:
        start = time.monotonic()
        if not self._encoder_semaphore.acquire(timeout=self.timeout_seconds):
            raise TimeoutError(f"{modality} encoder workers are saturated")
        try:
            with self._encoder_pool_lock:
                future = self._encoder_pool.submit(encoder.encode, payload)
            return future.result(timeout=self.timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            self._replace_encoder_pool()
            self.logger.warning(
                "multimodal encoder timed out",
                extra={"modality": modality, "bytes": len(payload)},
            )
            raise TimeoutError(f"{modality} encoder timed out") from exc
        except (RuntimeError, ValueError, OSError) as exc:
            self.logger.exception(
                "multimodal encoder failed",
                extra={"modality": modality, "bytes": len(payload)},
            )
            raise RuntimeError(f"{modality} encoder failed") from exc
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

    def _replace_encoder_pool(self) -> None:
        """Rotate the pool after a stuck encoder exceeds the timeout."""
        with self._encoder_pool_lock:
            old_pool = self._encoder_pool
            self._encoder_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_encoder_workers,
                thread_name_prefix="multimodal-encoder",
            )
        old_pool.shutdown(wait=False, cancel_futures=True)

    def _replace_llm_pool(self) -> None:
        """Rotate the single-worker LLM pool after a stuck call times out."""
        with self._llm_pool_lock:
            old_pool = self._llm_pool
            self._llm_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="multimodal-llm",
            )
        old_pool.shutdown(wait=False, cancel_futures=True)

    def perceive(self, turn: MultiModalInput) -> dict[str, Any]:
        """Convert raw modalities into model-ready representations."""
        perception: dict[str, Any] = {}
        if turn.text:
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

    def respond(self, turn: MultiModalInput) -> str:
        """End-to-end perceive -> reason -> respond loop."""
        perception = self.perceive(turn)
        prompt = (
            "Respond to this multimodal user turn. Available fields:\n"
            f"{sorted(perception.keys())}\n\n"
            f"Text: {perception.get('text', '')}"
        )
        start = time.monotonic()
        try:
            for attempt in range(self.max_llm_retries):
                try:
                    if hasattr(self.llm_client, "complete"):
                        with self._llm_pool_lock:
                            future = self._llm_pool.submit(
                                self.llm_client.complete,
                                prompt=prompt,
                                timeout=self.timeout_seconds,
                            )
                        try:
                            response = future.result(
                                timeout=self.timeout_seconds
                            )
                            return getattr(response, "text", str(response))
                        except concurrent.futures.TimeoutError as exc:
                            future.cancel()
                            self._replace_llm_pool()
                            raise TimeoutError("LLM complete timed out") from exc
                    if callable(self.llm_client):
                        with self._llm_pool_lock:
                            future = self._llm_pool.submit(
                                self.llm_client, prompt
                            )
                        try:
                            return str(
                                future.result(timeout=self.timeout_seconds)
                            )
                        except concurrent.futures.TimeoutError as exc:
                            future.cancel()
                            self._replace_llm_pool()
                            raise TimeoutError("LLM callable timed out") from exc
                    raise TypeError("llm_client must expose complete() or be callable")
                except self.retryable_llm_errors as exc:
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
                },
            )
