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
from typing import Any, Optional, Protocol


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
    ) -> None:
        self.llm_client = llm_client
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder

    # Size limits to prevent attacker-supplied payloads from OOMing
    # the encoder. Real deployments should tighten these against
    # observed traffic profiles.
    MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MiB
    MAX_AUDIO_BYTES = 50 * 1024 * 1024  # 50 MiB

    def perceive(self, turn: MultiModalInput) -> dict[str, Any]:
        """Convert raw modalities into model-ready representations."""
        perception: dict[str, Any] = {}
        if turn.text:
            perception["text"] = turn.text
        if turn.image_bytes and self.vision_encoder is not None:
            if len(turn.image_bytes) > self.MAX_IMAGE_BYTES:
                raise ValueError(
                    f"image_bytes exceeds MAX_IMAGE_BYTES "
                    f"({len(turn.image_bytes)} > {self.MAX_IMAGE_BYTES})"
                )
            perception["image_embedding"] = self.vision_encoder.encode(
                turn.image_bytes
            )
        if turn.audio_bytes and self.audio_encoder is not None:
            if len(turn.audio_bytes) > self.MAX_AUDIO_BYTES:
                raise ValueError(
                    f"audio_bytes exceeds MAX_AUDIO_BYTES "
                    f"({len(turn.audio_bytes)} > {self.MAX_AUDIO_BYTES})"
                )
            perception["audio_embedding"] = self.audio_encoder.encode(
                turn.audio_bytes
            )
        return perception

    def respond(self, turn: MultiModalInput) -> str:
        """End-to-end perceive -> reason -> respond loop (sketch)."""
        _ = self.perceive(turn)
        # Production: assemble a prompt from the perception dict, call
        # ``self.llm_client``, optionally invoke tools, and return text.
        raise NotImplementedError(
            "MultiModalAgent.respond: chapter-level sketch only. "
            "See Books 1 and 2 for the orchestration and guard patterns."
        )
