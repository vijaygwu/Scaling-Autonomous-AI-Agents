"""
Complete Example: Customer Service Platform

Code listings from Chapter 05, Book 3:
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

# Core data models for the customer service platform.
# These models define the contract between components.

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Optional
import inspect
import textwrap
import uuid


class ConversationChannel(Enum):
    """Supported customer contact channels."""

    WEB_CHAT = "web_chat"
    MOBILE_APP = "mobile_app"
    VOICE = "voice"
    EMAIL = "email"
    SMS = "sms"


class ConversationStatus(Enum):
    """Conversation lifecycle states."""

    ACTIVE = "active"
    WAITING_CUSTOMER = "waiting_customer"
    WAITING_AGENT = "waiting_agent"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class Priority(Enum):
    """Customer priority levels."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class AgentType(Enum):
    """Types of specialized agents."""

    TRIAGE = "triage"
    ORDER = "order"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


@dataclass
class Customer:
    """Customer profile with relevant context."""

    customer_id: str
    email: str
    name: str
    phone: Optional[str] = None
    tier: str = "standard"  # standard, premium, enterprise
    lifetime_value: float = 0.0
    account_age_days: int = 0
    open_tickets: int = 0
    # Cap in-memory recent-order summary; full history lives in the
    # orders service. Callers should treat ``recent_orders`` as a
    # display window, not a complete audit log.
    recent_orders: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=20)
    )
    preferences: dict[str, Any] = field(default_factory=dict)

    @property
    def priority(self) -> Priority:
        """Calculate customer priority based on profile."""
        if self.tier == "enterprise":
            return Priority.CRITICAL
        elif self.tier == "premium" or self.lifetime_value > 10000:
            return Priority.HIGH
        elif self.open_tickets > 2:
            return Priority.HIGH
        return Priority.NORMAL


@dataclass
class Message:
    """A single message in a conversation."""

    message_id: str
    conversation_id: str
    role: str  # customer, agent, system
    content: str
    timestamp: datetime
    agent_type: Optional[AgentType] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        conversation_id: str,
        role: str,
        content: str,
        agent_type: Optional[AgentType] = None,
    ) -> "Message":
        return cls(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc),
            agent_type=agent_type,
        )


@dataclass
class ConversationContext:
    """Context accumulated during a conversation."""

    customer: Customer
    intent: Optional[str] = None
    intent_confidence: float = 0.0
    extracted_entities: dict[str, Any] = field(default_factory=dict)
    current_agent: Optional[AgentType] = None
    # Bound both lists so a long-running conversation cannot exhaust
    # memory. Per-conversation handoffs and tool calls rarely exceed
    # these caps; if you need full history, persist to the conversation
    # store rather than holding it in process.
    previous_agents: deque[AgentType] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    tool_results: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=50)
    )
    escalation_reason: Optional[str] = None
    sentiment_score: float = 0.0  # -1 to 1

    def add_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        """Record a tool execution result."""
        self.tool_results.append(
            {
                "tool": tool_name,
                "result": result,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


@dataclass
class Conversation:
    """Complete conversation state."""

    MAX_IN_MEMORY_MESSAGES = 200

    conversation_id: str
    channel: ConversationChannel
    status: ConversationStatus
    context: ConversationContext
    messages: deque[Message] = field(default_factory=deque)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
    quality_score: Optional[float] = None
    turn_count: int = 0

    def __post_init__(self) -> None:
        self.messages = deque(
            self.messages, maxlen=self.MAX_IN_MEMORY_MESSAGES
        )

    @classmethod
    def create(
        cls, customer: Customer, channel: ConversationChannel
    ) -> "Conversation":
        return cls(
            conversation_id=str(uuid.uuid4()),
            channel=channel,
            status=ConversationStatus.ACTIVE,
            context=ConversationContext(customer=customer),
        )

    def add_message(
        self, role: str, content: str, agent_type: Optional[AgentType] = None
    ) -> Message:
        """Add a message to the conversation."""
        message = Message.create(
            self.conversation_id, role, content, agent_type
        )
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)
        return message

    def get_history_for_llm(
        self, max_messages: int = 20
    ) -> list[dict[str, str]]:
        """Format conversation history for LLM context."""
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        recent = list(self.messages)[-max_messages:]
        return [{"role": m.role, "content": m.content} for m in recent]


@dataclass
class EscalationRequest:
    """Request to escalate to human agent."""

    escalation_id: str
    conversation_id: str
    reason: str
    priority: Priority
    required_skills: list[str]
    context_summary: str
    attempted_resolutions: list[str]
    customer_sentiment: float
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @classmethod
    def create(
        cls,
        conversation: Conversation,
        reason: str,
        required_skills: list[str],
    ) -> "EscalationRequest":
        return cls(
            escalation_id=str(uuid.uuid4()),
            conversation_id=conversation.conversation_id,
            reason=reason,
            priority=conversation.context.customer.priority,
            required_skills=required_skills,
            context_summary=cls._summarize_context(conversation),
            attempted_resolutions=[
                r["tool"] for r in conversation.context.tool_results
            ],
            customer_sentiment=conversation.context.sentiment_score,
        )

    @staticmethod
    def _summarize_context(conversation: Conversation) -> str:
        """Create a summary for human agents."""
        ctx = conversation.context
        return textwrap.dedent(f"""\
            Customer: {ctx.customer.name} ({ctx.customer.tier} tier)
            Intent: {ctx.intent}
            Sentiment: {"Negative" if ctx.sentiment_score < -0.3 else "Neutral" if ctx.sentiment_score < 0.3 else "Positive"}
            Previous agents: {", ".join(a.value for a in ctx.previous_agents)}
            Key entities: {ctx.extracted_entities}
            """).strip()

# ============================================================================
# Block 2 (chapter listing #2)
# ============================================================================

# Configuration management for the customer service platform.
# Uses environment-based configuration with sensible defaults.

from dataclasses import dataclass, field
from typing import Optional
import json
import os


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "anthropic"),
            model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT", "30")),
        )


@dataclass
class AgentConfig:
    """Configuration for individual agents."""

    agent_type: str
    enabled: bool = True
    max_turns: int = 10
    confidence_threshold: float = 0.7
    escalation_threshold: int = 3  # Failed attempts before escalation
    timeout_seconds: float = 30.0
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    max_transient_retries: int = 1
    llm_retry_backoff_seconds: float = 1.5
    tools: list[str] = field(default_factory=list)
    system_prompt_template: str = ""

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.escalation_threshold < 0:
            raise ValueError("escalation_threshold must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.max_transient_retries < 0:
            raise ValueError("max_transient_retries must be >= 0")
        if self.llm_retry_backoff_seconds < 0:
            raise ValueError("llm_retry_backoff_seconds must be >= 0")


@dataclass
class IntegrationConfig:
    """Configuration for external system integrations."""

    crm_base_url: str = ""
    crm_api_key: str = ""
    orders_base_url: str = ""
    orders_api_key: str = ""
    payments_base_url: str = ""
    payments_api_key: str = ""
    knowledge_base_url: str = ""
    knowledge_base_api_key: str = ""
    ticketing_base_url: str = ""
    ticketing_api_key: str = ""

    @classmethod
    def from_env(cls) -> "IntegrationConfig":
        return cls(
            crm_base_url=os.getenv("CRM_BASE_URL", ""),
            crm_api_key=os.getenv("CRM_API_KEY", ""),
            orders_base_url=os.getenv("ORDERS_BASE_URL", ""),
            orders_api_key=os.getenv("ORDERS_API_KEY", ""),
            payments_base_url=os.getenv("PAYMENTS_BASE_URL", ""),
            payments_api_key=os.getenv("PAYMENTS_API_KEY", ""),
            knowledge_base_url=os.getenv("KNOWLEDGE_BASE_URL", ""),
            knowledge_base_api_key=os.getenv("KNOWLEDGE_BASE_API_KEY", ""),
            ticketing_base_url=os.getenv("TICKETING_BASE_URL", ""),
            ticketing_api_key=os.getenv("TICKETING_API_KEY", ""),
        )


@dataclass
class HTTPClientConfig:
    """Shared HTTP client settings for platform integrations."""

    timeout_seconds: float = 10.0
    connect_timeout_seconds: float = 2.0
    read_timeout_seconds: float = 10.0
    write_timeout_seconds: float = 5.0
    max_connections: int = 50
    max_keepalive_connections: int = 20

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.connect_timeout_seconds <= 0:
            raise ValueError("connect_timeout_seconds must be > 0")
        if self.read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be > 0")
        if self.write_timeout_seconds <= 0:
            raise ValueError("write_timeout_seconds must be > 0")
        if self.max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        if self.max_keepalive_connections < 0:
            raise ValueError("max_keepalive_connections must be >= 0")
        if self.max_keepalive_connections > self.max_connections:
            raise ValueError(
                "max_keepalive_connections must be <= max_connections"
            )

    @classmethod
    def from_env(cls) -> "HTTPClientConfig":
        return cls(
            timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
            connect_timeout_seconds=float(
                os.getenv("HTTP_CONNECT_TIMEOUT_SECONDS", "2")
            ),
            read_timeout_seconds=float(os.getenv("HTTP_READ_TIMEOUT_SECONDS", "10")),
            write_timeout_seconds=float(os.getenv("HTTP_WRITE_TIMEOUT_SECONDS", "5")),
            max_connections=int(os.getenv("HTTP_MAX_CONNECTIONS", "50")),
            max_keepalive_connections=int(
                os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20")
            ),
        )


@dataclass
class QualityConfig:
    """Configuration for quality assurance."""

    min_satisfaction_score: float = 4.0
    max_handle_time_seconds: int = 480
    max_first_response_ms: int = 30_000
    min_automation_rate: float = 70.0
    max_escalation_rate: float = 15.0
    sample_rate_for_review: float = 0.1
    sentiment_alert_threshold: float = -0.5
    assessment_timeout_seconds: float = 30.0
    assessment_model: str = "claude-sonnet-4-20250514"
    assessment_max_tokens: int = 200

    @classmethod
    def from_env(cls) -> "QualityConfig":
        return cls(
            min_satisfaction_score=float(
                os.getenv("MIN_SATISFACTION", "4.0")
            ),
            max_handle_time_seconds=int(os.getenv("MAX_HANDLE_TIME", "480")),
            max_first_response_ms=int(
                os.getenv("MAX_FIRST_RESPONSE_MS", "30000")
            ),
            min_automation_rate=float(os.getenv("MIN_AUTOMATION_RATE", "70")),
            max_escalation_rate=float(os.getenv("MAX_ESCALATION_RATE", "15")),
            sample_rate_for_review=float(os.getenv("SAMPLE_RATE", "0.1")),
            sentiment_alert_threshold=float(
                os.getenv("SENTIMENT_ALERT", "-0.5")
            ),
            assessment_timeout_seconds=float(
                os.getenv("QA_ASSESSMENT_TIMEOUT", "30")
            ),
            assessment_model=os.getenv(
                "QA_ASSESSMENT_MODEL", "claude-sonnet-4-20250514"
            ),
            assessment_max_tokens=int(
                os.getenv("QA_ASSESSMENT_MAX_TOKENS", "200")
            ),
        )


@dataclass
class PlatformConfig:
    """Root configuration for the entire platform."""

    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    integrations: IntegrationConfig = field(
        default_factory=IntegrationConfig.from_env
    )
    http: HTTPClientConfig = field(default_factory=HTTPClientConfig.from_env)
    quality: QualityConfig = field(default_factory=QualityConfig.from_env)
    agents: dict[str, AgentConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "PlatformConfig":
        """Load configuration from environment and optional file."""
        config = cls()
        file_config = {}
        if config_path:
            with open(config_path, encoding="utf-8") as f:
                file_config = json.load(f)

            def merge(instance: Any, values: dict[str, Any]) -> Any:
                allowed = instance.__dataclass_fields__
                merged = {
                    **instance.__dict__,
                    **{
                        key: value
                        for key, value in values.items()
                        if key in allowed
                    },
                }
                return type(instance)(**merged)

            if "llm" in file_config:
                config.llm = merge(config.llm, file_config["llm"])
            if "integrations" in file_config:
                config.integrations = merge(
                    config.integrations, file_config["integrations"]
                )
            if "http" in file_config:
                config.http = merge(config.http, file_config["http"])
            if "quality" in file_config:
                config.quality = merge(
                    config.quality, file_config["quality"]
                )

        # Define default agent configurations
        config.agents = {
            "triage": AgentConfig(
                agent_type="triage",
                max_turns=3,
                confidence_threshold=0.8,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                tools=["identify_customer", "classify_intent"],
            ),
            "order": AgentConfig(
                agent_type="order",
                max_turns=10,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                tools=[
                    "get_order_status",
                    "get_order_details",
                    "modify_order",
                    "initiate_return",
                    "track_shipment",
                ],
            ),
            "technical": AgentConfig(
                agent_type="technical",
                max_turns=15,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                tools=[
                    "search_knowledge_base",
                    "run_diagnostic",
                    "get_product_info",
                    "create_ticket",
                ],
            ),
            "billing": AgentConfig(
                agent_type="billing",
                max_turns=10,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                tools=[
                    "get_account_balance",
                    "get_payment_history",
                    "process_refund",
                    "update_payment_method",
                ],
            ),
            "escalation": AgentConfig(
                agent_type="escalation",
                max_turns=5,
                model=config.llm.model,
                max_tokens=config.llm.max_tokens,
                tools=[
                    "create_escalation_ticket",
                    "find_available_agent",
                    "transfer_conversation",
                ],
            ),
        }

        for name, values in file_config.get("agents", {}).items():
            base = config.agents.get(name, AgentConfig(agent_type=name))
            allowed = base.__dataclass_fields__
            merged = {
                **base.__dict__,
                **{key: value for key, value in values.items() if key in allowed},
            }
            config.agents[name] = AgentConfig(**merged)

        return config

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

# Tool framework for customer service agents.
# Provides a consistent interface for all tool implementations.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, ParamSpec, TypeVar
import json
import logging
import functools
import asyncio
import httpx

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[list[str | int | float | bool]] = None


@dataclass
class ToolDefinition:
    """Complete tool definition for LLM function calling."""

    name: str
    description: str
    parameters: list[ToolParameter]
    requires_confirmation: bool = False
    audit_level: str = "standard"  # minimal, standard, detailed

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""

    success: bool
    data: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_llm_response(self) -> str:
        """Format result for LLM consumption."""
        if self.success:
            if isinstance(self.data, dict):
                return json.dumps(self.data, indent=2, default=str)
            return str(self.data)
        return f"Error: {self.error}"


@dataclass(frozen=True)
class ToolExecutionSettings:
    """Shared retry and timeout settings for HTTP/LLM-backed tools."""

    max_attempts: int = 3
    timeout_seconds: float = 10.0
    retry_backoff_seconds: float = 0.25

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")


ORDER_MUTATION_TOOL_SETTINGS = ToolExecutionSettings(timeout_seconds=15.0)
RETURN_TOOL_SETTINGS = ToolExecutionSettings(timeout_seconds=20.0)
TRACKING_TOOL_SETTINGS = ToolExecutionSettings(
    max_attempts=2, retry_backoff_seconds=1.0
)
BILLING_READ_TOOL_SETTINGS = ToolExecutionSettings()
REFUND_TOOL_SETTINGS = ToolExecutionSettings(
    timeout_seconds=30.0, retry_backoff_seconds=0.5
)
SUPPORT_SEARCH_TOOL_SETTINGS = ToolExecutionSettings()
DIAGNOSTIC_TOOL_SETTINGS = ToolExecutionSettings(timeout_seconds=60.0)
CRM_LOOKUP_TOOL_SETTINGS = ToolExecutionSettings()
INTENT_CLASSIFIER_TOOL_SETTINGS = ToolExecutionSettings()


class Tool(ABC):
    """Base class for all tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's definition."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def validate_params(self, **kwargs: Any) -> Optional[str]:
        """Validate parameters against definition. Returns error message if invalid."""
        type_checks = {
            "string": lambda value: isinstance(value, str),
            "integer": lambda value: isinstance(value, int)
            and not isinstance(value, bool),
            "number": lambda value: isinstance(value, (int, float))
            and not isinstance(value, bool),
            "boolean": lambda value: isinstance(value, bool),
            "array": lambda value: isinstance(value, list),
            "object": lambda value: isinstance(value, dict),
        }
        for param in self.definition.parameters:
            if param.required and param.name not in kwargs:
                return f"Missing required parameter: {param.name}"
            if param.name in kwargs:
                value = kwargs[param.name]
                check = type_checks.get(param.type)
                if check and not check(value):
                    return (
                        f"Invalid type for {param.name}. "
                        f"Expected {param.type}"
                    )
                if param.enum and value not in param.enum:
                    return f"Invalid value for {param.name}. Must be one of: {param.enum}"
        return None


def with_retry(
    max_attempts: int = 3, backoff_seconds: float = 1.0
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator for retrying failed tool executions.

    Only retries on transient failures (timeouts, connection errors,
    5xx upstream responses). Permanent errors (4xx HTTP statuses,
    authentication failures, validation errors) are raised
    immediately so the caller doesn't waste its retry budget on a
    request the server will never accept.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be >= 0")

    # Errors that are worth retrying. httpx raises ReadTimeout /
    # ConnectTimeout / ConnectError; non-2xx responses surface as
    # HTTPStatusError after raise_for_status().
    _retryable_classes: tuple = (
        asyncio.TimeoutError,
        TimeoutError,
        ConnectionError,
        OSError,
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )

    def _provider_retryable_classes() -> tuple[type[BaseException], ...]:
        provider = globals().get("anthropic")
        if provider is None:
            return ()
        classes: list[type[BaseException]] = []
        for name in (
            "APITimeoutError",
            "RateLimitError",
            "APIConnectionError",
        ):
            cls = getattr(provider, name, None)
            if isinstance(cls, type) and issubclass(cls, BaseException):
                classes.append(cls)
        return tuple(classes)

    # Resolve provider-specific retryable classes once at decorator
    # construction time; the SDK module cannot appear mid-process, so
    # there is no value in re-probing the global on every exception.
    _cached_provider_retryable: tuple[type[BaseException], ...] = (
        _provider_retryable_classes()
    )

    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, _retryable_classes + _cached_provider_retryable):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            # 429 is explicit backpressure. Retry it with backoff (and
            # Retry-After when present); other 4xx statuses are permanent
            # caller-side failures.
            return (
                exc.response.status_code == 429
                or 500 <= exc.response.status_code < 600
            )
        return False

    def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
        if isinstance(exc, httpx.HTTPStatusError):
            retry_after = exc.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    from email.utils import parsedate_to_datetime

                    try:
                        retry_at = parsedate_to_datetime(retry_after)
                    except (TypeError, ValueError, IndexError, OverflowError):
                        pass
                    else:
                        if retry_at.tzinfo is None:
                            retry_at = retry_at.replace(tzinfo=timezone.utc)
                        return max(
                            0.0,
                            (
                                retry_at
                                - datetime.now(timezone.utc)
                            ).total_seconds(),
                        )
        return backoff_seconds * (2**attempt)

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_error: Optional[BaseException] = None
            for attempt in range(max_attempts):
                handled_classes = (
                    _retryable_classes
                    + _cached_provider_retryable
                    + (httpx.HTTPStatusError,)
                )
                try:
                    return await func(*args, **kwargs)
                except handled_classes as e:
                    last_error = e
                    if not _is_retryable(e):
                        # Permanent: don't waste retries on it.
                        raise
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(_retry_delay_seconds(e, attempt))
                        logger.warning(
                            f"Retry {attempt + 1} for {func.__name__}: {e}"
                        )
            if last_error is not None:
                raise last_error
            raise RuntimeError("retry loop exited without an exception")

        return wrapper

    return decorator


def with_timeout(
    seconds: float,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator for adding a timeout to tool executions.

    Raises ``asyncio.TimeoutError`` on expiry so that an outer
    ``@with_retry`` can observe and retry. Callers who want a
    ToolResult-on-timeout instead of an exception should wrap the
    decorated function and convert.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return await asyncio.wait_for(
                func(*args, **kwargs), timeout=seconds
            )

        return wrapper

    return decorator

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

# Tools for order-related operations.
# Integrates with the order management system.

from datetime import datetime, timezone, timedelta
from typing import Optional
import httpx


class OrderStatusTool(Tool):
    """Get the current status of an order."""

    def __init__(
        self,
        orders_client: httpx.AsyncClient,
        max_attempts: int = 3,
        timeout_seconds: float = 10.0,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")
        self.client = orders_client
        self.max_attempts = max_attempts
        self.timeout_seconds = timeout_seconds
        self.retry_backoff_seconds = retry_backoff_seconds

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_order_status",
            description="Retrieve the current status of a customer order including shipping information",
            parameters=[
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The unique order identifier (e.g., ORD-12345)",
                )
            ],
            audit_level="standard",
        )

    async def execute(self, order_id: str) -> ToolResult:
        operation = with_retry(
            max_attempts=self.max_attempts,
            backoff_seconds=self.retry_backoff_seconds,
        )(with_timeout(self.timeout_seconds)(self._execute_once))
        return await operation(order_id=order_id)

    async def _execute_once(self, order_id: str) -> ToolResult:
        start = datetime.now(timezone.utc)

        validation_error = self.validate_params(order_id=order_id)
        if validation_error:
            return ToolResult(
                success=False, data=None, error=validation_error
            )

        try:
            response = await self.client.get(f"/orders/{order_id}")
            response.raise_for_status()
            order_data = response.json()
            if not isinstance(order_data, dict):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Order service returned a non-object payload",
                )
            required = {"id", "status", "total", "created_at"}
            missing = sorted(required - set(order_data))
            if missing:
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        "Order service response missing fields: "
                        + ", ".join(missing)
                    ),
                )

            # Transform to customer-friendly format
            result = {
                "order_id": order_data["id"],
                "status": order_data["status"],
                "status_description": self._get_status_description(
                    order_data["status"]
                ),
                "items_count": len(order_data.get("items", [])),
                "total": order_data["total"],
                "placed_date": order_data["created_at"],
                "estimated_delivery": order_data.get("estimated_delivery"),
                "tracking_number": order_data.get("tracking_number"),
                "carrier": order_data.get("carrier"),
            }

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True, data=result, execution_time_ms=execution_time
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            if e.response.status_code == 404:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Order {order_id} not found",
                )
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to retrieve order: {e}",
            )

    def _get_status_description(self, status: str) -> str:
        descriptions = {
            "pending": "Order received, awaiting processing",
            "processing": "Order is being prepared",
            "shipped": "Order has been shipped",
            "in_transit": "Order is on its way",
            "out_for_delivery": "Order will be delivered today",
            "delivered": "Order has been delivered",
            "cancelled": "Order was cancelled",
        }
        return descriptions.get(status, status)


class ModifyOrderTool(Tool):
    """Modify an existing order."""

    def __init__(self, orders_client: httpx.AsyncClient) -> None:
        if orders_client is None:
            raise ValueError("orders_client is required")
        self.client = orders_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="modify_order",
            description="Modify an existing order. Can update shipping address or cancel items. Only works for orders not yet shipped.",
            parameters=[
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The unique order identifier",
                ),
                ToolParameter(
                    name="action",
                    type="string",
                    description="The modification action to perform",
                    enum=["update_address", "cancel_item", "cancel_order"],
                ),
                ToolParameter(
                    name="details",
                    type="object",
                    description="Action-specific details (address for update_address, item_id for cancel_item)",
                ),
            ],
            requires_confirmation=True,
            audit_level="detailed",
        )

    @with_retry(
        max_attempts=ORDER_MUTATION_TOOL_SETTINGS.max_attempts,
        backoff_seconds=ORDER_MUTATION_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(ORDER_MUTATION_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self, order_id: str, action: str, details: dict[str, Any]
    ) -> ToolResult:
        start = datetime.now(timezone.utc)
        if not isinstance(details, dict):
            return ToolResult(
                success=False,
                data=None,
                error="details must be an object",
            )
        if action == "update_address" and not details:
            return ToolResult(
                success=False,
                data=None,
                error="update_address requires address details",
            )
        if action == "cancel_item" and not details.get("item_id"):
            return ToolResult(
                success=False,
                data=None,
                error="cancel_item requires details.item_id",
            )
        if action not in {"update_address", "cancel_item", "cancel_order"}:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown action: {action}",
            )
        idempotency_key = details.get("idempotency_key") or (
            f"modify:{order_id}:{action}:"
            f"{json.dumps(details, sort_keys=True, default=str)}"
        )
        headers = {"Idempotency-Key": idempotency_key}

        try:
            # First check if order can be modified
            check_response = await self.client.get(f"/orders/{order_id}")
            if check_response.status_code == 404:
                return ToolResult(
                    success=False, data=None, error=f"Order {order_id} not found"
                )
            check_response.raise_for_status()

            order = check_response.json()
            if not isinstance(order, dict) or "status" not in order:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Order service response missing status",
                )
            if order["status"] in ["shipped", "in_transit", "delivered"]:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Cannot modify order in '{order['status']}' status. Please initiate a return instead.",
                )

            if action == "update_address":
                response = await self.client.patch(
                    f"/orders/{order_id}/address",
                    json=details,
                    headers=headers,
                )
            elif action == "cancel_item":
                response = await self.client.delete(
                    f"/orders/{order_id}/items/{details['item_id']}",
                    headers=headers,
                )
            else:
                response = await self.client.post(
                    f"/orders/{order_id}/cancel",
                    json={
                        "reason": details.get("reason", "Customer requested")
                    },
                    headers=headers,
                )

            response.raise_for_status()

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "order_id": order_id,
                    "action": action,
                    "result": "completed",
                    "message": f"Successfully performed {action} on order {order_id}",
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False,
                data=None,
                error=f"Modification failed: {e.response.text}",
            )


class InitiateReturnTool(Tool):
    """Start the return process for an order."""

    def __init__(self, orders_client: httpx.AsyncClient) -> None:
        if orders_client is None:
            raise ValueError("orders_client is required")
        self.client = orders_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="initiate_return",
            description="Start a return for delivered items. Generates a return label and instructions.",
            parameters=[
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The order to return items from",
                ),
                ToolParameter(
                    name="item_ids",
                    type="array",
                    description="List of item IDs to return. Empty list for full order return.",
                    required=False,
                ),
                ToolParameter(
                    name="reason",
                    type="string",
                    description="Reason for the return",
                    enum=[
                        "defective",
                        "wrong_item",
                        "not_as_described",
                        "no_longer_needed",
                        "arrived_late",
                        "other",
                    ],
                ),
                ToolParameter(
                    name="reason_details",
                    type="string",
                    description="Additional details about the return reason",
                    required=False,
                ),
                ToolParameter(
                    name="idempotency_key",
                    type="string",
                    description="Stable unique key for this return request",
                ),
            ],
            requires_confirmation=True,
            audit_level="detailed",
        )

    @with_retry(
        max_attempts=RETURN_TOOL_SETTINGS.max_attempts,
        backoff_seconds=RETURN_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(RETURN_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self,
        order_id: str,
        reason: str,
        item_ids: Optional[list] = None,
        reason_details: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ToolResult:
        start = datetime.now(timezone.utc)
        if not idempotency_key:
            return ToolResult(
                success=False,
                data=None,
                error=(
                    "idempotency_key is required for return side effects; "
                    "derive it from the conversation action id"
                ),
            )

        try:
            # Check order eligibility
            order_response = await self.client.get(f"/orders/{order_id}")
            if order_response.status_code == 404:
                return ToolResult(
                    success=False, data=None, error=f"Order {order_id} not found"
                )
            order_response.raise_for_status()

            order = order_response.json()

            # Check return window (30 days from delivery)
            if order["status"] != "delivered":
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Cannot return order in '{order['status']}' status. Order must be delivered first.",
                )

            delivery_date = datetime.fromisoformat(
                order["delivered_at"].replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) - delivery_date > timedelta(days=30):
                return ToolResult(
                    success=False,
                    data=None,
                    error="Return window has expired (30 days from delivery)",
                )

            return_request = {
                "order_id": order_id,
                "item_ids": item_ids
                or [item["id"] for item in order["items"]],
                "reason": reason,
                "reason_details": reason_details,
            }

            response = await self.client.post(
                "/returns",
                json=return_request,
                headers={"Idempotency-Key": idempotency_key},
            )
            response.raise_for_status()
            return_data = response.json()

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "return_id": return_data["return_id"],
                    "status": "initiated",
                    "return_label_url": return_data["label_url"],
                    "instructions": return_data["instructions"],
                    "refund_estimate": return_data["refund_amount"],
                    "refund_timeline": "3-5 business days after we receive the return",
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to initiate return: {e.response.text}",
            )


class TrackShipmentTool(Tool):
    """Track shipment status with carrier."""

    def __init__(self, shipping_client: httpx.AsyncClient) -> None:
        if shipping_client is None:
            raise ValueError("shipping_client is required")
        self.client = shipping_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="track_shipment",
            description="Get detailed tracking information for a shipment",
            parameters=[
                ToolParameter(
                    name="tracking_number",
                    type="string",
                    description="The carrier tracking number",
                ),
                ToolParameter(
                    name="carrier",
                    type="string",
                    description="The shipping carrier",
                    enum=["fedex", "ups", "usps", "dhl"],
                    required=False,
                ),
            ],
            audit_level="minimal",
        )

    @with_retry(
        max_attempts=TRACKING_TOOL_SETTINGS.max_attempts,
        backoff_seconds=TRACKING_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(TRACKING_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self, tracking_number: str, carrier: Optional[str] = None
    ) -> ToolResult:
        start = datetime.now(timezone.utc)

        try:
            params = {"tracking_number": tracking_number}
            if carrier:
                params["carrier"] = carrier

            response = await self.client.get("/track", params=params)
            response.raise_for_status()
            tracking_data = response.json()

            # Format tracking events
            events = []
            for event in tracking_data.get("events", []):
                events.append(
                    {
                        "timestamp": event["timestamp"],
                        "location": event.get("location", ""),
                        "status": event["status"],
                        "description": event["description"],
                    }
                )

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "tracking_number": tracking_number,
                    "carrier": tracking_data["carrier"],
                    "current_status": tracking_data["status"],
                    "estimated_delivery": tracking_data.get(
                        "estimated_delivery"
                    ),
                    "events": events[:10],  # Last 10 events
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            if e.response.status_code == 404:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Tracking number not found",
                )
            return ToolResult(
                success=False, data=None, error=f"Tracking failed: {e}"
            )

# ============================================================================
# Block 5 (chapter listing #5)
# ============================================================================

# Tools for billing and payment operations.
# Implements patterns that support PCI-DSS compliance requirements.
# Note: Full PCI-DSS compliance requires organizational controls beyond code.


class GetAccountBalanceTool(Tool):
    """Get customer account balance and payment status."""

    def __init__(self, billing_client: httpx.AsyncClient) -> None:
        self.client = billing_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_account_balance",
            description="Get the customer's current account balance, outstanding invoices, and payment status",
            parameters=[
                ToolParameter(
                    name="customer_id",
                    type="string",
                    description="The customer's unique identifier",
                )
            ],
            audit_level="standard",
        )

    @with_retry(
        max_attempts=BILLING_READ_TOOL_SETTINGS.max_attempts,
        backoff_seconds=BILLING_READ_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(BILLING_READ_TOOL_SETTINGS.timeout_seconds)
    async def execute(self, customer_id: str) -> ToolResult:
        start = datetime.now(timezone.utc)

        try:
            response = await self.client.get(
                f"/customers/{customer_id}/balance"
            )
            response.raise_for_status()
            balance_data = response.json()

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "current_balance": balance_data["balance"],
                    "currency": balance_data["currency"],
                    "outstanding_invoices": balance_data["outstanding_count"],
                    "oldest_outstanding": balance_data.get("oldest_due_date"),
                    "credit_available": balance_data.get("credit_limit", 0)
                    - balance_data["balance"],
                    "payment_status": (
                        "current"
                        if balance_data["balance"] <= 0
                        else "outstanding"
                    ),
                    "auto_pay_enabled": balance_data.get("auto_pay", False),
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to retrieve balance: {e}",
            )


class ProcessRefundTool(Tool):
    """Process a refund for a customer."""

    def __init__(self, billing_client: httpx.AsyncClient) -> None:
        self.client = billing_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="process_refund",
            description="Process a refund to the customer's original payment method",
            parameters=[
                ToolParameter(
                    name="customer_id",
                    type="string",
                    description="The customer's unique identifier",
                ),
                ToolParameter(
                    name="order_id",
                    type="string",
                    description="The order ID associated with this refund",
                ),
                ToolParameter(
                    name="amount",
                    type="number",
                    description="The refund amount in cents",
                ),
                ToolParameter(
                    name="reason",
                    type="string",
                    description="Reason for the refund",
                    enum=[
                        "return",
                        "cancellation",
                        "price_adjustment",
                        "service_issue",
                        "goodwill",
                    ],
                ),
                ToolParameter(
                    name="idempotency_key",
                    type="string",
                    description=(
                        "Stable unique key for this logical refund operation"
                    ),
                ),
            ],
            requires_confirmation=True,
            audit_level="detailed",
        )

    @with_retry(
        max_attempts=REFUND_TOOL_SETTINGS.max_attempts,
        backoff_seconds=REFUND_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(REFUND_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self,
        customer_id: str,
        order_id: str,
        amount: int,
        reason: str,
        idempotency_key: Optional[str] = None,
    ) -> ToolResult:
        start = datetime.now(timezone.utc)

        try:
            # Validate refund amount against order
            order_response = await self.client.get(f"/orders/{order_id}")
            if order_response.status_code == 404:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Order {order_id} not found",
                )
            order_response.raise_for_status()

            order = order_response.json()
            max_refundable = order["total"] - order.get("refunded_amount", 0)

            if amount > max_refundable:
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        f"Refund amount ({amount}) exceeds maximum "
                        f"refundable ({max_refundable})"
                    ),
                )

            if not idempotency_key:
                return ToolResult(
                    success=False,
                    data=None,
                    error=(
                        "idempotency_key is required for refund side effects; "
                        "derive it from the conversation action id"
                    ),
                )

            refund_request = {
                "customer_id": customer_id,
                "order_id": order_id,
                "amount": amount,
                "reason": reason,
                "idempotency_key": idempotency_key,
            }

            response = await self.client.post("/refunds", json=refund_request)
            response.raise_for_status()
            refund_data = response.json()

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "refund_id": refund_data["refund_id"],
                    "amount": amount,
                    "status": refund_data["status"],
                    "estimated_arrival": "3-5 business days",
                    "refund_method": refund_data["method"],
                    "confirmation_number": refund_data["confirmation"],
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False,
                data=None,
                error=f"Refund failed: {e.response.text}",
            )


class GetPaymentHistoryTool(Tool):
    """Get customer payment history."""

    def __init__(self, billing_client: httpx.AsyncClient) -> None:
        self.client = billing_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_payment_history",
            description="Get recent payment transactions for a customer",
            parameters=[
                ToolParameter(
                    name="customer_id",
                    type="string",
                    description="The customer's unique identifier",
                ),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Number of transactions to retrieve (max 50)",
                    required=False,
                ),
            ],
            audit_level="standard",
        )

    @with_retry(
        max_attempts=BILLING_READ_TOOL_SETTINGS.max_attempts,
        backoff_seconds=BILLING_READ_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(BILLING_READ_TOOL_SETTINGS.timeout_seconds)
    async def execute(self, customer_id: str, limit: int = 10) -> ToolResult:
        start = datetime.now(timezone.utc)

        limit = min(limit, 50)  # Cap at 50

        try:
            response = await self.client.get(
                f"/customers/{customer_id}/payments", params={"limit": limit}
            )
            response.raise_for_status()
            payments = response.json()

            # Format for customer display (mask card numbers)
            formatted_payments = []
            for payment in payments["transactions"]:
                formatted_payments.append(
                    {
                        "date": payment["created_at"],
                        "amount": payment["amount"],
                        "type": payment["type"],
                        "status": payment["status"],
                        "payment_method": (
                            f"****{payment['card_last_four']}"
                            if payment.get("card_last_four")
                            else payment.get("method")
                        ),
                        "order_id": payment.get("order_id"),
                        "description": payment.get("description"),
                    }
                )

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "transactions": formatted_payments,
                    "total_count": payments["total"],
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to retrieve payment history: {e}",
            )

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

# Tools for technical support operations.
# Integrates with knowledge base and diagnostic systems.


class SearchKnowledgeBaseTool(Tool):
    """Search the product knowledge base."""

    def __init__(
        self, kb_client: httpx.AsyncClient, max_results: int = 5
    ) -> None:
        if kb_client is None:
            raise ValueError("kb_client is required")
        if max_results < 1:
            raise ValueError("max_results must be >= 1")
        self.client = kb_client
        self.max_results = max_results

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="search_knowledge_base",
            description="Search the knowledge base for product documentation, troubleshooting guides, and FAQs",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query describing the customer's issue or question",
                ),
                ToolParameter(
                    name="product_category",
                    type="string",
                    description="Filter by product category",
                    required=False,
                ),
                ToolParameter(
                    name="article_type",
                    type="string",
                    description="Filter by article type",
                    enum=[
                        "troubleshooting",
                        "how_to",
                        "faq",
                        "specification",
                    ],
                    required=False,
                ),
            ],
            audit_level="minimal",
        )

    @with_retry(
        max_attempts=SUPPORT_SEARCH_TOOL_SETTINGS.max_attempts,
        backoff_seconds=SUPPORT_SEARCH_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(SUPPORT_SEARCH_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self,
        query: str,
        product_category: Optional[str] = None,
        article_type: Optional[str] = None,
    ) -> ToolResult:
        start = datetime.now(timezone.utc)

        try:
            params = {"q": query, "limit": self.max_results}
            if product_category:
                params["category"] = product_category
            if article_type:
                params["type"] = article_type

            response = await self.client.get(
                "/articles/search", params=params
            )
            response.raise_for_status()
            results = response.json()

            articles = []
            for article in results["articles"]:
                articles.append(
                    {
                        "article_id": article["id"],
                        "title": article["title"],
                        "summary": article["summary"],
                        "relevance_score": article["score"],
                        "article_type": article["type"],
                        "url": article["url"],
                    }
                )

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "articles_found": len(articles),
                    "articles": articles,
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False,
                data=None,
                error=f"Knowledge base search failed: {e}",
            )


class RunDiagnosticTool(Tool):
    """Run automated diagnostics for a customer's product."""

    def __init__(self, diagnostic_client: httpx.AsyncClient) -> None:
        self.client = diagnostic_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="run_diagnostic",
            description="Run automated diagnostics on a customer's product or account to identify issues",
            parameters=[
                ToolParameter(
                    name="customer_id",
                    type="string",
                    description="The customer's unique identifier",
                ),
                ToolParameter(
                    name="diagnostic_type",
                    type="string",
                    description="Type of diagnostic to run",
                    enum=[
                        "connectivity",
                        "account_health",
                        "product_status",
                        "integration_check",
                        "performance",
                    ],
                ),
                ToolParameter(
                    name="product_id",
                    type="string",
                    description="Specific product to diagnose",
                    required=False,
                ),
            ],
            audit_level="standard",
        )

    @with_retry(
        max_attempts=DIAGNOSTIC_TOOL_SETTINGS.max_attempts,
        backoff_seconds=DIAGNOSTIC_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(DIAGNOSTIC_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self,
        customer_id: str,
        diagnostic_type: str,
        product_id: Optional[str] = None,
    ) -> ToolResult:
        start = datetime.now(timezone.utc)

        try:
            diagnostic_request = {
                "customer_id": customer_id,
                "type": diagnostic_type,
                "product_id": product_id,
            }

            response = await self.client.post(
                "/diagnostics/run", json=diagnostic_request
            )
            response.raise_for_status()
            diagnostic_result = response.json()

            # Format findings
            findings = []
            for finding in diagnostic_result.get("findings", []):
                findings.append(
                    {
                        "severity": finding["severity"],
                        "component": finding["component"],
                        "issue": finding["description"],
                        "recommendation": finding.get("recommendation"),
                    }
                )

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "diagnostic_id": diagnostic_result["id"],
                    "status": diagnostic_result["status"],
                    "overall_health": diagnostic_result["health_score"],
                    "findings_count": len(findings),
                    "critical_issues": sum(
                        1 for f in findings if f["severity"] == "critical"
                    ),
                    "findings": findings,
                    "next_steps": diagnostic_result.get(
                        "recommended_actions", []
                    ),
                },
                execution_time_ms=execution_time,
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False, data=None, error=f"Diagnostic failed: {e}"
            )

# ============================================================================
# Block 7 (chapter listing #7)
# ============================================================================

# Tools for CRM operations.
# Used by all agents for customer context.


class IdentifyCustomerTool(Tool):
    """Identify and authenticate a customer."""

    def __init__(self, crm_client: httpx.AsyncClient) -> None:
        if crm_client is None:
            raise ValueError("crm_client is required")
        self.client = crm_client

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="identify_customer",
            description="Look up and verify a customer's identity using their email, phone, or account number",
            parameters=[
                ToolParameter(
                    name="identifier",
                    type="string",
                    description="Customer email, phone number, or account number",
                ),
                ToolParameter(
                    name="identifier_type",
                    type="string",
                    description="Type of identifier provided",
                    enum=["email", "phone", "account_number"],
                    required=False,
                ),
            ],
            audit_level="detailed",
        )

    @with_retry(
        max_attempts=CRM_LOOKUP_TOOL_SETTINGS.max_attempts,
        backoff_seconds=CRM_LOOKUP_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(CRM_LOOKUP_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self, identifier: str, identifier_type: Optional[str] = None
    ) -> ToolResult:
        start = datetime.now(timezone.utc)

        # Auto-detect identifier type if not provided
        if not identifier_type:
            if "@" in identifier:
                identifier_type = "email"
            elif identifier.replace("-", "").replace("+", "").isdigit():
                identifier_type = "phone"
            else:
                identifier_type = "account_number"

        try:
            response = await self.client.get(
                "/customers/lookup",
                params={"identifier": identifier, "type": identifier_type},
            )

            if response.status_code == 404:
                return ToolResult(
                    success=False,
                    data=None,
                    error="Customer not found. Please verify the information provided.",
                )

            response.raise_for_status()
            customer_data = response.json()

            # Build customer object
            customer = {
                "customer_id": customer_data["id"],
                "name": customer_data["name"],
                "email": customer_data["email"],
                "phone": customer_data.get("phone"),
                "tier": customer_data.get("tier", "standard"),
                "lifetime_value": customer_data.get("lifetime_value", 0),
                "account_age_days": customer_data.get("account_age_days", 0),
                "open_tickets": customer_data.get("open_tickets", 0),
                "recent_orders": customer_data.get("recent_orders", [])[:5],
                "preferences": customer_data.get("preferences", {}),
                "verified": True,
            }

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True, data=customer, execution_time_ms=execution_time
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code >= 500:
                raise
            return ToolResult(
                success=False, data=None, error=f"Customer lookup failed: {e}"
            )


class ClassifyIntentTool(Tool):
    """Classify customer intent from their message."""

    def __init__(
        self,
        llm_client: "anthropic.AsyncAnthropic",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 200,
    ) -> None:
        if not model:
            raise ValueError("model must be non-empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.llm_client = llm_client
        self.model = model
        self.max_tokens = max_tokens

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="classify_intent",
            description="Analyze a customer message to determine their primary intent and route appropriately",
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="The customer's message to analyze",
                ),
                ToolParameter(
                    name="conversation_context",
                    type="string",
                    description="Previous conversation context for better classification",
                    required=False,
                ),
            ],
            audit_level="standard",
        )

    @with_retry(
        max_attempts=INTENT_CLASSIFIER_TOOL_SETTINGS.max_attempts,
        backoff_seconds=INTENT_CLASSIFIER_TOOL_SETTINGS.retry_backoff_seconds,
    )
    @with_timeout(INTENT_CLASSIFIER_TOOL_SETTINGS.timeout_seconds)
    async def execute(
        self, message: str, conversation_context: Optional[str] = None
    ) -> ToolResult:
        start = datetime.now(timezone.utc)

        classification_prompt = f"""Analyze this customer service message and classify the intent.

Customer message: {message}
{"Previous context: " + conversation_context if conversation_context else ""}

Classify into exactly one primary intent:
- order_status: Checking on an order, tracking, delivery questions
- order_modification: Changing, canceling, or modifying an order
- return_refund: Returning items or requesting refunds
- technical_support: Product issues, troubleshooting, how-to questions
- billing_inquiry: Payment questions, account balance, invoices
- billing_dispute: Disputing a charge, incorrect billing
- account_management: Profile updates, password, preferences
- general_inquiry: General questions not fitting other categories
- complaint: Expressing dissatisfaction, requesting escalation
- feedback: Providing positive or constructive feedback

Respond with JSON only:
{{"intent": "intent_name", "confidence": 0.0-1.0, "entities": {{}}, "sentiment": -1.0 to 1.0}}"""

        try:
            response = await self.llm_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": classification_prompt}],
            )

            import json as _json
            import logging as _logging

            raw_text = response.content[0].text
            try:
                result = _json.loads(raw_text)
            except _json.JSONDecodeError as parse_err:
                _logging.getLogger(__name__).warning(
                    "Intent classifier returned non-JSON: %s", raw_text
                )
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Intent parsing failed: {parse_err}",
                )

            # Map intent to agent type
            intent_to_agent = {
                "order_status": "order",
                "order_modification": "order",
                "return_refund": "order",
                "technical_support": "technical",
                "billing_inquiry": "billing",
                "billing_dispute": "billing",
                "account_management": "billing",
                "general_inquiry": "triage",
                "complaint": "escalation",
                "feedback": "triage",
            }

            execution_time = (
                datetime.now(timezone.utc) - start
            ).total_seconds() * 1000
            return ToolResult(
                success=True,
                data={
                    "intent": str(result["intent"]),
                    "confidence": float(result["confidence"]),
                    "recommended_agent": intent_to_agent.get(
                        str(result["intent"]), "triage"
                    ),
                    "entities": result.get("entities", {}),
                    "sentiment": result.get("sentiment", 0.0),
                },
                execution_time_ms=execution_time,
            )

        except (KeyError, TypeError, ValueError) as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Intent classification response invalid: {e}",
            )

# ============================================================================
# Block 8 (chapter listing #8)
# ============================================================================

# Base agent implementation providing common capabilities.
# All specialized agents inherit from this class.

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import asyncio
import json
import logging
import textwrap
import threading

try:  # Optional provider SDK; keep this module importable without it.
    import anthropic  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - dependency not required for examples
    class _AnthropicStub:
        """Fallback shim so ``except anthropic.*`` clauses still resolve.

        When the real ``anthropic`` SDK is not installed, code paths that
        catch provider exceptions would otherwise raise ``NameError`` at
        except-evaluation time and skip their fallback (e.g., returning a
        neutral quality score). The stub exposes the exception names this
        module references; the stub classes never actually fire because no
        real call site can raise them.
        """

        class APITimeoutError(Exception):
            pass

        class RateLimitError(Exception):
            pass

        class APIConnectionError(Exception):
            pass

        class APIError(Exception):
            pass

        class AuthenticationError(Exception):
            pass

        class BadRequestError(Exception):
            pass

        class _MissingClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise ImportError(
                    "anthropic package is required for LLM client creation"
                )

        class AsyncAnthropic(_MissingClient):
            pass

        class Anthropic(_MissingClient):
            pass

    anthropic = _AnthropicStub  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Response from an agent."""

    message: str
    agent_type: AgentType
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    should_transfer: bool = False
    transfer_to: Optional[AgentType] = None
    transfer_reason: Optional[str] = None
    should_escalate: bool = False
    escalation_reason: Optional[str] = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderCircuitBreaker:
    """Small fail-fast circuit breaker for provider outages.

    Dual-context concurrency contract: state mutations (_failures,
    _open_until) are guarded by a ``threading.Lock`` so synchronous
    callers running on worker threads see a consistent view. An
    ``asyncio.Lock`` is exposed via ``async_lock`` for cooperative
    callers that want to serialize their own check-then-act sequences
    around ``allow_request``/``record_failure``. Both locks must be
    held by the same logical caller; do not interleave sync and async
    mutators on the same instance from different threads without
    additional coordination at the call site.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        open_seconds: float = 30.0,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        if open_seconds <= 0:
            raise ValueError("open_seconds must be > 0")
        self.failure_threshold = failure_threshold
        self.open_seconds = open_seconds
        self._failures = 0
        self._open_until: Optional[datetime] = None
        self._lock = threading.Lock()
        self.async_lock = asyncio.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            return (
                self._open_until is None
                or datetime.now(timezone.utc) >= self._open_until
            )

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._open_until = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._open_until = datetime.now(timezone.utc) + timedelta(
                    seconds=self.open_seconds
                )


class BaseAgent(ABC):
    """Base class for all customer service agents.

    Turn accounting lives on the ``Conversation`` (not on the agent
    instance) so a single shared agent serving multiple concurrent
    conversations does not leak its turn counter across them. See
    :py:meth:`process_message`, which reads/writes
    ``conversation.turn_count``.
    """

    def __init__(
        self,
        config: AgentConfig,
        tools: list[Tool],
        llm_client: anthropic.AsyncAnthropic,
    ) -> None:
        self.config = config
        self.tools = {tool.definition.name: tool for tool in tools}
        self.llm_client = llm_client
        self._llm_circuit = ProviderCircuitBreaker()

    @property
    @abstractmethod
    def agent_type(self) -> AgentType:
        """Return the agent type."""
        pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM."""
        return [
            tool.definition.to_anthropic_schema()
            for tool in self.tools.values()
        ]

    async def _create_llm_message(self, **kwargs: Any) -> Any:
        """Call the provider through timeout, retry, and circuit controls."""
        if not self._llm_circuit.allow_request():
            raise RuntimeError("LLM provider circuit breaker is open")
        try:
            response = await asyncio.wait_for(
                self.llm_client.messages.create(**kwargs),
                timeout=self.config.timeout_seconds,
            )
            self._llm_circuit.record_success()
            return response
        except (
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            asyncio.TimeoutError,
        ):
            self._llm_circuit.record_failure()
            raise

    async def process_message(
        self, conversation: Conversation, user_message: str
    ) -> AgentResponse:
        """Process a user message and generate a response."""
        # Per-conversation turn count (rather than per-agent) so a
        # shared agent does not bleed counts across conversations.
        conversation.turn_count += 1

        # Check turn limit
        if conversation.turn_count > self.config.max_turns:
            return AgentResponse(
                message="I've been working on this for a while. Let me connect you with a colleague who can continue helping.",
                agent_type=self.agent_type,
                should_escalate=True,
                escalation_reason="Turn limit exceeded",
            )

        # Build messages for LLM
        messages = await self._build_messages(conversation, user_message)

        # Specific-exception handling: a blanket ``except Exception``
        # turns every transient provider blip into a wasted human-review
        # slot. Distinguish (a) transient infrastructure errors that
        # warrant a single retry within this turn, (b) permanent
        # caller-side errors (auth, malformed request) that must
        # escalate without retry, and (c) genuine unknowns.
        max_transient_retries = self.config.max_transient_retries
        backoff_seconds = self.config.llm_retry_backoff_seconds
        last_exc: Optional[Exception] = None

        response = None
        for attempt in range(max_transient_retries + 1):
            try:
                response = await self._create_llm_message(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=self.system_prompt,
                    tools=self.get_tool_schemas(),
                    messages=messages,
                    timeout=self.config.timeout_seconds,
                )
                break

            except (
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                asyncio.TimeoutError,
            ) as exc:
                # Transient infrastructure: one retry, then escalate as
                # an infra problem so the human reviewer knows it was
                # not a user-facing failure.
                last_exc = exc
                if attempt < max_transient_retries:
                    logger.warning(
                        "Transient LLM error in %s (attempt %d/%d): %s",
                        self.agent_type, attempt + 1,
                        max_transient_retries + 1, exc,
                    )
                    await asyncio.sleep(backoff_seconds * (2 ** attempt))
                    continue
                logger.warning(
                    "Transient retries exhausted for %s: %s",
                    self.agent_type, exc,
                )
                return AgentResponse(
                    message=(
                        "Our system is briefly slow. Let me connect you "
                        "with a colleague who can continue helping."
                    ),
                    agent_type=self.agent_type,
                    should_escalate=True,
                    escalation_reason=(
                        f"Infrastructure timeout/rate-limit: "
                        f"{type(exc).__name__}"
                    ),
                )

            except anthropic.AuthenticationError as exc:
                # Permanent: no point retrying or burning a human slot;
                # alert ops by re-raising so the surrounding error path
                # surfaces this as a deployment problem.
                logger.exception(
                    "Auth error in %s; surfacing to ops: %s",
                    self.agent_type, exc,
                )
                raise

            except anthropic.BadRequestError as exc:
                # Permanent 4xx (e.g., context too long after history
                # append). Escalate with the specific reason rather
                # than retry the same request.
                logger.warning(
                    "Permanent 4xx in %s: %s", self.agent_type, exc,
                )
                return AgentResponse(
                    message=(
                        "I'm having trouble with this request. "
                        "Connecting you with a human agent who can help."
                    ),
                    agent_type=self.agent_type,
                    should_escalate=True,
                    escalation_reason=(
                        f"Bad request {exc.status_code}: {exc}"
                    ),
                )

            except (
                anthropic.APIError,
                TimeoutError,
                ConnectionError,
                OSError,
            ) as exc:
                # Known non-retryable runtime failure: log the stack
                # trace and escalate generically.
                logger.exception(
                    "Unhandled error in %s: %s", self.agent_type, exc,
                )
                return AgentResponse(
                    message=(
                        "I apologize, but I'm having a technical issue. "
                        "Let me connect you with someone who can help."
                    ),
                    agent_type=self.agent_type,
                    should_escalate=True,
                    escalation_reason=f"Unhandled: {type(exc).__name__}",
                )

        if response is None:
            logger.error(
                "process_message retry loop exited without response for %s: %s",
                self.agent_type, last_exc,
            )
            return AgentResponse(
                message="Connecting you with a human agent.",
                agent_type=self.agent_type,
                should_escalate=True,
                escalation_reason="Exhausted retries with no result",
            )

        try:
            return await self._process_response(response, conversation)
        except anthropic.AuthenticationError as exc:
            logger.exception(
                "Auth error in %s; surfacing to ops: %s",
                self.agent_type, exc,
            )
            raise
        except anthropic.BadRequestError as exc:
            logger.warning(
                "Permanent 4xx in %s: %s", self.agent_type, exc,
            )
            return AgentResponse(
                message=(
                    "I'm having trouble with this request. "
                    "Connecting you with a human agent who can help."
                ),
                agent_type=self.agent_type,
                should_escalate=True,
                escalation_reason=f"Bad request {exc.status_code}: {exc}",
            )
        except (
            anthropic.APIError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            logger.exception(
                "Unhandled post-tool error in %s: %s", self.agent_type, exc,
            )
            return AgentResponse(
                message=(
                    "I apologize, but I'm having a technical issue. "
                    "Let me connect you with someone who can help."
                ),
                agent_type=self.agent_type,
                should_escalate=True,
                escalation_reason=f"Unhandled: {type(exc).__name__}",
            )

        # Defensive: loop should always return; surface explicit error.
        logger.error(
            "process_message retry loop exited without return for %s: %s",
            self.agent_type, last_exc,
        )
        return AgentResponse(
            message="Connecting you with a human agent.",
            agent_type=self.agent_type,
            should_escalate=True,
            escalation_reason="Exhausted retries with no result",
        )

    async def _build_messages(
        self, conversation: Conversation, user_message: str
    ) -> list[dict[str, Any]]:
        """Build message history for LLM.

        Production deployments should plug in the ``ConversationManager``
        from Chapter 2 here (token-aware truncation plus rolling summary)
        so that long conversations do not lose critical early context
        such as the customer identification turn or escalation notes.
        When a ``conversation_manager`` is attached we delegate to its
        ``get_context_window`` so the LLM sees a summary + the recent
        window instead of a hard slice of the last N messages.
        """
        messages = []

        # Prefer the Ch02 ConversationManager when available, so the LLM
        # sees a summary plus a recent window rather than a blind slice.
        cm = getattr(self, "conversation_manager", None)
        if cm is not None and hasattr(cm, "get_context_window"):
            window = cm.get_context_window(conversation.conversation_id)
            if inspect.isawaitable(window):
                window = await window
            if getattr(window, "summary", None):
                messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Earlier conversation summary: "
                            f"{window.summary}"
                        ),
                    }
                )
            for msg in getattr(window, "messages", []):
                role = "user" if msg.role == "customer" else "assistant"
                messages.append({"role": role, "content": msg.content})
        else:
            # Fallback: last 10 turns. Documented limitation -- long
            # conversations will lose context before the window.
            for msg in list(conversation.messages)[-10:]:
                role = "user" if msg.role == "customer" else "assistant"
                messages.append({"role": role, "content": msg.content})

        # Add current message
        messages.append({"role": "user", "content": user_message})

        return messages

    async def _process_response(
        self, response: Any, conversation: Conversation
    ) -> AgentResponse:
        """Process LLM response and execute tool calls."""
        tool_calls = []
        final_message = ""

        for content_block in response.content:
            if content_block.type == "text":
                final_message = content_block.text
            elif content_block.type == "tool_use":
                tool_result = await self._execute_tool(
                    content_block.name, content_block.input, conversation
                )
                tool_calls.append(
                    {
                        "tool": content_block.name,
                        "input": content_block.input,
                        "result": tool_result.to_llm_response(),
                        "success": tool_result.success,
                    }
                )

        # If we had tool calls, make another LLM call with results
        if tool_calls:
            final_message = await self._get_final_response(
                conversation, tool_calls, preamble=final_message
            )

        # Check for transfer or escalation signals
        should_transfer, transfer_to, transfer_reason = self._check_transfer(
            final_message
        )
        should_escalate, escalation_reason = self._check_escalation(
            final_message, conversation
        )

        return AgentResponse(
            message=final_message,
            agent_type=self.agent_type,
            tool_calls=tool_calls,
            should_transfer=should_transfer,
            transfer_to=transfer_to,
            transfer_reason=transfer_reason,
            should_escalate=should_escalate,
            escalation_reason=escalation_reason,
        )

    async def _execute_tool(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        conversation: Conversation,
    ) -> ToolResult:
        """Execute a tool and record the result."""
        if tool_name not in self.tools:
            return ToolResult(
                success=False, data=None, error=f"Unknown tool: {tool_name}"
            )

        tool = self.tools[tool_name]

        # Check if tool requires confirmation
        if tool.definition.requires_confirmation:
            logger.info(
                f"Tool {tool_name} requires confirmation: {tool_input}"
            )

        if not isinstance(tool_input, dict):
            result = ToolResult(
                success=False,
                data=None,
                error=f"Invalid input for tool {tool_name}",
            )
        else:
            validation_error = tool.validate_params(**tool_input)
            if validation_error is not None:
                result = ToolResult(
                    success=False,
                    data=None,
                    error=validation_error,
                )
            else:
                result = None

        try:
            if result is None:
                result = await asyncio.wait_for(
                    tool.execute(**tool_input),
                    timeout=self.config.timeout_seconds,
                )
        except asyncio.TimeoutError:
            result = ToolResult(
                success=False,
                data=None,
                error=f"Tool {tool_name} timed out",
            )
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as exc:
            logger.exception(
                "Tool %s failed with %s", tool_name, type(exc).__name__
            )
            result = ToolResult(
                success=False,
                data=None,
                error=f"Tool {tool_name} failed: {type(exc).__name__}",
            )
        except Exception as exc:
            # Generic fallback: an unexpected tool bug should degrade
            # the agent turn to a failed ToolResult rather than crash
            # the whole conversation loop. Stack trace is preserved
            # via logger.exception for triage.
            logger.exception(
                "Tool %s raised unhandled %s",
                tool_name,
                type(exc).__name__,
            )
            result = ToolResult(
                success=False,
                data=None,
                error=str(exc),
            )

        # Record in conversation context
        conversation.context.add_tool_result(
            tool_name,
            {
                "input": tool_input,
                "success": result.success,
                "execution_time_ms": result.execution_time_ms,
            },
        )

        return result

    async def _get_final_response(
        self,
        conversation: Conversation,
        tool_calls: list[dict[str, Any]],
        preamble: str = "",
    ) -> str:
        """Get final response after tool execution.

        Note: ``BadRequestError`` (permanent error from a malformed
        prompt or schema) is NOT caught here; callers should
        pre-validate inputs or wrap this method in their own
        try/except so a permanent 4xx becomes a degraded
        ``AgentResponse`` rather than crashing the agent loop.
        """
        # Build tool results message
        tool_results_text = "\n".join(
            [
                f"Tool: {tc['tool']}\nResult: {tc['result']}"
                for tc in tool_calls
            ]
        )

        messages = []
        for msg in conversation.get_history_for_llm():
            source_role = msg.get("role")
            if source_role in ("customer", "user"):
                role = "user"
            elif source_role in ("agent", "assistant"):
                role = "assistant"
            else:
                continue
            content = msg.get("content")
            if content:
                messages.append({"role": role, "content": content})
        prompt = "Based on these tool results, provide a helpful response to the customer:"
        if preamble:
            prompt += f"\n\nModel preamble before tool execution:\n{preamble}"
        messages.append(
            {
                "role": "user",
                "content": f"{prompt}\n\n{tool_results_text}",
            }
        )

        max_transient_retries = self.config.max_transient_retries
        backoff_seconds = self.config.llm_retry_backoff_seconds
        for attempt in range(max_transient_retries + 1):
            try:
                response = await self._create_llm_message(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=self.system_prompt,
                    messages=messages,
                    timeout=self.config.timeout_seconds,
                )
                return response.content[0].text
            except (
                anthropic.APITimeoutError,
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                asyncio.TimeoutError,
            ) as exc:
                if attempt == max_transient_retries:
                    raise
                logger.warning(
                    "Transient final-response error in %s: %s",
                    self.agent_type,
                    exc,
                )
                await asyncio.sleep(backoff_seconds * (2**attempt))

        raise RuntimeError("final-response retry loop exhausted")

    def _check_transfer(
        self, message: str
    ) -> tuple[bool, Optional[AgentType], Optional[str]]:
        """Check if the response indicates a transfer is needed."""
        # Simple keyword detection - in production, use more sophisticated detection
        transfer_keywords = {
            "billing": AgentType.BILLING,
            "payment": AgentType.BILLING,
            "technical": AgentType.TECHNICAL,
            "troubleshoot": AgentType.TECHNICAL,
            "order": AgentType.ORDER,
            "shipping": AgentType.ORDER,
        }

        message_lower = message.lower()
        if (
            "let me transfer" in message_lower
            or "connect you with" in message_lower
        ):
            for keyword, agent_type in transfer_keywords.items():
                if keyword in message_lower:
                    return (
                        True,
                        agent_type,
                        f"Transfer requested for {keyword} issue",
                    )

        return False, None, None

    def _check_escalation(
        self, message: str, conversation: Conversation
    ) -> tuple[bool, Optional[str]]:
        """Check if escalation to human is needed.

        Heuristic: word-boundary matches for unambiguous escalation
        keywords (``supervisor``, ``escalate``) plus a verb+target
        context pattern (``speak``/``talk``/... within 20 characters of
        ``supervisor``/``manager``/``human``). Avoids false positives
        on substrings like "my manager said ...". This is a
        deterministic backstop only, not a replacement for LLM intent
        classification.
        """
        import re

        # Check sentiment
        if conversation.context.sentiment_score < -0.7:
            return True, "Customer sentiment very negative"

        keyword_re = re.compile(
            r"\b(supervisor|escalate|escalation)\b", re.IGNORECASE
        )
        request_re = re.compile(
            r"\b(speak|talk|connect|transfer|escalate)\b"
            r".{0,20}\b(supervisor|manager|human|agent|person)\b",
            re.IGNORECASE | re.DOTALL,
        )
        if keyword_re.search(message):
            return True, "Customer requested human agent"
        if request_re.search(message):
            return True, "Customer requested human agent"

        return False, None

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

# Triage agent - the front door of customer service.
# Routes customers to appropriate specialized agents.


class TriageAgent(BaseAgent):
    """
    The triage agent is the first point of contact.
    It identifies the customer, classifies their intent,
    and routes to the appropriate specialist.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.TRIAGE

    @property
    def system_prompt(self) -> str:
        return textwrap.dedent(
            """\
            You are a customer service triage agent. Your role is to:

            1. GREET the customer warmly and professionally
            2. IDENTIFY the customer (ask for email or account number if not already known)
            3. UNDERSTAND their issue by asking clarifying questions if needed
            4. ROUTE them to the appropriate specialist agent

            You have access to these tools:
            - identify_customer: Look up customer by email, phone, or account number
            - classify_intent: Analyze the customer's message to determine their needs

            ROUTING GUIDELINES:
            - Order issues (status, modifications, returns) -> Order Agent
            - Technical problems (product issues, troubleshooting) -> Technical Support Agent
            - Payment/billing questions -> Billing Agent
            - Complaints or requests for human -> Escalation Agent

            IMPORTANT RULES:
            - Be concise but friendly
            - Never attempt to solve issues yourself - route to specialists
            - If the customer's intent is unclear, ask ONE clarifying question
            - Always confirm the customer's identity before routing
            - If you cannot identify the customer after 2 attempts, escalate

            Example interaction:
            Customer: "My order hasn't arrived"
            You: "I'd be happy to help you track that order. Could you please provide your email address or account number so I can look up your information?"
            Customer: "john@example.com"
            [Use identify_customer tool]
            [Use classify_intent tool]
            You: "Thank you, John. I can see you're asking about an order delivery. Let me connect you with our order specialist who can provide detailed tracking information."

            Remember: Your job is to route efficiently, not to resolve issues.
            """
        ).rstrip("\n")

    async def process_message(
        self, conversation: Conversation, user_message: str
    ) -> AgentResponse:
        """Override to add triage-specific logic."""
        # Check if customer is already identified
        if not conversation.context.customer.customer_id:
            # Need to identify customer first
            identify_result = await self._attempt_identification(user_message)
            if identify_result:
                conversation.context.customer = Customer(**identify_result)

        # If customer is identified, classify intent and route
        if conversation.context.customer.customer_id:
            intent_result = await self._classify_and_route(
                user_message, conversation
            )
            if intent_result:
                return intent_result

        # Fall back to base processing
        return await super().process_message(conversation, user_message)

    async def _attempt_identification(self, message: str) -> Optional[dict]:
        """Try to identify customer from message."""
        # Extract potential identifiers from message
        import re

        # Check for email
        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", message)
        if email_match:
            tool = self.tools.get("identify_customer")
            if tool:
                result = await tool.execute(
                    identifier=email_match.group(), identifier_type="email"
                )
                if result.success:
                    return result.data

        return None

    async def _classify_and_route(
        self, message: str, conversation: Conversation
    ) -> Optional[AgentResponse]:
        """Classify intent and determine routing."""
        tool = self.tools.get("classify_intent")
        if not tool:
            return None

        context = "\n".join(
            [m.content for m in list(conversation.messages)[-3:]]
        )
        result = await tool.execute(
            message=message, conversation_context=context
        )

        if not result.success:
            return None

        intent_data = result.data
        if not isinstance(intent_data, dict):
            logger.warning("Intent classifier returned non-object payload")
            return None
        required_fields = {
            "intent",
            "confidence",
            "sentiment",
            "recommended_agent",
        }
        if not required_fields.issubset(intent_data):
            logger.warning(
                "Intent classifier missing fields: %s",
                sorted(required_fields - set(intent_data)),
            )
            return None
        recommended_agent = intent_data["recommended_agent"]
        try:
            target_agent = AgentType(recommended_agent)
        except ValueError:
            logger.warning(
                "Intent classifier recommended invalid agent: %s",
                recommended_agent,
            )
            return None
        try:
            confidence = float(intent_data["confidence"])
            sentiment = float(intent_data["sentiment"])
        except (TypeError, ValueError):
            logger.warning("Intent classifier returned invalid scores")
            return None

        conversation.context.intent = intent_data["intent"]
        conversation.context.intent_confidence = confidence
        conversation.context.sentiment_score = sentiment

        # Generate handoff message
        handoff_messages = {
            AgentType.ORDER: "I'll connect you with our order specialist who can help you with that.",
            AgentType.TECHNICAL: "Let me transfer you to our technical support team who can assist with this.",
            AgentType.BILLING: "I'll connect you with our billing team who can help with your payment question.",
            AgentType.ESCALATION: "I understand your concern. Let me connect you with a senior specialist.",
        }

        if target_agent != AgentType.TRIAGE:
            return AgentResponse(
                message=handoff_messages.get(
                    target_agent, "Let me connect you with a specialist."
                ),
                agent_type=self.agent_type,
                should_transfer=True,
                transfer_to=target_agent,
                transfer_reason=f"Intent: {intent_data['intent']} (confidence: {intent_data['confidence']:.2f})",
            )

        return None

# ============================================================================
# Block 10 (chapter listing #10)
# ============================================================================

# Order agent - handles order-related inquiries and actions.


class OrderAgent(BaseAgent):
    """
    Specialized agent for order management.
    Handles status inquiries, modifications, and returns.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ORDER

    @property
    def system_prompt(self) -> str:
        return textwrap.dedent(
            """\
            You are a customer service specialist for order management. You help customers with:

            1. Order status and tracking
            2. Order modifications (before shipping)
            3. Order cancellations
            4. Returns and exchanges

            You have access to these tools:
            - get_order_status: Check order status and shipping info
            - modify_order: Change shipping address, cancel items, or cancel order
            - initiate_return: Start return process for delivered items
            - track_shipment: Get detailed tracking information

            GUIDELINES:

            For ORDER STATUS:
            - Always provide the current status clearly
            - If shipped, include tracking information
            - Give estimated delivery dates when available

            For ORDER MODIFICATIONS:
            - Check if the order can be modified (not shipped yet)
            - Confirm changes before making them
            - Explain any impacts (refunds, timing)

            For RETURNS:
            - Verify the item is within the return window (30 days)
            - Ask for the reason to set expectations
            - Provide clear return instructions and timeline

            For TRACKING:
            - Provide the most recent tracking events
            - Explain any delays if visible
            - Set realistic delivery expectations

            TONE:
            - Be empathetic if there are delivery issues
            - Be proactive in offering solutions
            - Never make promises you cannot keep

            If the customer's issue requires billing help or technical support, acknowledge their need and indicate you'll transfer them to the right specialist.
            """
        ).rstrip("\n")

    async def process_message(
        self, conversation: Conversation, user_message: str
    ) -> AgentResponse:
        """Process order-related messages."""
        # Extract order ID if mentioned
        import re

        order_match = re.search(r"ORD-\d+|\b\d{6,}\b", user_message)
        if order_match:
            conversation.context.extracted_entities["order_id"] = (
                order_match.group()
            )

        return await super().process_message(conversation, user_message)

# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

# Technical support agent - handles product issues and troubleshooting.


class TechnicalSupportAgent(BaseAgent):
    """
    Specialized agent for technical support.
    Uses knowledge base and diagnostics to resolve issues.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.TECHNICAL

    @property
    def system_prompt(self) -> str:
        return textwrap.dedent(
            """\
            You are a technical support specialist. You help customers with:

            1. Product troubleshooting
            2. Setup and configuration guidance
            3. Feature explanations
            4. Technical issue diagnosis

            You have access to these tools:
            - search_knowledge_base: Find relevant documentation and guides
            - run_diagnostic: Run automated diagnostics on customer's product/account
            - get_product_info: Get product specifications and compatibility info
            - create_ticket: Create a support ticket for issues requiring engineering

            TROUBLESHOOTING APPROACH:

            1. UNDERSTAND the issue
               - Ask clarifying questions about symptoms
               - Understand when the issue started
               - Check if anything changed recently

            2. DIAGNOSE systematically
               - Search knowledge base for similar issues
               - Run relevant diagnostics
               - Identify root cause if possible

            3. RESOLVE or ESCALATE
               - Provide step-by-step guidance for resolvable issues
               - Create a ticket for issues requiring engineering
               - Escalate to human for complex or sensitive issues

            GUIDELINES:
            - Be patient and avoid jargon
            - Provide numbered steps for instructions
            - Confirm each step before moving to the next
            - If an issue persists after 3 attempts, create a ticket or escalate
            - Never ask customers to do anything that could harm their data

            For billing or order issues, acknowledge and indicate transfer to appropriate specialist.
            """
        ).rstrip("\n")


class BillingAgent(BaseAgent):
    """
    Specialized agent for billing and payment issues.
    Has access to payment systems with PCI-DSS controls.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.BILLING

    @property
    def system_prompt(self) -> str:
        return textwrap.dedent(
            """\
            You are a billing specialist. You help customers with:

            1. Account balance inquiries
            2. Payment history review
            3. Refund processing
            4. Payment method updates

            You have access to these tools:
            - get_account_balance: Check current balance and outstanding invoices
            - get_payment_history: View recent transactions
            - process_refund: Issue refunds for eligible orders
            - update_payment_method: Help update payment information

            IMPORTANT SECURITY RULES:
            - NEVER ask for full credit card numbers
            - NEVER display full card numbers (only last 4 digits)
            - Always verify customer identity before discussing account details
            - Refunds require customer confirmation

            REFUND GUIDELINES:
            - Verify the order is eligible for refund
            - Check refund hasn't already been processed
            - Explain the refund timeline (3-5 business days)
            - Provide confirmation number

            DISPUTE HANDLING:
            - Listen to the customer's concern fully
            - Review the transaction details
            - If valid, process appropriate refund
            - If unclear, escalate to human for review

            TONE:
            - Be precise with financial information
            - Show empathy for billing frustrations
            - Be transparent about timelines and limitations

            For technical or order issues, acknowledge and indicate transfer.
            """
        ).rstrip("\n")

# ============================================================================
# Block 12 (chapter listing #12)
# ============================================================================

# Escalation agent - handles human handoff.


class EscalationAgent(BaseAgent):
    """
    Handles escalations to human agents.
    Prepares context and manages the handoff process.
    """

    @property
    def agent_type(self) -> AgentType:
        return AgentType.ESCALATION

    @property
    def system_prompt(self) -> str:
        return (
            "You are the escalation specialist. Your role is to:\n"
            "\n"
            "1. Acknowledge the customer's need for human assistance\n"
            "2. Gather any final context needed for the human agent\n"
            "3. Set expectations about wait times\n"
            "4. Execute a smooth handoff\n"
            "\n"
            "You have access to these tools:\n"
            "- create_escalation_ticket: Create a prioritized ticket"
            " for human review\n"
            "- find_available_agent: Check human agent availability\n"
            "- transfer_conversation: Execute the transfer to a human\n"
            "\n"
            "ESCALATION PROCESS:\n"
            "\n"
            "1. ACKNOWLEDGE\n"
            "   - Thank the customer for their patience\n"
            "   - Validate their concern/frustration\n"
            "   - Confirm they'll be connected to a human\n"
            "\n"
            "2. PREPARE\n"
            "   - Summarize the issue briefly\n"
            "   - Note any attempted resolutions\n"
            "   - Capture any additional context needed\n"
            "\n"
            "3. SET EXPECTATIONS\n"
            "   - Provide estimated wait time\n"
            "   - Explain what happens next\n"
            "   - Offer callback option if wait is long\n"
            "\n"
            "4. EXECUTE\n"
            "   - Create escalation ticket with full context\n"
            "   - Transfer to appropriate human agent\n"
            "   - Ensure warm handoff (context preserved)\n"
            "\n"
            "PRIORITIZATION:\n"
            "- Critical: VIP customers, safety concerns, legal issues\n"
            "- High: Very frustrated customers, repeated failures\n"
            "- Normal: Standard escalation requests\n"
            "\n"
            "Always be empathetic and professional. The customer has"
            " likely already had a frustrating experience."
        )

    async def process_message(
        self, conversation: Conversation, user_message: str
    ) -> AgentResponse:
        """Handle escalation process."""
        # Always escalate from this agent
        escalation = EscalationRequest.create(
            conversation=conversation,
            reason=conversation.context.escalation_reason
            or "Customer requested human agent",
            required_skills=self._determine_required_skills(conversation),
        )

        # Determine wait time based on priority and availability
        wait_time = await self._estimate_wait_time(escalation.priority)

        message = f"""I understand you'd like to speak with a human agent. I'm arranging that now.

Based on your {conversation.context.customer.tier} account status, you're being placed in our priority queue.

Estimated wait time: {wait_time}

A specialist will have full context of our conversation and the steps we've already tried. Is there anything else you'd like me to note for them?"""

        return AgentResponse(
            message=message,
            agent_type=self.agent_type,
            should_escalate=True,
            escalation_reason=escalation.reason,
            metadata={"escalation_id": escalation.escalation_id},
        )

    def _determine_required_skills(
        self, conversation: Conversation
    ) -> list[str]:
        """Determine what skills the human agent needs."""
        skills = []

        intent = conversation.context.intent
        if intent:
            if "billing" in intent or "payment" in intent:
                skills.append("billing_specialist")
            if "technical" in intent:
                skills.append("technical_specialist")
            if "order" in intent:
                skills.append("order_specialist")

        if conversation.context.customer.tier == "enterprise":
            skills.append("enterprise_support")

        if conversation.context.sentiment_score < -0.5:
            skills.append("de_escalation")

        return skills or ["general_support"]

    async def _estimate_wait_time(self, priority: Priority) -> str:
        """Estimate wait time based on priority."""
        # In production, this would check actual queue depth
        wait_times = {
            Priority.CRITICAL: "Less than 1 minute",
            Priority.URGENT: "1-2 minutes",
            Priority.HIGH: "2-5 minutes",
            Priority.NORMAL: "5-10 minutes",
            Priority.LOW: "10-15 minutes",
        }
        return wait_times.get(priority, "5-10 minutes")

# ============================================================================
# Block 13 (chapter listing #13)
# ============================================================================

# Conversation manager - orchestrates the multi-agent system.

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Metrics collected during a conversation."""

    start_time: datetime
    end_time: Optional[datetime] = None
    message_count: int = 0
    agent_transfers: int = 0
    tool_calls: int = 0
    escalated: bool = False
    resolved: bool = False
    resolution_time_seconds: Optional[float] = None
    agents_involved: set[AgentType] = field(default_factory=set)


class InMemoryConversationStore:
    """Demo persistence boundary for live conversations.

    Production deployments should provide a Redis/Postgres-backed store
    and treat ``ConversationManager.active_conversations`` as a bounded
    hot cache for currently active turns.
    """

    def __init__(self, max_records: int = 100_000) -> None:
        if max_records < 1:
            raise ValueError("max_records must be >= 1")
        self._records: "OrderedDict[str, Conversation]" = OrderedDict()
        self._max_records = max_records
        self._lock = asyncio.Lock()

    async def save(self, conversation: Conversation) -> None:
        async with self._lock:
            self._records[conversation.conversation_id] = conversation
            self._records.move_to_end(conversation.conversation_id)
            while len(self._records) > self._max_records:
                self._records.popitem(last=False)

    async def load(self, conversation_id: str) -> Optional[Conversation]:
        async with self._lock:
            conversation = self._records.get(conversation_id)
            if conversation is not None:
                self._records.move_to_end(conversation_id)
            return conversation


class ConversationManager:
    """
    Manages customer conversations across multiple agents.
    Handles routing, context preservation, and handoffs.
    """

    # Default cap on in-memory active conversations. Older entries are
    # evicted in LRU order. For production, externalize state to Redis
    # (Chapter 2) and treat this dict as a hot cache.
    MAX_ACTIVE_CONVERSATIONS = 10_000

    def __init__(
        self,
        config: PlatformConfig,
        agents: dict[AgentType, BaseAgent],
        max_active_conversations: int = MAX_ACTIVE_CONVERSATIONS,
        conversation_store: Optional[InMemoryConversationStore] = None,
    ) -> None:
        if max_active_conversations < 1:
            raise ValueError("max_active_conversations must be >= 1")
        self.config = config
        self.agents = agents
        self.max_active_conversations = max_active_conversations
        self.conversation_store = (
            conversation_store or InMemoryConversationStore()
        )
        self.active_conversations: "OrderedDict[str, Conversation]" = (
            OrderedDict()
        )
        self.conversation_metrics: "OrderedDict[str, ConversationMetrics]" = (
            OrderedDict()
        )
        self._conversation_locks: "OrderedDict[str, asyncio.Lock]" = (
            OrderedDict()
        )
        self.shutting_down = False

    def _record_active(
        self,
        conv_id: str,
        conversation: Conversation,
        metrics: ConversationMetrics,
    ) -> None:
        """Insert (or refresh LRU position for) a conversation."""
        self.active_conversations[conv_id] = conversation
        self.active_conversations.move_to_end(conv_id)
        self.conversation_metrics[conv_id] = metrics
        self.conversation_metrics.move_to_end(conv_id)
        self._conversation_locks.setdefault(conv_id, asyncio.Lock())
        self._conversation_locks.move_to_end(conv_id)
        while len(self.active_conversations) > self.max_active_conversations:
            evicted_id = None
            for candidate_id in list(self.active_conversations.keys()):
                if candidate_id == conv_id:
                    continue
                lock = self._conversation_locks.get(candidate_id)
                if lock is not None and lock.locked():
                    self.active_conversations.move_to_end(candidate_id)
                    self.conversation_metrics.move_to_end(candidate_id)
                    self._conversation_locks.move_to_end(candidate_id)
                    continue
                evicted_id = candidate_id
                break

            if evicted_id is None:
                self.active_conversations.pop(conv_id, None)
                self.conversation_metrics.pop(conv_id, None)
                self._conversation_locks.pop(conv_id, None)
                raise RuntimeError(
                    "active conversation cap reached; all existing "
                    "conversations are currently in flight"
                )

            self.active_conversations.pop(evicted_id, None)
            self.conversation_metrics.pop(evicted_id, None)
            self._conversation_locks.pop(evicted_id, None)

    async def start_conversation(
        self, customer: Customer, channel: ConversationChannel
    ) -> Conversation:
        """Start a new conversation."""
        if self.shutting_down:
            raise RuntimeError("ConversationManager is shutting down")
        conversation = Conversation.create(customer, channel)
        self._record_active(
            conversation.conversation_id,
            conversation,
            ConversationMetrics(start_time=datetime.now(timezone.utc)),
        )
        await self.conversation_store.save(conversation)

        logger.info(
            f"Started conversation {conversation.conversation_id} "
            f"for customer {customer.customer_id}"
        )

        return conversation

    async def process_message(
        self, conversation_id: str, message: str
    ) -> AgentResponse:
        """Process an incoming customer message."""
        if self.shutting_down:
            raise RuntimeError("ConversationManager is shutting down")
        # ``setdefault`` is atomic under the single-threaded asyncio
        # event loop: two coroutines reaching this path concurrently
        # would otherwise both fall through the ``get is None`` branch
        # and install two distinct locks, defeating the per-conversation
        # serialization. ``setdefault`` reuses the existing lock object
        # when one is already present.
        lock = self._conversation_locks.setdefault(
            conversation_id, asyncio.Lock()
        )
        self._conversation_locks.move_to_end(conversation_id)
        async with lock:
            if conversation_id not in self.active_conversations:
                stored = await self.conversation_store.load(conversation_id)
                if stored is None:
                    raise ValueError(
                        f"Conversation {conversation_id} not found"
                    )
                self._record_active(
                    conversation_id,
                    stored,
                    self.conversation_metrics.get(
                        conversation_id,
                        ConversationMetrics(start_time=stored.created_at),
                    ),
                )

            response = await self._process_message_locked(
                conversation_id, message
            )
            await self.conversation_store.save(
                self.active_conversations[conversation_id]
            )
            return response

    async def drain_active_conversations(
        self, drain_timeout_seconds: float = 30.0
    ) -> None:
        """Wait for currently locked conversations to finish.

        ``drain_timeout_seconds`` caps how long we will block waiting
        for outstanding per-conversation locks. If the deadline elapses
        the still-locked conversations are logged so a stuck handler is
        visible to ops, and the method returns rather than blocking
        shutdown forever.
        """
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max(0.0, drain_timeout_seconds)
        while any(lock.locked() for lock in self._conversation_locks.values()):
            if loop.time() >= deadline:
                stuck = [
                    cid
                    for cid, lock in self._conversation_locks.items()
                    if lock.locked()
                ]
                logger.warning(
                    "drain_active_conversations timeout after %.1fs; "
                    "%d conversation(s) still locked: %s",
                    drain_timeout_seconds,
                    len(stuck),
                    stuck[:10],
                )
                return
            await asyncio.sleep(0.05)

    async def _process_message_locked(
        self, conversation_id: str, message: str
    ) -> AgentResponse:
        """Process a message while holding the per-conversation lock."""
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            raise ValueError(f"Conversation {conversation_id} not found")
        # Mark recently used so the LRU cap evicts cold conversations first.
        self.active_conversations.move_to_end(conversation_id)
        self.conversation_metrics.move_to_end(conversation_id)

        metrics = self.conversation_metrics[conversation_id]
        metrics.message_count += 1

        # Add customer message to history
        conversation.add_message("customer", message)

        # Determine which agent should handle this
        current_agent = self._get_current_agent(conversation)

        # Process through agent
        response = await current_agent.process_message(conversation, message)

        # Record metrics
        metrics.tool_calls += len(response.tool_calls)
        metrics.agents_involved.add(current_agent.agent_type)

        # Add agent response to history
        conversation.add_message(
            "agent", response.message, response.agent_type
        )

        # Handle transfers
        if response.should_transfer and response.transfer_to:
            await self._handle_transfer(
                conversation, response.transfer_to, response.transfer_reason
            )
            metrics.agent_transfers += 1

        # Handle escalations
        if response.should_escalate:
            await self._handle_escalation(
                conversation, response.escalation_reason
            )
            metrics.escalated = True

        return response

    def _get_current_agent(self, conversation: Conversation) -> BaseAgent:
        """Get the agent that should handle the current message."""
        current = conversation.context.current_agent

        # Default to triage for new conversations
        if not current:
            current = AgentType.TRIAGE
            conversation.context.current_agent = current

        return self.agents[current]

    async def _handle_transfer(
        self, conversation: Conversation, target: AgentType, reason: str
    ) -> None:
        """Handle agent-to-agent transfer."""
        previous = conversation.context.current_agent

        # Record the transfer
        if previous:
            conversation.context.previous_agents.append(previous)
        conversation.context.current_agent = target

        logger.info(
            f"Conversation {conversation.conversation_id} "
            f"transferred from {previous} to {target}: {reason}"
        )

    async def _handle_escalation(
        self, conversation: Conversation, reason: str
    ) -> None:
        """Handle escalation to human agent."""
        conversation.status = ConversationStatus.ESCALATED
        conversation.context.escalation_reason = reason

        # Transfer to escalation agent if not already there
        if conversation.context.current_agent != AgentType.ESCALATION:
            await self._handle_transfer(
                conversation, AgentType.ESCALATION, reason
            )

        logger.info(
            f"Conversation {conversation.conversation_id} escalated: {reason}"
        )

    async def resolve_conversation(
        self, conversation_id: str, resolution: str = "resolved"
    ) -> None:
        """Mark a conversation as resolved."""
        conversation = self.active_conversations.get(conversation_id)
        if not conversation:
            return

        conversation.status = ConversationStatus.RESOLVED
        conversation.resolved_at = datetime.now(timezone.utc)

        metrics = self.conversation_metrics[conversation_id]
        metrics.end_time = datetime.now(timezone.utc)
        metrics.resolved = True
        metrics.resolution_time_seconds = (
            metrics.end_time - metrics.start_time
        ).total_seconds()

        logger.info(
            f"Conversation {conversation_id} resolved in "
            f"{metrics.resolution_time_seconds:.1f}s"
        )
        await self.conversation_store.save(conversation)

    def get_conversation_summary(self, conversation_id: str) -> dict[str, Any]:
        """Get a summary of a conversation for reporting."""
        conversation = self.active_conversations.get(conversation_id)
        metrics = self.conversation_metrics.get(conversation_id)

        if not conversation or not metrics:
            return {}

        return {
            "conversation_id": conversation_id,
            "customer_id": conversation.context.customer.customer_id,
            "customer_tier": conversation.context.customer.tier,
            "channel": conversation.channel.value,
            "status": conversation.status.value,
            "intent": conversation.context.intent,
            "sentiment": conversation.context.sentiment_score,
            "message_count": metrics.message_count,
            "agent_transfers": metrics.agent_transfers,
            "tool_calls": metrics.tool_calls,
            "agents_involved": [
                a.value for a in sorted(
                    metrics.agents_involved, key=lambda agent: agent.value
                )
            ],
            "escalated": metrics.escalated,
            "resolved": metrics.resolved,
            "duration_seconds": metrics.resolution_time_seconds,
            "started_at": metrics.start_time.isoformat(),
            "ended_at": (
                metrics.end_time.isoformat() if metrics.end_time else None
            ),
        }

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

# Quality assurance system for customer service conversations.

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol
import json
import logging
import textwrap

logger = logging.getLogger(__name__)


class _QualityLLMMessages(Protocol):
    async def create(self, **kwargs: Any) -> Any:
        ...


class _QualityLLMClient(Protocol):
    messages: _QualityLLMMessages


@dataclass
class QualityScore:
    """Quality assessment for a conversation."""

    conversation_id: str
    overall_score: float  # 0-100
    dimensions: dict[str, float] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    reviewed_by: Optional[str] = None  # Human reviewer ID if applicable
    reviewed_at: Optional[datetime] = None


class QualityAssessment:
    """
    Automated quality assessment for conversations.
    Evaluates multiple dimensions and flags issues.
    """

    def __init__(self, config: QualityConfig, llm_client: _QualityLLMClient) -> None:
        self.config = config
        self.llm_client = llm_client
        # Counter for malformed-JSON assessment responses so operators can
        # alert on a rising rate instead of only inspecting logs.
        self._malformed_json_count = 0

    async def assess_conversation(
        self, conversation: Conversation, metrics: ConversationMetrics
    ) -> QualityScore:
        """Perform comprehensive quality assessment."""
        scores = {}
        flags = []
        recommendations = []

        # Dimension 1: Resolution effectiveness
        resolution_score = self._assess_resolution(conversation, metrics)
        scores["resolution"] = resolution_score

        # Dimension 2: Response quality
        response_score = await self._assess_response_quality(conversation)
        scores["response_quality"] = response_score

        # Dimension 3: Efficiency
        efficiency_score = self._assess_efficiency(metrics)
        scores["efficiency"] = efficiency_score

        # Dimension 4: Customer sentiment trajectory
        sentiment_score = self._assess_sentiment_trajectory(conversation)
        scores["sentiment"] = sentiment_score

        # Dimension 5: Policy compliance
        compliance_score = self._assess_compliance(conversation)
        scores["compliance"] = compliance_score

        # Generate flags
        if resolution_score < 50:
            flags.append("low_resolution_effectiveness")
            recommendations.append(
                "Review conversation for missed resolution opportunities"
            )

        if efficiency_score < 50:
            flags.append("efficiency_concern")
            recommendations.append(
                "Analyze for unnecessary transfers or tool failures"
            )

        if sentiment_score < 50:
            flags.append("sentiment_decline")
            recommendations.append("Review for customer frustration points")

        if compliance_score < 80:
            flags.append("compliance_review_needed")
            recommendations.append("Manual review required for compliance")

        # Example launch weights. Calibrate these against human QA labels
        # before using the score for routing, compensation, or compliance.
        weights = getattr(self.config, "quality_weights", None) or {
            "resolution": 0.30,
            "response_quality": 0.25,
            "efficiency": 0.15,
            "sentiment": 0.15,
            "compliance": 0.15,
        }

        overall = sum(scores[dim] * weights[dim] for dim in scores)

        return QualityScore(
            conversation_id=conversation.conversation_id,
            overall_score=overall,
            dimensions=scores,
            flags=flags,
            recommendations=recommendations,
        )

    def _assess_resolution(
        self, conversation: Conversation, metrics: ConversationMetrics
    ) -> float:
        """Assess whether the customer's issue was resolved."""
        score = 100.0

        # Penalize escalations
        if metrics.escalated:
            score -= 30

        # Penalize unresolved conversations
        if not metrics.resolved:
            score -= 50

        # Penalize excessive transfers
        if metrics.agent_transfers > 2:
            score -= (metrics.agent_transfers - 2) * 10

        return max(0, score)

    async def _assess_response_quality(
        self, conversation: Conversation
    ) -> float:
        """Use LLM to assess response quality."""
        # Sample recent agent responses
        agent_messages = [
            m for m in conversation.messages if m.role == "agent"
        ]
        if not agent_messages:
            return 50.0

        sample = agent_messages[-3:]  # Last 3 responses

        responses_json = json.dumps(
            [{"content": m.content} for m in sample], indent=2
        )
        assessment_prompt = textwrap.dedent("""\
            Evaluate these customer service responses for quality.

            Responses:
            __RESPONSES__

            Score each dimension 0-100:
            1. Clarity: Is the response clear and easy to understand?
            2. Helpfulness: Does it address the customer's need?
            3. Professionalism: Is the tone appropriate?
            4. Accuracy: Does it provide correct information?

            Return JSON: {"clarity": X, "helpfulness": X, "professionalism": X, "accuracy": X}""").replace("__RESPONSES__", responses_json)

        try:
            for attempt in range(2):
                try:
                    response = await asyncio.wait_for(
                        self.llm_client.messages.create(
                            model=self.config.assessment_model,
                            max_tokens=self.config.assessment_max_tokens,
                            messages=[
                                {"role": "user", "content": assessment_prompt}
                            ],
                        ),
                        timeout=self.config.assessment_timeout_seconds,
                    )
                    break
                except (
                    anthropic.APITimeoutError,
                    anthropic.RateLimitError,
                    anthropic.APIConnectionError,
                    asyncio.TimeoutError,
                    TimeoutError,
                    ConnectionError,
                    OSError,
                ) as e:
                    if attempt == 1:
                        raise
                    logger.warning(
                        "Response quality assessment retrying: %s", e
                    )
                    await asyncio.sleep(0.5 * (2**attempt))

            scores = json.loads(response.content[0].text)
            required = {"clarity", "helpfulness", "professionalism", "accuracy"}
            if not required.issubset(scores):
                raise KeyError(f"missing score keys: {required - set(scores)}")
            numeric_scores = [float(scores[key]) for key in required]
            return sum(numeric_scores) / len(numeric_scores)

        except json.JSONDecodeError as e:
            self._malformed_json_count += 1
            logger.warning("Response quality assessment returned non-JSON: %s", e)
            return 50.0
        except (KeyError, TypeError, ValueError) as e:
            logger.warning("Response quality assessment malformed: %s", e)
            return 50.0
        except (
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            asyncio.TimeoutError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as e:
            logger.warning("Response quality assessment provider failed: %s", e)
            return 50.0

    def _assess_efficiency(self, metrics: ConversationMetrics) -> float:
        """Assess conversation efficiency."""
        score = 100.0

        # Penalize long conversations
        if metrics.resolution_time_seconds:
            if metrics.resolution_time_seconds > 600:  # 10 minutes
                score -= 20
            if metrics.resolution_time_seconds > 900:  # 15 minutes
                score -= 20

        # Penalize excessive messages; clamp the penalty so a single
        # runaway transcript cannot drive the composite score below the
        # other dimensions' contribution floor.
        if metrics.message_count > 15:
            penalty = min(100, max(0, (metrics.message_count - 15) * 2))
            score -= penalty

        # Penalize failed tool calls (would need to track this)

        return max(0, score)

    def _assess_sentiment_trajectory(
        self, conversation: Conversation
    ) -> float:
        """Assess how customer sentiment changed during conversation."""
        # Ideal: sentiment improves or stays positive
        # For this example, use the final sentiment score
        sentiment = conversation.context.sentiment_score

        # Convert -1 to 1 scale to 0-100
        return (sentiment + 1) * 50

    def _assess_compliance(self, conversation: Conversation) -> float:
        """Check for compliance violations."""
        score = 100.0

        # Check for sensitive data exposure (simplified).
        # Bare \d{13,16} catches order IDs, tracking numbers, and
        # phone numbers as false positives; gate the penalty on a
        # Luhn check so we only flag plausible card numbers.
        for message in conversation.messages:
            content_lower = message.content.lower()

            import re

            for candidate in re.findall(r"\b\d{13,16}\b", message.content):
                if self._looks_like_pan(candidate):
                    score -= 50
                    break

            # Policy-configured phrases. The default list is illustrative;
            # compliance teams should provide jurisdiction/workflow-specific
            # terms and penalties.
            prohibited = getattr(
                self.config,
                "prohibited_phrases",
                {"i promise": 10, "definitely": 10, "always": 10, "never fails": 10},
            )
            for phrase, penalty in prohibited.items():
                if phrase in content_lower:
                    score -= penalty

        return max(0, score)

    @staticmethod
    def _looks_like_pan(digits: str) -> bool:
        """Luhn-mod-10 check; weeds out non-PAN numeric strings."""
        total = 0
        for i, ch in enumerate(reversed(digits)):
            d = int(ch)
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            total += d
        return total % 10 == 0


class FeedbackCollector:
    """
    Collects and processes customer feedback.
    """

    def __init__(self, storage_client: "FeedbackStorage") -> None:
        self.storage: "FeedbackStorage" = storage_client

    async def collect_csat(
        self, conversation_id: str, score: int, comment: Optional[str] = None
    ) -> None:
        """Collect customer satisfaction score."""
        feedback = {
            "conversation_id": conversation_id,
            "type": "csat",
            "score": score,  # 1-5
            "comment": comment,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.storage.store_feedback(feedback)

        # Alert on low scores
        if score <= 2:
            await self._trigger_low_score_alert(
                conversation_id, score, comment
            )

    async def collect_resolution_feedback(
        self,
        conversation_id: str,
        resolved: bool,
        reason: Optional[str] = None,
    ) -> None:
        """Collect feedback on whether issue was resolved."""
        feedback = {
            "conversation_id": conversation_id,
            "type": "resolution",
            "resolved": resolved,
            "reason": reason,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

        await self.storage.store_feedback(feedback)

    async def _trigger_low_score_alert(
        self, conversation_id: str, score: int, comment: Optional[str]
    ) -> None:
        """Alert on low satisfaction scores."""
        logger.warning(
            f"Low CSAT score ({score}) for conversation {conversation_id}"
        )
        # In production: send to alerting system, queue for review

# ============================================================================
# Block 15 (chapter listing #15)
# ============================================================================

# Metrics collection and reporting for customer service platform.
#
# Includes structural Protocols for the two storage backends the
# collectors talk to. These document the actual surface used; any
# implementation that quacks the same way (in-memory ring buffer,
# Prometheus pushgateway, CloudWatch, OTel Collector, custom feedback
# sink) can be passed in.

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Protocol, runtime_checkable
import asyncio


@runtime_checkable
class MetricsStorage(Protocol):
    """Sink for time-series metric events."""

    async def record(self, metric: dict[str, Any]) -> None: ...
    async def flush(self) -> None: ...


@runtime_checkable
class FeedbackStorage(Protocol):
    """Sink for customer-feedback records (CSAT, resolution, NPS)."""

    async def store_feedback(self, feedback: dict[str, Any]) -> None: ...


@dataclass
class PlatformMetrics:
    """Aggregated platform metrics."""

    period_start: datetime
    period_end: datetime

    # Volume metrics
    total_conversations: int = 0
    conversations_by_channel: dict[str, int] = field(default_factory=dict)
    conversations_by_intent: dict[str, int] = field(default_factory=dict)

    # Resolution metrics
    automated_resolutions: int = 0
    escalated_conversations: int = 0
    abandoned_conversations: int = 0

    # Time metrics
    avg_first_response_ms: float = 0.0
    p95_first_response_ms: float = 0.0
    avg_resolution_time_seconds: float = 0.0
    p95_resolution_time_seconds: float = 0.0
    avg_handle_time_seconds: float = 0.0

    # Quality metrics
    avg_csat_score: float = 0.0
    avg_quality_score: float = 0.0

    # Agent metrics
    transfers_by_agent: dict[str, int] = field(default_factory=dict)
    tool_usage: dict[str, int] = field(default_factory=dict)
    tool_failures: dict[str, int] = field(default_factory=dict)

    @property
    def automation_rate(self) -> float:
        """Percentage of conversations resolved without human."""
        if self.total_conversations == 0:
            return 0.0
        return (self.automated_resolutions / self.total_conversations) * 100

    @property
    def escalation_rate(self) -> float:
        """Percentage of conversations escalated to human."""
        if self.total_conversations == 0:
            return 0.0
        return (self.escalated_conversations / self.total_conversations) * 100


class MetricsCollector:
    """
    Collects and aggregates metrics from the customer service platform.

    Keeps the last 24 hourly buckets in memory; older buckets are
    flushed to long-term storage and dropped. Each timing sample queue
    is bounded to the last 10,000 measurements (running percentiles
    rather than full histograms). For production, swap to Prometheus
    / OpenTelemetry exporters (Book 2, Chapter 6).
    """

    MAX_HOURLY_BUCKETS = 24
    MAX_TIMING_SAMPLES = 10_000
    MAX_TIMING_KEYS = 1_000
    MAX_COUNTER_KEYS_PER_BUCKET = 1_000
    MAX_REPORT_HOURS = 24 * 31

    def __init__(self, storage_client: "MetricsStorage") -> None:
        self.storage: "MetricsStorage" = storage_client
        self._current_metrics: "OrderedDict[str, OrderedDict[str, int]]" = (
            OrderedDict()
        )
        # Use bounded deques instead of unbounded lists.
        self._timing_samples = defaultdict(
            lambda: deque(maxlen=self.MAX_TIMING_SAMPLES)
        )
        # Track the most recently observed hour bucket so eviction only
        # runs when the hour rolls over rather than on every hot-path
        # record_* call.
        self._last_hour_key: Optional[str] = None

    def _metric_component(self, value: str) -> str:
        safe = "".join(
            ch if ch.isalnum() or ch in ("_", "-", ".") else "_"
            for ch in str(value)
        )
        return safe[:80] or "unknown"

    def _counter_bucket(self, hour_key: str) -> "OrderedDict[str, int]":
        bucket = self._current_metrics.get(hour_key)
        if bucket is None:
            bucket = OrderedDict()
            self._current_metrics[hour_key] = bucket
        self._current_metrics.move_to_end(hour_key)
        return bucket

    def _increment_counter(
        self, hour_key: str, metric_name: str, by: int = 1
    ) -> None:
        bucket = self._counter_bucket(hour_key)
        if metric_name not in bucket and len(bucket) >= self.MAX_COUNTER_KEYS_PER_BUCKET:
            bucket.popitem(last=False)
        bucket[metric_name] = bucket.get(metric_name, 0) + by
        bucket.move_to_end(metric_name)

    def _evict_old_buckets(self) -> None:
        """Drop the oldest hourly buckets once we exceed the cap.

        Only does work when the current hour bucket differs from the
        last one we saw. ``record_*`` methods call into this on every
        event, so without the rollover guard we would sort the bucket
        keys on every hot-path call even though eviction can only
        change state once per hour.
        """
        if not self._current_metrics:
            return
        current_hour = next(reversed(self._current_metrics))
        if current_hour == self._last_hour_key:
            return
        self._last_hour_key = current_hour
        if len(self._current_metrics) > self.MAX_HOURLY_BUCKETS:
            # Buckets are keyed by ISO hour string, so lexicographic
            # ordering matches chronological ordering.
            stale = sorted(self._current_metrics.keys())[
                : -self.MAX_HOURLY_BUCKETS
            ]
            for k in stale:
                self._current_metrics.pop(k, None)
                # Drop the matching timing samples too.
                for tk in list(self._timing_samples):
                    if tk.startswith(k):
                        self._timing_samples.pop(tk, None)

    def _evict_timing_keys(self) -> None:
        """Cap timing metric cardinality for arbitrary tool names."""
        if len(self._timing_samples) <= self.MAX_TIMING_KEYS:
            return
        stale = sorted(self._timing_samples.keys())[
            : -self.MAX_TIMING_KEYS
        ]
        for key in stale:
            self._timing_samples.pop(key, None)

    async def record_conversation_start(self, conversation: Conversation) -> None:
        """Record a new conversation."""
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")

        self._increment_counter(hour_key, "total_conversations")
        self._increment_counter(
            hour_key, f"channel_{conversation.channel.value}"
        )
        # Enforce the bucket retention cap on every new-hour write.
        self._evict_old_buckets()

    async def record_first_response(
        self, conversation_id: str, latency_ms: float
    ) -> None:
        """Record first response latency."""
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        self._timing_samples[f"{hour_key}_first_response"].append(latency_ms)
        self._evict_old_buckets()
        self._evict_timing_keys()

    async def record_conversation_end(
        self, conversation: Conversation, metrics: ConversationMetrics
    ) -> None:
        """Record conversation completion."""
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")

        if metrics.resolved and not metrics.escalated:
            self._increment_counter(hour_key, "automated_resolutions")

        if metrics.escalated:
            self._increment_counter(hour_key, "escalations")

        if conversation.context.intent:
            intent = self._metric_component(conversation.context.intent)
            self._increment_counter(hour_key, f"intent_{intent}")

        if metrics.resolution_time_seconds:
            self._timing_samples[f"{hour_key}_resolution_time"].append(
                metrics.resolution_time_seconds
            )
        self._evict_old_buckets()
        self._evict_timing_keys()

        # Record agent involvement
        for agent in sorted(metrics.agents_involved, key=lambda item: item.value):
            agent_name = self._metric_component(agent.value)
            self._increment_counter(hour_key, f"agent_{agent_name}_involved")

    async def record_tool_usage(
        self, tool_name: str, success: bool, latency_ms: float
    ) -> None:
        """Record tool execution."""
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")

        tool = self._metric_component(tool_name)
        self._increment_counter(hour_key, f"tool_{tool}_calls")
        if success:
            self._increment_counter(hour_key, f"tool_{tool}_success")
        else:
            self._increment_counter(hour_key, f"tool_{tool}_failures")

        self._timing_samples[f"{hour_key}_tool_{tool}_latency"].append(
            latency_ms
        )
        self._evict_old_buckets()
        self._evict_timing_keys()

    async def get_metrics(
        self, start: datetime, end: datetime
    ) -> PlatformMetrics:
        """Get aggregated metrics for a time period."""
        if end <= start:
            raise ValueError("end must be after start")
        report_hours = (end - start).total_seconds() / 3600
        if report_hours > self.MAX_REPORT_HOURS:
            raise ValueError(
                f"metrics report window exceeds {self.MAX_REPORT_HOURS} hours"
            )
        metrics = PlatformMetrics(period_start=start, period_end=end)

        # Aggregate hourly data
        current = start
        while current < end:
            hour_key = current.strftime("%Y-%m-%d-%H")
            hour_data = self._current_metrics.get(hour_key, {})

            metrics.total_conversations += hour_data.get(
                "total_conversations", 0
            )
            metrics.automated_resolutions += hour_data.get(
                "automated_resolutions", 0
            )
            metrics.escalated_conversations += hour_data.get("escalations", 0)

            # Aggregate by channel
            for key, value in hour_data.items():
                if key.startswith("channel_"):
                    channel = key.replace("channel_", "")
                    metrics.conversations_by_channel[channel] = (
                        metrics.conversations_by_channel.get(channel, 0)
                        + value
                    )
                elif key.startswith("intent_"):
                    intent = key.replace("intent_", "")
                    metrics.conversations_by_intent[intent] = (
                        metrics.conversations_by_intent.get(intent, 0) + value
                    )
                elif key.startswith("tool_") and key.endswith("_calls"):
                    tool = key.replace("tool_", "").replace("_calls", "")
                    metrics.tool_usage[tool] = (
                        metrics.tool_usage.get(tool, 0) + value
                    )
                elif key.startswith("tool_") and key.endswith("_failures"):
                    tool = key.replace("tool_", "").replace(
                        "_failures", ""
                    )
                    metrics.tool_failures[tool] = (
                        metrics.tool_failures.get(tool, 0) + value
                    )

            current += timedelta(hours=1)

        # Calculate averages from timing samples
        all_first_response = []
        all_resolution_time = []

        current = start
        while current < end:
            hour_key = current.strftime("%Y-%m-%d-%H")
            all_first_response.extend(
                self._timing_samples.get(f"{hour_key}_first_response", [])
            )
            all_resolution_time.extend(
                self._timing_samples.get(f"{hour_key}_resolution_time", [])
            )
            current += timedelta(hours=1)

        if all_first_response:
            metrics.avg_first_response_ms = sum(all_first_response) / len(
                all_first_response
            )
            metrics.p95_first_response_ms = self._percentile(
                all_first_response, 95
            )

        if all_resolution_time:
            metrics.avg_resolution_time_seconds = sum(
                all_resolution_time
            ) / len(all_resolution_time)
            metrics.p95_resolution_time_seconds = self._percentile(
                all_resolution_time, 95
            )

        return metrics

    def _percentile(self, values: list[float], percentile: int) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(
            len(ordered) - 1,
            max(0, round((percentile / 100) * (len(ordered) - 1))),
        )
        return ordered[index]

    async def get_real_time_stats(self) -> dict[str, Any]:
        """Get near-real-time statistics for dashboards.

        Returned values reflect the current hour bucket; dashboards
        polling every 5-30 seconds will see p95 staleness under
        1 minute, which is the operational meaning of "near-real-time"
        here. This is not streaming - sub-second freshness requires
        a different pipeline (e.g. Kafka + a stream processor).
        """
        hour_key = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H")
        current = self._current_metrics.get(hour_key, {})

        return {
            "current_hour": hour_key,
            "conversations_this_hour": current.get("total_conversations", 0),
            "escalations_this_hour": current.get("escalations", 0),
            "automation_rate": (
                current.get("automated_resolutions", 0)
                / max(current.get("total_conversations", 1), 1)
                * 100
            ),
        }


class MetricsDashboard:
    """
    Dashboard for monitoring platform health.
    """

    def __init__(self, collector: MetricsCollector, config: QualityConfig) -> None:
        self.collector = collector
        self.config = config

    async def generate_daily_report(self, date: datetime) -> dict[str, Any]:
        """Generate daily metrics report."""
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        metrics = await self.collector.get_metrics(start, end)

        # Calculate SLA compliance
        sla_compliance = {
            "first_response": metrics.avg_first_response_ms
            < self.config.max_first_response_ms,
            "resolution_rate": (
                metrics.automation_rate >= self.config.min_automation_rate
            ),
            "escalation_rate": (
                metrics.escalation_rate <= self.config.max_escalation_rate
            ),
            "handle_time": metrics.avg_resolution_time_seconds
            < self.config.max_handle_time_seconds,
        }

        return {
            "date": date.strftime("%Y-%m-%d"),
            "summary": {
                "total_conversations": metrics.total_conversations,
                "automation_rate": f"{metrics.automation_rate:.1f}%",
                "escalation_rate": f"{metrics.escalation_rate:.1f}%",
                "avg_first_response": f"{metrics.avg_first_response_ms:.0f}ms",
                "avg_resolution_time": f"{metrics.avg_resolution_time_seconds:.0f}s",
            },
            "sla_compliance": sla_compliance,
            "channels": metrics.conversations_by_channel,
            "intents": metrics.conversations_by_intent,
            "tool_usage": metrics.tool_usage,
            "recommendations": self._generate_recommendations(
                metrics, sla_compliance
            ),
        }

    def _generate_recommendations(
        self, metrics: PlatformMetrics, sla_compliance: dict[str, bool]
    ) -> list[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []

        if not sla_compliance["first_response"]:
            recommendations.append(
                "First response time exceeds SLA. Consider scaling triage capacity "
                "or optimizing intent classification."
            )

        if not sla_compliance["resolution_rate"]:
            recommendations.append(
                "Automation rate below target. Review escalation reasons and "
                "expand agent capabilities for common escalation triggers."
            )

        if not sla_compliance["escalation_rate"]:
            recommendations.append(
                "Escalation rate above target. Analyze escalation patterns and "
                "enhance agent training for common escalation scenarios."
            )

        # Check for tool failures: flag any tool whose recent failure
        # rate exceeds ten percent so operators can investigate. The
        # previous version had a placeholder loop that simply iterated
        # without producing any recommendation.
        for tool, calls in metrics.tool_usage.items():
            tool_failures = metrics.tool_failures.get(tool, 0)
            if calls and tool_failures / calls > 0.1:
                recommendations.append(
                    f"Review {tool} (failure rate "
                    f"{tool_failures / calls:.1%})"
                )

        return recommendations

# ============================================================================
# Block 16 (chapter listing #16)
# ============================================================================

# Complete customer service platform integration.

import asyncio
import httpx
import logging
import os
import signal


from contextlib import asynccontextmanager
from typing import AsyncIterator


@asynccontextmanager
async def create_platform() -> AsyncIterator[
    tuple["ConversationManager", "MetricsCollector"]
]:
    """Initialize the complete platform as an async context manager.

    Yields ``(manager, metrics_collector)``. The httpx clients live for
    the duration of the ``async with`` block; the manager's tools hold
    references to them and will fail if used after the block exits.
    Earlier drafts of this function returned the tuple from inside the
    ``async with`` -- which closed the clients immediately. This
    context-manager form fixes that lifecycle bug.
    """

    # Load configuration
    config = PlatformConfig.load()

    # Initialize clients
    llm_client = anthropic.AsyncAnthropic()
    http_timeout = httpx.Timeout(
        config.http.timeout_seconds,
        connect=config.http.connect_timeout_seconds,
        read=config.http.read_timeout_seconds,
        write=config.http.write_timeout_seconds,
    )
    http_limits = httpx.Limits(
        max_connections=config.http.max_connections,
        max_keepalive_connections=config.http.max_keepalive_connections,
    )

    def auth_headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async with httpx.AsyncClient(
        base_url=config.integrations.crm_base_url,
        headers=auth_headers(config.integrations.crm_api_key),
        timeout=http_timeout,
        limits=http_limits,
    ) as crm_client, httpx.AsyncClient(
        base_url=config.integrations.orders_base_url,
        headers=auth_headers(config.integrations.orders_api_key),
        timeout=http_timeout,
        limits=http_limits,
    ) as orders_client, httpx.AsyncClient(
        base_url=config.integrations.payments_base_url,
        headers=auth_headers(config.integrations.payments_api_key),
        timeout=http_timeout,
        limits=http_limits,
    ) as billing_client, httpx.AsyncClient(
        base_url=config.integrations.knowledge_base_url,
        headers=auth_headers(config.integrations.knowledge_base_api_key),
        timeout=http_timeout,
        limits=http_limits,
    ) as knowledge_base_client:

        # Create tools
        crm_tools = [
            IdentifyCustomerTool(crm_client),
            ClassifyIntentTool(llm_client),
        ]

        order_tools = [
            OrderStatusTool(orders_client),
            ModifyOrderTool(orders_client),
            InitiateReturnTool(orders_client),
            TrackShipmentTool(orders_client),
        ]

        billing_tools = [
            GetAccountBalanceTool(billing_client),
            GetPaymentHistoryTool(billing_client),
            ProcessRefundTool(billing_client),
        ]

        # Technical and escalation tools live in the knowledge base /
        # diagnostics path the chapter introduced earlier.
        technical_tools = [
            SearchKnowledgeBaseTool(knowledge_base_client),
            RunDiagnosticTool(orders_client),
        ]
        escalation_tools = [
            # Re-use customer-lookup so an escalation agent can hand off
            # with full context; teams typically add a paging tool here.
            IdentifyCustomerTool(crm_client),
        ]

        # Create agents
        agents = {
            AgentType.TRIAGE: TriageAgent(
                config.agents["triage"], crm_tools, llm_client
            ),
            AgentType.ORDER: OrderAgent(
                config.agents["order"], order_tools, llm_client
            ),
            AgentType.TECHNICAL: TechnicalSupportAgent(
                config.agents["technical"], technical_tools, llm_client
            ),
            AgentType.BILLING: BillingAgent(
                config.agents["billing"], billing_tools, llm_client
            ),
            AgentType.ESCALATION: EscalationAgent(
                config.agents["escalation"], escalation_tools, llm_client
            ),
        }

        # Create manager
        manager = ConversationManager(config, agents)

        # Create metrics collector. InMemoryMetricsStorage is fine for
        # examples and tests; swap in your real time-series backend
        # (Prometheus pushgateway, CloudWatch, etc.) for production.
        metrics_collector = MetricsCollector(
            storage_client=InMemoryMetricsStorage()
        )

        # Graceful shutdown: when the process receives SIGTERM (or
        # SIGINT under a non-interactive runner), stop accepting new
        # conversations, drain in-flight ones up to grace_period, and
        # let the context exit. Without this, a rolling deploy that
        # signals workers can lose every active conversation's
        # in-memory state.
        shutdown_event = asyncio.Event()
        manager.shutting_down = False
        grace_period_seconds = float(
            os.environ.get("CS_SHUTDOWN_GRACE_SECONDS", "60")
        )

        def _request_shutdown(signame: str) -> None:
            logging.getLogger(__name__).warning(
                "platform shutdown requested",
                extra={"signal": signame},
            )
            manager.shutting_down = True
            shutdown_event.set()

        loop = asyncio.get_running_loop()
        installed_signals: list[Any] = []
        for signame in ("SIGTERM", "SIGINT"):
            sig = getattr(signal, signame, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(
                    sig, _request_shutdown, signame
                )
                installed_signals.append(sig)
            except (NotImplementedError, RuntimeError):
                # Non-Unix loops (e.g., Windows ProactorEventLoop) and
                # nested-loop contexts (Jupyter, some test runners)
                # cannot install signal handlers; rely on the caller's
                # higher-level orchestrator to drive shutdown instead.
                pass

        try:
            yield manager, metrics_collector
        finally:
            if shutdown_event.is_set():
                drain = getattr(manager, "drain_active_conversations", None)
                if callable(drain):
                    try:
                        await asyncio.wait_for(
                            drain(), timeout=grace_period_seconds
                        )
                    except asyncio.TimeoutError:
                        logging.getLogger(__name__).warning(
                            "conversation drain hit %ss grace period; "
                            "some in-flight state may not be persisted",
                            grace_period_seconds,
                        )
            for sig in installed_signals:
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
            # Anything platform-wide to flush goes here. The httpx
            # clients are closed automatically when this block exits.
            await metrics_collector.storage.flush()


class InMemoryMetricsStorage:
    """Trivial in-process metrics backend for examples and tests.

    Records every recorded event in a bounded ring buffer; flush() is a
    no-op. Swap for your production backend (Prometheus pushgateway,
    CloudWatch, OpenTelemetry collector) when deploying.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._buffer = deque(maxlen=capacity)

    async def record(self, metric: dict[str, Any]) -> None:
        self._buffer.append(metric)

    async def flush(self) -> None:
        # Real backends drain to remote storage here.
        return None

    def recent(self, n: int = 100) -> list[dict[str, Any]]:
        return list(self._buffer)[-n:]


async def handle_customer_message(
    manager: ConversationManager,
    metrics: MetricsCollector,
    conversation_id: str,
    message: str,
) -> str:
    """Handle an incoming customer message."""

    start_time = datetime.now(timezone.utc)

    # Process the message
    response = await manager.process_message(conversation_id, message)

    # Record metrics
    latency_ms = (
        datetime.now(timezone.utc) - start_time
    ).total_seconds() * 1000
    await metrics.record_first_response(conversation_id, latency_ms)

    for tool_call in response.tool_calls:
        # Use an explicit success flag rather than substring-sniffing
        # the result string (which would misclassify any benign output
        # that contains the word "error").
        success = bool(tool_call.get("success", False))
        await metrics.record_tool_usage(
            tool_call["tool"],
            success,
            0,  # Would need actual timing
        )

    return response.message


async def example_conversation() -> None:
    """Demonstrate a complete conversation flow.

    ``create_platform`` is an ``@asynccontextmanager`` (it yields
    ``(manager, metrics)`` for the lifetime of an ``async with``
    block); using ``await create_platform()`` would TypeError. The
    block scoping also ensures the httpx clients inside the platform
    stay open while the conversation runs and are closed cleanly when
    the block exits.
    """
    async with create_platform() as (manager, metrics):
        # Create a test customer
        customer = Customer(
            customer_id="CUST-12345",
            email="jane.doe@example.com",
            name="Jane Doe",
            tier="premium",
            lifetime_value=5000,
        )

        # Start conversation
        conversation = await manager.start_conversation(
            customer, ConversationChannel.WEB_CHAT
        )

        await metrics.record_conversation_start(conversation)

        # Simulate conversation
        messages = [
            "Hi, I need help with my recent order",
            "jane.doe@example.com",
            "Order ORD-98765, it was supposed to arrive yesterday",
            "Yes, please check the tracking",
            "That's helpful, thanks!",
        ]

        for msg in messages:
            response = await handle_customer_message(
                manager, metrics, conversation.conversation_id, msg
            )
            print(f"Customer: {msg}")
            print(f"Agent: {response}")
            print("---")

        # Resolve conversation
        await manager.resolve_conversation(conversation.conversation_id)

        conv_metrics = manager.conversation_metrics[
            conversation.conversation_id
        ]
        await metrics.record_conversation_end(conversation, conv_metrics)

        # Get summary
        summary = manager.get_conversation_summary(
            conversation.conversation_id
        )
        print(f"\nConversation Summary: {summary}")


if __name__ == "__main__":
    asyncio.run(example_conversation())
