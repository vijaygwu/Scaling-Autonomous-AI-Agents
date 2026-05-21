from __future__ import annotations

"""
Complete Example: Procurement Automation

Code listings from Chapter 06, Book 3:
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

from collections import Counter, OrderedDict, deque
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
from types import SimpleNamespace
import logging
import time

MAX_APPROVAL_CHAIN_ENTRIES = 1000


class PurchaseCategory(Enum):
    """Categories of purchases with different handling rules."""

    SOFTWARE = "software"  # Requires security review above $10K
    HARDWARE = "hardware"  # Requires IT approval
    IT_EQUIPMENT = "hardware"  # Alias used by tests/examples
    SERVICES = "services"  # Requires legal review above $25K
    OFFICE_SUPPLIES = "office_supplies"  # Low friction
    TRAVEL = "travel"  # Requires manager approval
    TRAINING = "training"  # HR must be notified


class ApprovalLevel(Enum):
    """
    Approval levels based on amount and risk.

    Note: Dollar thresholds below are illustrative. Organizations
    vary significantly by size, industry, and risk tolerance.
    Startups may use higher auto-approval limits for agility;
    regulated industries require lower thresholds with additional controls.
    """

    AUTO = "auto"  # System auto-approves (under $500 routine items)
    MANAGER = "manager"  # Up to $5,000
    DIRECTOR = "director"  # $5,001 - $25,000
    VP = "vp"  # $25,001 - $100,000
    EXECUTIVE = "executive"  # Over $100,000
    BOARD = "board"  # Over $1,000,000


class RequestStatus(Enum):
    """Status of a purchase request through the workflow."""

    DRAFT = "draft"
    VALIDATING = "validating"
    PENDING_ANALYSIS = "pending_analysis"
    ANALYZING = "analyzing"
    PENDING_APPROVAL = "pending_approval"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_FULFILLMENT = "pending_fulfillment"
    ORDERED = "ordered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

# ============================================================================
# Block 2 (chapter listing #2)
# ============================================================================

@dataclass
class PurchaseItem:
    """An item in a purchase request."""

    name: str
    category: PurchaseCategory
    quantity: int
    unit_price: float
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    justification: str = ""
    budget_code: str = ""

    @property
    def total_price(self) -> float:
        return self.quantity * self.unit_price

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category.value,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price,
            "vendor_id": self.vendor_id,
            "vendor_name": self.vendor_name,
            "justification": self.justification,
            "budget_code": self.budget_code,
        }


@dataclass
class PurchaseRequest:
    """A purchase request to be processed through the workflow."""

    id: str
    requester_id: str
    requester_name: str
    requester_email: str
    requester_department: str
    items: list[PurchaseItem]
    business_justification: str
    urgency: str = "normal"  # low, normal, high, urgent
    status: RequestStatus = RequestStatus.DRAFT
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # Processing state
    assigned_agent: Optional[str] = None
    validation_result: Optional[dict] = None
    analysis_result: Optional[dict] = None
    approval_chain: list[dict] = field(default_factory=list)

    # Audit trail. Bounded in-memory ring; production deployments
    # MUST also persist each transition to a durable audit log
    # (database, write-once event store) for compliance retention.
    # The 1000-entry cap protects long-running processes from OOM.
    status_history: deque = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    notes: deque = field(
        default_factory=lambda: deque(maxlen=1000)
    )

    @property
    def total_amount(self) -> float:
        return sum(item.total_price for item in self.items)

    @property
    def categories(self) -> set[PurchaseCategory]:
        return set(item.category for item in self.items)

    @property
    def required_approval_level(self) -> ApprovalLevel:
        """Determine required approval level based on amount and category."""
        amount = self.total_amount

        # Board approval for very large purchases
        if amount > 1_000_000:
            return ApprovalLevel.BOARD

        # Executive for large purchases
        if amount > 100_000:
            return ApprovalLevel.EXECUTIVE

        # VP for significant purchases
        if amount > 25_000:
            return ApprovalLevel.VP

        # Director for medium purchases
        if amount > 5_000:
            return ApprovalLevel.DIRECTOR

        # Manager for small purchases
        if amount > 500:
            return ApprovalLevel.MANAGER

        # Auto-approve very small routine items
        if self._is_routine():
            return ApprovalLevel.AUTO

        return ApprovalLevel.MANAGER

    def _is_routine(self) -> bool:
        """Check if this is a routine, low-risk purchase."""
        routine_categories = {PurchaseCategory.OFFICE_SUPPLIES}
        return self.total_amount <= 500 and all(
            item.category in routine_categories for item in self.items
        )

    def update_status(
        self, new_status: RequestStatus, agent_id: str, reason: str = ""
    ):
        """Update status with audit trail."""
        self.status_history.append(
            {
                "from_status": self.status.value,
                "to_status": new_status.value,
                "timestamp": time.time(),
                "agent_id": agent_id,
                "reason": reason,
            }
        )
        self.status = new_status
        self.updated_at = time.time()

    def add_note(self, author: str, content: str, note_type: str = "info"):
        """Add a note to the request."""
        self.notes.append(
            {
                "author": author,
                "content": content,
                "type": note_type,
                "timestamp": time.time(),
            }
        )

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

@dataclass
class Vendor:
    """A vendor/supplier in the system."""

    id: str
    name: str
    categories: list[PurchaseCategory]
    rating: float = 0.0  # 0-5 stars
    compliance_certified: bool = False
    preferred: bool = False
    contact_email: str = ""
    payment_terms: str = "NET30"
    minimum_order: float = 0.0

    # Compliance tracking
    last_audit_date: Optional[float] = None
    certifications: list[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0-100, lower is better

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

"""
Supporting services for the procurement system.

Minimal but importable implementations of the services and result
types that the agents and orchestrator reference. Each class is
either a thin in-process stand-in (IdentityService, BudgetService,
EventBus, AgentMetrics, MockLLM, etc.) or a result/error dataclass.

For production, swap in the equivalents from Book 2 (Chapter 1 for
identity, Chapter 3 for cost control, Chapter 6 for observability).
The shapes in this section match what the orchestrator and agent
code in the rest of this chapter passes at the call sites; if you
extend the orchestrator, keep these shapes or update the call sites.
"""

import asyncio
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Optional

try:
    from opentelemetry import trace  # noqa: F401
except ImportError:
    # PRODUCTION NOTE: the NoOp stub below lets the chapter examples
    # run without an OpenTelemetry install. In a real deployment,
    # OpenTelemetry should be a hard dependency: a missing tracer
    # means missing audit traces, which silently hides where
    # approvals stalled or which agent dropped a request. Fail fast
    # in production builds and keep this stub only for unit tests.
    # Provide a minimal tracer stub so examples can run without OTel installed.
    # Mirrors the surface used by the orchestrator: get_tracer + both
    # start_as_current_span and start_span.
    class _NoOpSpan:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def set_attribute(self, *a, **k):
            pass

        def set_attributes(self, *a, **k):
            pass

        def record_exception(self, *a, **k):
            pass

        def add_event(self, *a, **k):
            pass

    class _NoOpTracer:
        def start_as_current_span(self, *a, **k):
            return _NoOpSpan()

        def start_span(self, *a, **k):
            return _NoOpSpan()

    class _NoOpTraceModule:
        def get_tracer(self, *a, **k):
            return _NoOpTracer()

    trace = _NoOpTraceModule()


# ------------------------------------------------------------
# Error hierarchy
# ------------------------------------------------------------


class ProcurementError(Exception):
    """Base class for procurement-flow errors."""


class IntakeError(ProcurementError):
    pass


class AnalysisError(ProcurementError):
    pass


class ApprovalError(ProcurementError):
    pass


# ------------------------------------------------------------
# Stage-result and routing dataclasses
#
# These match the shapes the orchestrator and agent code pass
# at construction; do not rename fields without updating the
# call sites.
# ------------------------------------------------------------


@dataclass
class ValidationResult:
    """Outcome of intake-stage validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0


@dataclass
class ValidationRule:
    rule_id: str
    description: str
    check: Optional[Callable[[Any], ValidationResult]] = None


@dataclass
class AuthResult:
    """Outcome of an authorization check."""

    authorized: bool = False
    reason: str = ""
    agent_id: str = ""
    scopes: list[str] = field(default_factory=list)


@dataclass
class IntakeResult:
    """Outcome of the intake stage."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    enrichments: dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "enrichments": dict(self.enrichments),
            "processing_time": self.processing_time,
        }


@dataclass
class ItemAnalysis:
    """Per-item analysis result, mutated by the AnalysisAgent.

    Field names match the attributes the AnalysisAgent writes to at
    runtime: ``potential_savings`` (set from ContractService comparison)
    and ``alternative_vendors`` (populated from VendorDatabase.find_by_category).
    """

    item_name: str
    alternative_vendors: list[Any] = field(default_factory=list)
    potential_savings: float = 0.0  # matches AnalysisAgent's write
    estimated_savings: float = 0.0  # legacy alias retained for older callers
    risk_flags: list[str] = field(default_factory=list)
    notes: str = ""
    # Fields written by AnalysisAgent._analyze_item; declared explicitly so
    # @dataclass(slots=True) compatibility and serialization both see them.
    recommended_vendor: Optional[Any] = None
    savings_reason: str = ""
    compliance_issues: list["ComplianceIssue"] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Aggregate risk view for a purchase request."""

    level: str = "low"  # 'low', 'medium', 'high'
    score: float = 0.0  # 0-100, lower is safer
    factors: list[str] = field(default_factory=list)


@dataclass
class ComplianceIssue:
    """A specific compliance concern."""

    severity: str = "low"
    rule: str = ""
    message: str = ""


@dataclass
class ComplianceRule:
    rule_id: str
    description: str
    check: Optional[Callable[[Any], list[ComplianceIssue]]] = None


@dataclass
class AnalysisResult:
    """Output of the AnalysisAgent."""

    item_analyses: list[ItemAnalysis] = field(default_factory=list)
    total_potential_savings: float = 0.0
    compliance_issues: list[ComplianceIssue] = field(default_factory=list)
    risk_level: str = "low"
    risk_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    processing_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_analyses": [
                {
                    "item_name": a.item_name,
                    "alternative_vendors": list(a.alternative_vendors),
                    "potential_savings": a.potential_savings,
                    "risk_flags": list(a.risk_flags),
                    "notes": a.notes,
                    "recommended_vendor": getattr(
                        a.recommended_vendor, "id", a.recommended_vendor
                    ),
                    "savings_reason": a.savings_reason,
                    "compliance_issues": [
                        {
                            "severity": ci.severity,
                            "rule": ci.rule,
                            "message": ci.message,
                        }
                        for ci in a.compliance_issues
                    ],
                }
                for a in self.item_analyses
            ],
            "total_potential_savings": self.total_potential_savings,
            "compliance_issues": [
                {"severity": c.severity, "rule": c.rule, "message": c.message}
                for c in self.compliance_issues
            ],
            "risk_level": self.risk_level,
            "risk_factors": list(self.risk_factors),
            "recommendations": list(self.recommendations),
            "processing_time": self.processing_time,
        }


@dataclass
class Approver:
    approver_id: str = ""
    name: str = ""
    role: str = ""
    max_approval_amount: float = 0.0
    email: str = ""
    level: Optional["ApprovalLevel"] = None
    departments: list[str] = field(default_factory=list)
    # Accept ``id`` as an alias for ``approver_id`` so that demo construction
    # sites that pass ``id=`` continue to work.
    id: str = ""

    def __post_init__(self) -> None:
        if self.id and not self.approver_id:
            self.approver_id = self.id
        elif self.approver_id and not self.id:
            self.id = self.approver_id


@dataclass
class ApprovalTask:
    """A pending approval task surfaced to a human approver.

    The dashboard / approval-interface builds these from a request and
    its analysis; the orchestrator builds them from the approval chain.
    All fields beyond ``request_id`` are optional so both construction
    sites work with the same dataclass.
    """

    request_id: str
    task_id: str = ""
    approver: Optional[Approver] = None
    created_at: Optional[datetime] = None
    timeout_at: Optional[datetime] = None
    status: str = "pending"
    decision_notes: str = ""
    # Fields populated by the dashboard view:
    requester: str = ""
    department: str = ""
    amount: float = 0.0
    urgency: str = ""
    categories: list[str] = field(default_factory=list)
    risk_level: str = ""
    recommendations: list[str] = field(default_factory=list)
    compliance_issues: list[Any] = field(default_factory=list)
    submitted_at: Optional[float] = None
    waiting_time: float = 0.0


@dataclass
class ApprovalRoutingResult:
    """Routing decision from the ApprovalAgent."""

    status: str = "pending"  # 'auto_approved', 'pending_approval', 'rejected'
    reason: str = ""
    required_level: Any = None
    approval_chain: list[Any] = field(default_factory=list)
    estimated_time: float = 0.0  # seconds until expected resolution

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "required_level": (
                str(self.required_level)
                if self.required_level is not None
                else None
            ),
            "approval_chain": [
                getattr(a, "approver_id", a) for a in self.approval_chain
            ],
            "estimated_time": self.estimated_time,
        }


@dataclass
class ApprovalDecisionResult:
    """Final approval verdict.

    ``order`` is populated once fulfillment has emitted the PO; consumers
    that only need the verdict can ignore it.
    """

    status: str = (
        "pending"  # 'fully_approved', 'partially_approved', 'rejected', 'fulfilled'
    )
    next_step: str = ""
    next_approver: Any = None
    reason: str = ""
    order: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "next_step": self.next_step,
            "next_approver": getattr(
                self.next_approver, "approver_id", self.next_approver
            ),
            "reason": self.reason,
            "order": self.order,
        }


@dataclass
class AutoApprovalResult:
    """Outcome of the auto-approval fast path."""

    eligible: bool = False
    reason: str = ""


@dataclass
class DelegationRule:
    primary_approver_id: str
    delegate_approver_id: str
    valid_until: Optional[datetime] = None


@dataclass
class SubmissionResult:
    """Result of submitting a request to the orchestrator.

    The orchestrator constructs this at three call sites with slightly
    different kwarg sets (validation_failed, completed, pending_approval),
    so warnings, order, and analysis_summary are all optional.
    """

    status: str = (
        ""  # 'completed', 'validation_failed', 'pending_approval', etc.
    )
    request_id: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    auto_approved: bool = False
    approval_info: dict[str, Any] = field(default_factory=dict)
    analysis_summary: dict[str, Any] = field(default_factory=dict)
    order: Optional[dict[str, Any]] = None


@dataclass
class RequestDetails:
    """Hydrated view of a request returned by the dashboard."""

    request: Any
    items: list[dict] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    analysis: Optional[AnalysisResult] = None
    approval_history: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    status_history: list[dict] = field(default_factory=list)


# ------------------------------------------------------------
# IdentityService
# ------------------------------------------------------------


@dataclass
class _User:
    user_id: str
    permissions: list[str] = field(default_factory=list)
    spending_limit: float = 0.0

    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions


class IdentityService:
    """In-process identity service.

    For production, swap in the X.509 / OIDC implementation from
    Book 2, Chapter 1.
    """

    def __init__(self, max_agents: int = 10_000, max_users: int = 10_000):
        self._agents: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._users: "OrderedDict[str, _User]" = OrderedDict()
        self._max_agents = max_agents
        self._max_users = max_users

    def register_agent(self, agent_id: str, scopes: list[str]) -> None:
        self._agents[agent_id] = {"scopes": list(scopes)}
        self._agents.move_to_end(agent_id)
        while len(self._agents) > self._max_agents:
            self._agents.popitem(last=False)

    def register_user(
        self,
        user_id: str,
        permissions: list[str],
        spending_limit: float = 0.0,
    ) -> None:
        self._users[user_id] = _User(
            user_id, list(permissions), spending_limit
        )
        self._users.move_to_end(user_id)
        while len(self._users) > self._max_users:
            self._users.popitem(last=False)

    async def get_user(self, user_id: str) -> Optional[_User]:
        user = self._users.get(user_id)
        if user is not None:
            self._users.move_to_end(user_id)
        return user

    async def authenticate(self, agent_id: str, token: str) -> AuthResult:
        # Demo-only token convention: valid-token-for-<id>
        if agent_id not in self._agents:
            return AuthResult(
                authorized=False, reason="unknown agent", agent_id=agent_id
            )
        if token != f"valid-token-for-{agent_id}":
            return AuthResult(
                authorized=False, reason="invalid token", agent_id=agent_id
            )
        self._agents.move_to_end(agent_id)
        return AuthResult(
            authorized=True,
            agent_id=agent_id,
            scopes=self._agents[agent_id]["scopes"],
        )


# ------------------------------------------------------------
# BudgetService
# ------------------------------------------------------------


class BudgetService:
    """Tracks daily and monthly spend caps per cost center.

    For production, swap in the BudgetManager from Book 2, Chapter 3.
    """

    def __init__(self, max_cost_centers: int = 10_000):
        self._caps: "OrderedDict[str, dict[str, float]]" = OrderedDict()
        self._spent: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._max_cost_centers = max_cost_centers

    def set_cap(self, cost_center: str, daily: float, monthly: float) -> None:
        self._caps[cost_center] = {"daily": daily, "monthly": monthly}
        self._caps.move_to_end(cost_center)
        while len(self._caps) > self._max_cost_centers:
            stale_cost_center, _ = self._caps.popitem(last=False)
            self._spent.pop(stale_cost_center, None)

    def can_spend(self, cost_center: str, amount: float) -> bool:
        cap = self._caps.get(cost_center)
        if not cap:
            return False
        self._caps.move_to_end(cost_center)
        s = self._get_spend_bucket(cost_center)
        return (
            s["daily"] + amount <= cap["daily"]
            and s["monthly"] + amount <= cap["monthly"]
        )

    def record_spend(self, cost_center: str, amount: float) -> None:
        bucket = self._get_spend_bucket(cost_center)
        bucket["daily"] += amount
        bucket["monthly"] += amount

    def _get_spend_bucket(self, cost_center: str) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        day_key = now.date().isoformat()
        month_key = f"{now.year:04d}-{now.month:02d}"
        bucket = self._spent.get(cost_center)
        if bucket is None:
            bucket = {
                "daily": 0.0,
                "monthly": 0.0,
                "day": day_key,
                "month": month_key,
            }
            self._spent[cost_center] = bucket
        self._spent.move_to_end(cost_center)
        if bucket["day"] != day_key:
            bucket["daily"] = 0.0
            bucket["day"] = day_key
        if bucket["month"] != month_key:
            bucket["monthly"] = 0.0
            bucket["month"] = month_key
        while len(self._spent) > self._max_cost_centers:
            self._spent.popitem(last=False)
        return bucket

    async def get_remaining(
        self,
        cost_center: str,
        period: str = "monthly",
    ) -> float:
        cap = self._caps.get(cost_center)
        if not cap:
            return 0.0
        return max(0.0, cap[period] - self._get_spend_bucket(cost_center)[period])


# ------------------------------------------------------------
# EventBus
# ------------------------------------------------------------


class EventBus:
    """In-process pub/sub bus.

    Supports exact-topic subscriptions and ``prefix.*`` wildcards
    (e.g., subscribe to ``request.*`` to receive every
    ``request.something`` event).
    """

    def __init__(
        self,
        max_subscribers_per_topic: int = 100,
        max_topics: int = 1_000,
        handler_timeout_seconds: float = 5.0,
    ):
        self._exact: dict[str, list[Callable[[dict], Awaitable[None]]]] = (
            defaultdict(list)
        )
        self._wildcards: dict[
            str, list[Callable[[dict], Awaitable[None]]]
        ] = defaultdict(list)
        self._max_subscribers_per_topic = max_subscribers_per_topic
        self._max_topics = max_topics
        self._handler_timeout_seconds = handler_timeout_seconds
        self.failed_deliveries: deque[dict[str, Any]] = deque(maxlen=1000)

    def subscribe(
        self,
        topic: str,
        handler: Callable[[dict], Awaitable[None]],
    ) -> Callable[[], None]:
        registry = self._wildcards if topic.endswith(".*") else self._exact
        key = topic[:-2] if topic.endswith(".*") else topic
        if key not in registry and self._topic_count() >= self._max_topics:
            raise ValueError("too many event topics registered")
        if len(registry[key]) >= self._max_subscribers_per_topic:
            raise ValueError(f"too many subscribers for topic {topic}")
        registry[key].append(handler)

        def unsubscribe() -> None:
            try:
                registry[key].remove(handler)
            except ValueError:
                pass

        return unsubscribe

    def _topic_count(self) -> int:
        return len(self._exact) + len(self._wildcards)

    async def emit(self, topic: str, event: dict) -> None:
        envelope = {
            "topic": topic,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        # Exact-match subscribers
        for h in self._exact.get(topic, []):
            await self._deliver(h, envelope, "exact")
        # Wildcard subscribers: walk every prefix of topic
        parts = topic.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            for h in self._wildcards.get(prefix, []):
                await self._deliver(h, envelope, "wildcard")

    async def _deliver(
        self,
        handler: Callable[[dict], Awaitable[None]],
        envelope: dict,
        subscription_type: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                handler(envelope), timeout=self._handler_timeout_seconds
            )
        except (
            asyncio.TimeoutError,
            TimeoutError,
            RuntimeError,
            OSError,
            ValueError,
            TypeError,
        ) as exc:
            self.failed_deliveries.append(
                {
                    "topic": envelope.get("topic"),
                    "subscription_type": subscription_type,
                    "error": str(exc),
                    "event": envelope.get("event"),
                }
            )
            logging.getLogger(__name__).warning(
                "%s subscriber failed for %s: %s",
                subscription_type,
                envelope.get("topic", "?"),
                exc,
            )

    # Backward-compatible alias
    publish = emit


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------


class AgentMetrics:
    """Per-agent counters and histograms.

    ``timings`` is a bounded ring buffer per metric name (last
    ``MAX_TIMING_SAMPLES`` measurements) so percentile estimates stay
    accurate while memory cannot grow unboundedly. For production,
    replace with the OpenTelemetry / Prometheus instrumentation from
    Book 2, Chapter 6.
    """

    MAX_TIMING_SAMPLES = 10_000
    MAX_METRIC_KEYS = 1_000

    def __init__(self, agent_id: str = "anonymous"):
        self.agent_id = agent_id
        self.counters: OrderedDict[str, int] = OrderedDict()
        self.timings: OrderedDict[str, deque[float]] = OrderedDict()

    def _ensure_metric_key(self, registry: OrderedDict, name: str, factory):
        if name in registry:
            registry.move_to_end(name)
            return registry[name]
        if len(registry) >= self.MAX_METRIC_KEYS:
            registry.popitem(last=False)
        registry[name] = factory()
        return registry[name]

    def increment(self, name: str, by: int = 1) -> None:
        current = self._ensure_metric_key(self.counters, name, lambda: 0)
        self.counters[name] = current + by

    def get(self, name: str, default: int = 0) -> int:
        return self.counters.get(name, default)

    def record_timing(self, name: str, seconds: float) -> None:
        samples = self._ensure_metric_key(
            self.timings,
            name,
            lambda: deque(maxlen=self.MAX_TIMING_SAMPLES),
        )
        samples.append(seconds)


class OrchestratorMetrics(AgentMetrics):
    """Orchestrator-level metrics with stage durations and routing."""

    def __init__(self, agent_id: str = "orchestrator"):
        super().__init__(agent_id)

    def record_stage(self, stage: str, seconds: float, success: bool) -> None:
        self.record_timing(f"stage.{stage}.seconds", seconds)
        self.increment(f"stage.{stage}.{'success' if success else 'failure'}")


# ------------------------------------------------------------
# Approver registry
# ------------------------------------------------------------


class ApproverRegistry:
    """Maps approval levels to eligible approvers."""

    def __init__(
        self, max_approvers: int = 10_000, max_delegations: int = 10_000
    ):
        self._max_approvers = max_approvers
        self._max_delegations = max_delegations
        self._by_level: dict[Any, list[Approver]] = defaultdict(list)
        self._by_id: dict[str, Approver] = {}
        self._delegations: list[DelegationRule] = []

    def register(self, level: Any, approver: Optional[Approver] = None) -> None:
        # Allow ``register(approver)`` shorthand by reading level off the
        # approver itself.
        if approver is None and isinstance(level, Approver):
            approver = level
            level = approver.level
        if approver is None:
            raise ValueError("approver is required")
        if approver.approver_id not in self._by_id:
            if len(self._by_id) >= self._max_approvers:
                raise ValueError("too many approvers registered")
        else:
            prior = self._by_id[approver.approver_id]
            for approvers in self._by_level.values():
                try:
                    approvers.remove(prior)
                except ValueError:
                    pass
        self._by_level[level].append(approver)
        self._by_id[approver.approver_id] = approver

    def add_delegation(self, rule: DelegationRule) -> None:
        if len(self._delegations) >= self._max_delegations:
            raise ValueError("too many delegation rules registered")
        self._delegations.append(rule)

    def approvers_for(self, level: Any) -> list[Approver]:
        return list(self._by_level.get(level, []))

    async def get(self, approver_id: str) -> Optional[Approver]:
        """Look up an approver by id (O(1))."""
        return self._by_id.get(approver_id)


# ------------------------------------------------------------
# Vendor and contract services
# ------------------------------------------------------------


class VendorDatabase:
    """In-memory vendor catalog.

    For production, replace with a real CRM/ERP integration.
    """

    def __init__(self, max_vendors: int = 100_000):
        self._vendors: "OrderedDict[str, Any]" = OrderedDict()
        self._max_vendors = max_vendors

    def add(self, vendor: Any) -> None:
        self._vendors[vendor.id] = vendor
        self._vendors.move_to_end(vendor.id)
        while len(self._vendors) > self._max_vendors:
            self._vendors.popitem(last=False)

    async def get(self, vendor_id: str) -> Optional[Any]:
        vendor = self._vendors.get(vendor_id)
        if vendor is not None:
            self._vendors.move_to_end(vendor_id)
        return vendor

    async def find_by_category(self, category: Any) -> list[Any]:
        return [
            v
            for v in self._vendors.values()
            if category in getattr(v, "categories", [])
        ]


class ContractService:
    """Looks up negotiated pricing for a vendor and item."""

    def __init__(self, max_contracts: int = 100_000):
        self._contracts: "OrderedDict[tuple[str, str], dict[str, Any]]" = (
            OrderedDict()
        )
        self._max_contracts = max_contracts

    def set_contract(
        self,
        vendor_id: str,
        item_name: str,
        unit_price: float,
        valid_until: Optional[datetime] = None,
    ) -> None:
        key = (vendor_id, item_name)
        self._contracts[key] = {
            "unit_price": unit_price,
            "valid_until": valid_until,
        }
        self._contracts.move_to_end(key)
        while len(self._contracts) > self._max_contracts:
            self._contracts.popitem(last=False)

    async def find_contract(
        self,
        vendor_id: str,
        item_name: str,
    ) -> Optional[dict[str, Any]]:
        key = (vendor_id, item_name)
        c = self._contracts.get(key)
        if not c:
            return None
        self._contracts.move_to_end(key)
        if c["valid_until"] and c["valid_until"] < datetime.now(timezone.utc):
            return None
        return c


# ------------------------------------------------------------
# MockLLM for tests
# ------------------------------------------------------------


@dataclass
class MockResponse:
    content: str
    tool_calls: list[dict] = field(default_factory=list)


class MockLLM:
    """Deterministic stand-in for an LLM, useful in tests."""

    def __init__(self, responses: Optional[list[MockResponse]] = None):
        self._responses = list(responses or [])
        self._default = MockResponse(content="(mock default response)")
        self.calls: list[dict] = []

    def add_response(self, response: MockResponse) -> None:
        self._responses.append(response)

    def set_default_response(self, response: MockResponse) -> None:
        self._default = response

    async def complete(
        self,
        messages: list[dict],
        **kwargs: Any,
    ) -> MockResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._responses:
            return self._responses.pop(0)
        return self._default


# ------------------------------------------------------------
# Processing context shared across stages
# ------------------------------------------------------------


@dataclass
class ProcessingContext:
    """Runtime context flowing through the orchestrator stages.

    Construct with the services and registries the agents need.
    Agents read from these via attribute access; tests substitute
    fakes by setting different services on the context.
    """

    request_id: str = ""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actor_agent_id: str = ""
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    identity_service: IdentityService = field(default_factory=IdentityService)
    budget_service: BudgetService = field(default_factory=BudgetService)
    vendor_db: VendorDatabase = field(default_factory=VendorDatabase)
    contract_service: ContractService = field(default_factory=ContractService)
    approver_registry: ApproverRegistry = field(
        default_factory=ApproverRegistry
    )
    attributes: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# FulfillmentAgent
# ------------------------------------------------------------


@dataclass
class FulfillmentResult:
    """Outcome of the fulfillment stage."""

    order_id: str
    items: list[dict[str, Any]] = field(default_factory=list)
    submitted_at: str = ""
    status: str = "submitted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "items": list(self.items),
            "submitted_at": self.submitted_at,
            "status": self.status,
        }


class FulfillmentAgent:
    """Final stage: emits the purchase order to the vendor system.

    Constructible with no arguments for parity with the other agents
    in this chapter; lazily creates default in-process services if
    none are passed.
    """

    def __init__(
        self,
        agent_id: str = "fulfillment",
        identity: Optional[IdentityService] = None,
        bus: Optional[EventBus] = None,
        metrics: Optional[AgentMetrics] = None,
    ):
        self.agent_id = agent_id
        self.identity = identity or IdentityService()
        self.bus = bus or EventBus()
        self.metrics = metrics or AgentMetrics(agent_id)

    async def process(
        self,
        request: Any,
        context: Optional[ProcessingContext] = None,
    ) -> FulfillmentResult:
        """Orchestrator-facing entry point.

        Builds the PO, emits the fulfillment event, and returns a
        ``FulfillmentResult`` (which has ``order_id`` and ``to_dict()``
        for the orchestrator to use).
        """
        ctx = context or ProcessingContext()
        rid = getattr(request, "id", getattr(request, "request_id", ""))
        result = FulfillmentResult(
            order_id=f"PO-{rid}",
            items=[
                {
                    "name": getattr(it, "name", str(it)),
                    "qty": getattr(it, "quantity", 1),
                    "unit_price": getattr(it, "unit_price", 0.0),
                }
                for it in getattr(request, "items", [])
            ],
            submitted_at=datetime.now(timezone.utc).isoformat(),
            status="submitted",
        )
        self.metrics.increment("fulfilled")
        await self.bus.emit(
            "procurement.fulfilled",
            {
                "request_id": rid,
                "order_id": result.order_id,
                "correlation_id": ctx.correlation_id,
            },
        )
        return result

    # Alias retained for callers that prefer the more descriptive name.
    async def fulfill(
        self,
        request: Any,
        approval: ApprovalDecisionResult,
        ctx: ProcessingContext,
    ) -> FulfillmentResult:
        if approval.status not in ("fully_approved", "partially_approved"):
            raise ProcurementError("Cannot fulfill an unapproved request.")
        return await self.process(request, ctx)

# ============================================================================
# Block 5 (chapter listing #5)
# ============================================================================

class IntakeAgent:
    """
    Validates and prepares purchase requests for processing.

    Responsibilities:
    - Validate required fields
    - Check requester authorization
    - Validate budget codes
    - Detect duplicate requests
    - Enrich request with default values
    """

    def __init__(self, agent_id: str = "intake-agent"):
        self.agent_id = agent_id
        self.validation_rules: list[ValidationRule] = []
        self.metrics = AgentMetrics()

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self.validation_rules.append(rule)

    async def process(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> IntakeResult:
        """Process a new purchase request."""
        self.metrics.increment("requests_processed")
        start_time = time.time()

        try:
            request.update_status(RequestStatus.VALIDATING, self.agent_id)

            errors = []
            warnings = []
            enrichments = {}

            # Basic validation
            basic_result = self._validate_basic(request)
            errors.extend(basic_result.errors)
            warnings.extend(basic_result.warnings)

            # Check requester authorization
            auth_result = await self._check_authorization(request, context)
            if not auth_result.authorized:
                errors.append(
                    f"Requester not authorized: {auth_result.reason}"
                )

            # Validate budget codes
            budget_result = await self._validate_budgets(request, context)
            errors.extend(budget_result.errors)
            warnings.extend(budget_result.warnings)

            # Apply custom rules
            for rule in self.validation_rules:
                if rule.check is None:
                    continue
                rule_result = rule.check(request)
                errors.extend(rule_result.errors)
                warnings.extend(rule_result.warnings)

            # Check for duplicates
            duplicate = await self._check_duplicates(request, context)
            if duplicate:
                warnings.append(
                    f"Similar request {duplicate.id} submitted on "
                    f"{time.strftime('%Y-%m-%d', time.localtime(duplicate.created_at))}"
                )

            # Enrich request
            if not errors:
                enrichments = await self._enrich_request(request, context)

            # Create result
            result = IntakeResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                enrichments=enrichments,
                processing_time=time.time() - start_time,
            )

            # Update request
            request.validation_result = result.to_dict()
            if result.valid:
                request.update_status(
                    RequestStatus.PENDING_ANALYSIS,
                    self.agent_id,
                    "Validation passed",
                )
            else:
                self.metrics.increment("requests_failed_validation")

            return result

        except (TimeoutError, ConnectionError, OSError, RuntimeError, ValueError) as e:
            self.metrics.increment("requests_errored")
            raise IntakeError(f"Intake processing failed: {e}") from e

    def _validate_basic(self, request: PurchaseRequest) -> ValidationResult:
        """Basic field validation."""
        errors = []
        warnings = []

        if not request.items:
            errors.append("Request must have at least one item")

        if not request.business_justification:
            warnings.append("No business justification provided")

        if request.total_amount <= 0:
            errors.append("Total amount must be positive")

        for i, item in enumerate(request.items):
            if item.quantity <= 0:
                errors.append(f"Item {i+1}: Quantity must be positive")
            if item.unit_price <= 0:
                errors.append(f"Item {i+1}: Unit price must be positive")
            if not item.name:
                errors.append(f"Item {i+1}: Name is required")

        return ValidationResult(errors=errors, warnings=warnings)

    async def _check_authorization(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> AuthResult:
        """Check if requester is authorized to make purchases."""
        # In production, check against identity service
        user = await context.identity_service.get_user(request.requester_id)

        if not user:
            return AuthResult(authorized=False, reason="User not found")

        if not user.has_permission("procurement:create"):
            return AuthResult(
                authorized=False, reason="Missing procurement permission"
            )

        # Check spending limit
        if request.total_amount > user.spending_limit:
            return AuthResult(
                authorized=False,
                reason=f"Amount ${request.total_amount:,.2f} exceeds "
                f"spending limit ${user.spending_limit:,.2f}",
            )

        return AuthResult(authorized=True)

    async def _validate_budgets(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> ValidationResult:
        """Validate that the request fits the available budget."""
        cost_center = request.requester_department
        if not context.budget_service.can_spend(
            cost_center, request.total_amount
        ):
            remaining = await context.budget_service.get_remaining(
                cost_center
            )
            return ValidationResult(
                errors=[
                    f"Insufficient budget for {cost_center}: "
                    f"${remaining:,.2f} remaining, "
                    f"${request.total_amount:,.2f} requested"
                ]
            )
        warnings = []
        for item in request.items:
            if not item.budget_code:
                warnings.append(
                    f"Item '{item.name}' has no budget code; "
                    f"defaulting to {cost_center}"
                )
        return ValidationResult(warnings=warnings)

    async def _check_duplicates(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> Optional[PurchaseRequest]:
        """Find a recent duplicate request if one is provided in context."""
        recent = context.attributes.get("recent_requests", [])
        request_items = {(item.name, item.vendor_id) for item in request.items}
        for candidate in recent:
            if candidate.id == request.id:
                continue
            candidate_items = {
                (item.name, item.vendor_id) for item in candidate.items
            }
            if (
                candidate.requester_id == request.requester_id
                and candidate_items == request_items
                and abs(candidate.total_amount - request.total_amount) < 0.01
            ):
                return candidate
        return None

    async def _enrich_request(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> dict[str, Any]:
        """Apply safe default enrichment after validation."""
        enrichments: dict[str, Any] = {}
        for item in request.items:
            if not item.budget_code:
                item.budget_code = request.requester_department
                enrichments.setdefault("budget_codes", {})[
                    item.name
                ] = item.budget_code
        context.attributes["last_validated_request_id"] = request.id
        return enrichments

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

class AnalysisAgent:
    """
    Analyzes purchase requests for optimization and compliance.

    Responsibilities:
    - Find alternative vendors
    - Identify cost savings opportunities
    - Flag compliance issues
    - Assess risk level
    - Generate recommendations
    """

    def __init__(self, agent_id: str = "analysis-agent"):
        self.agent_id = agent_id
        self.vendor_db: VendorDatabase = None
        self.compliance_rules: list[ComplianceRule] = []
        self.metrics = AgentMetrics()

    async def process(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> AnalysisResult:
        """Analyze a validated purchase request."""
        self.metrics.increment("requests_analyzed")
        start_time = time.time()

        try:
            request.update_status(RequestStatus.ANALYZING, self.agent_id)

            # Analyze each item
            item_analyses = []
            total_potential_savings = 0.0
            compliance_issues = []

            if not request.items:
                compliance_issues.append(
                    ComplianceIssue(
                        severity="high",
                        rule="MISSING_ITEMS",
                        message=(
                            "Purchase request must include at least one item"
                        ),
                    )
                )

            for item in request.items:
                item_analysis = await self._analyze_item(item, context)
                item_analyses.append(item_analysis)
                total_potential_savings += item_analysis.potential_savings
                compliance_issues.extend(item_analysis.compliance_issues)

            # Overall risk assessment
            risk = await self._assess_risk(
                request, item_analyses, compliance_issues, context
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                request, item_analyses, risk, compliance_issues
            )

            result = AnalysisResult(
                item_analyses=item_analyses,
                total_potential_savings=total_potential_savings,
                compliance_issues=compliance_issues,
                risk_level=risk.level,
                risk_factors=risk.factors,
                recommendations=recommendations,
                processing_time=time.time() - start_time,
            )

            # Update request
            request.analysis_result = result.to_dict()
            request.update_status(
                RequestStatus.PENDING_APPROVAL,
                self.agent_id,
                f"Analysis complete. Risk: {risk.level}",
            )

            return result

        except (TimeoutError, ConnectionError, OSError, RuntimeError, ValueError) as e:
            self.metrics.increment("requests_errored")
            raise AnalysisError(f"Analysis failed: {e}") from e

    async def _analyze_item(
        self, item: PurchaseItem, context: ProcessingContext
    ) -> ItemAnalysis:
        """Analyze a single item."""
        analysis = ItemAnalysis(item_name=item.name)

        # AnalysisAgent.vendor_db is None until the orchestrator wires
        # it in at startup; skip only alternative-vendor comparison so
        # contract pricing and compliance checks below can still run.
        if self.vendor_db is None:
            analysis.savings_reason = (
                "vendor_db not configured; alternative-vendor analysis "
                "skipped"
            )
        else:
            # Find alternative vendors
            vendors = await self.vendor_db.find_by_category(item.category)
            preferred = [
                v for v in vendors if v.preferred and v.compliance_certified
            ]

            if preferred:
                best = max(preferred, key=lambda v: v.rating)
                analysis.recommended_vendor = best

                # Estimate savings from preferred vendor pricing
                if best.id != item.vendor_id:
                    # Placeholder savings estimate. A production version would
                    # query ContractService.find_contract(vendor, item) for each
                    # alternative vendor, compute price deltas against the
                    # incumbent, and report the realized minimum. The 5%
                    # multiplier here is illustrative only.
                    analysis.potential_savings = item.total_price * 0.05
                    analysis.savings_reason = (
                        f"Switch to preferred vendor {best.name}"
                    )

            # Check compliance
            if item.total_price > 10000 and item.vendor_id:
                vendor = await self.vendor_db.get(item.vendor_id)
                if vendor and not vendor.compliance_certified:
                    analysis.compliance_issues.append(
                        ComplianceIssue(
                            severity="high",
                            rule="VENDOR_CERTIFICATION",
                            message=f"Vendor {vendor.name} not certified for "
                            f"purchases over $10,000",
                        )
                    )

        # Check for contract pricing
        contract = await context.contract_service.find_contract(
            item.vendor_id, item.name
        )
        if contract and item.unit_price > contract["unit_price"]:
            savings = (
                item.unit_price - contract["unit_price"]
            ) * item.quantity
            analysis.potential_savings += savings
            analysis.savings_reason = (
                f"Contract pricing available: ${contract['unit_price']:.2f} "
                f"vs ${item.unit_price:.2f}"
            )

        return analysis

    async def _assess_risk(
        self,
        request: PurchaseRequest,
        item_analyses: list[ItemAnalysis],
        compliance_issues: list[ComplianceIssue],
        context: ProcessingContext,
    ) -> RiskAssessment:
        """Assess overall risk of the request."""
        factors = []
        score = 0  # 0-100

        # Amount-based risk
        if request.total_amount > 100000:
            factors.append("High-value purchase (>$100K)")
            score += 30
        elif request.total_amount > 25000:
            factors.append("Significant purchase (>$25K)")
            score += 15

        # Compliance issues
        if compliance_issues:
            high_severity = [
                i for i in compliance_issues if i.severity == "high"
            ]
            if high_severity:
                factors.append(
                    f"{len(high_severity)} high-severity compliance issues"
                )
                score += 25 * len(high_severity)

        # New vendor risk
        for item in request.items:
            if item.vendor_id:
                vendor_db = self.vendor_db or context.vendor_db
                if vendor_db is None:
                    continue
                vendor = await vendor_db.get(item.vendor_id)
                if vendor and vendor.risk_score > 50:
                    factors.append(f"High-risk vendor: {vendor.name}")
                    score += 20

        # Urgency risk
        if request.urgency == "urgent":
            factors.append("Urgent request (reduced review time)")
            score += 10

        # Determine level
        if score >= 50:
            level = "high"
        elif score >= 25:
            level = "medium"
        else:
            level = "low"

        return RiskAssessment(level=level, score=score, factors=factors)

    def _generate_recommendations(
        self,
        request: PurchaseRequest,
        item_analyses: list[ItemAnalysis],
        risk: RiskAssessment,
        compliance_issues: list[ComplianceIssue],
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Cost savings
        total_savings = sum(a.potential_savings for a in item_analyses)
        if (
            total_savings > request.total_amount * 0.05
        ):  # >5% savings available
            recommendations.append(
                f"Potential cost savings of ${total_savings:,.2f} identified. "
                f"Consider vendor alternatives."
            )

        # Compliance
        if compliance_issues:
            recommendations.append(
                f"{len(compliance_issues)} compliance issues require attention "
                f"before approval."
            )

        # Risk mitigation
        if risk.level == "high":
            recommendations.append(
                "High-risk request. Recommend additional review by "
                "procurement specialist."
            )

        # Urgency
        if request.urgency == "urgent" and risk.level != "low":
            recommendations.append(
                "Urgent request with elevated risk. Consider expedited "
                "review process."
            )

        return recommendations

# ============================================================================
# Block 7 (chapter listing #7)
# ============================================================================

class ApprovalAgent:
    """
    Manages the approval workflow for purchase requests.

    Responsibilities:
    - Determine approval requirements
    - Route to appropriate approvers
    - Handle auto-approval for routine items
    - Manage approval chain
    - Handle escalation and delegation
    """

    def __init__(self, agent_id: str = "approval-agent"):
        self.agent_id = agent_id
        self.approver_registry: Optional[ApproverRegistry] = None
        self.delegation_rules: list[DelegationRule] = []
        self.metrics = AgentMetrics()

    async def process(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> ApprovalRoutingResult:
        """Determine approval routing for a request."""
        self.metrics.increment("requests_processed")

        required_level = request.required_approval_level

        # Check for auto-approval eligibility
        auto_result = await self._check_auto_approval(request, context)
        if auto_result.eligible:
            request.approval_chain.append(
                {
                    "level": "auto",
                    "approver_id": "system",
                    "approver_name": "Auto-Approval System",
                    "decision": "approved",
                    "timestamp": time.time(),
                    "reason": auto_result.reason,
                }
            )
            del request.approval_chain[:-MAX_APPROVAL_CHAIN_ENTRIES]
            request.update_status(
                RequestStatus.APPROVED,
                self.agent_id,
                f"Auto-approved: {auto_result.reason}",
            )
            self.metrics.increment("requests_auto_approved")
            return ApprovalRoutingResult(
                status="auto_approved", reason=auto_result.reason
            )

        # Build approval chain
        chain = await self._build_approval_chain(request, context)

        # Check for delegations
        chain = await self._apply_delegations(chain, context)

        request.approval_chain = [
            {
                "level": approver.level.value if approver.level else "",
                "approver_id": approver.approver_id,
                "approver_name": approver.name,
                "decision": "pending",
                "timestamp": time.time(),
            }
            for approver in chain
        ][-MAX_APPROVAL_CHAIN_ENTRIES:]

        # Notify approvers
        await self._notify_approvers(request, chain, context)

        request.update_status(
            RequestStatus.PENDING_APPROVAL,
            self.agent_id,
            f"Awaiting {required_level.value} approval",
        )

        return ApprovalRoutingResult(
            status="pending_approval",
            required_level=required_level,
            approval_chain=chain,
            estimated_time=self._estimate_approval_time(chain),
        )

    async def submit_decision(
        self,
        request: PurchaseRequest,
        approver_id: str,
        decision: str,
        notes: str = "",
    ) -> ApprovalDecisionResult:
        """Process an approval decision from a human approver."""
        self.metrics.increment("decisions_processed")

        # Validate approver
        approver = await self.approver_registry.get(approver_id)
        if not approver:
            raise ApprovalError(f"Unknown approver: {approver_id}")

        # Check if approver is authorized for this request
        if not self._is_authorized_approver(request, approver):
            raise ApprovalError(
                f"Approver {approver_id} not authorized for this request"
            )

        # Record decision against the pending approval entry when present.
        for approval in request.approval_chain:
            if (
                approval.get("approver_id") == approver_id
                and approval.get("decision") == "pending"
            ):
                approval.update(
                    {
                        "decision": decision,
                        "timestamp": time.time(),
                        "notes": notes,
                    }
                )
                break
        else:
            request.approval_chain.append(
                {
                    "level": approver.level.value if approver.level else "",
                    "approver_id": approver_id,
                    "approver_name": approver.name,
                    "decision": decision,
                    "timestamp": time.time(),
                    "notes": notes,
                }
            )
            del request.approval_chain[:-MAX_APPROVAL_CHAIN_ENTRIES]

        if decision == "approved":
            # Check if all required approvals are complete
            if self._all_approvals_complete(request):
                request.update_status(
                    RequestStatus.APPROVED,
                    self.agent_id,
                    f"Approved by {approver.name}",
                )
                self.metrics.increment("requests_approved")
                return ApprovalDecisionResult(
                    status="fully_approved", next_step="fulfillment"
                )
            else:
                # More approvals needed
                return ApprovalDecisionResult(
                    status="partially_approved",
                    next_approver=self._get_next_approver(request),
                )

        elif decision == "rejected":
            request.update_status(
                RequestStatus.REJECTED,
                self.agent_id,
                f"Rejected by {approver.name}: {notes}",
            )
            self.metrics.increment("requests_rejected")
            return ApprovalDecisionResult(status="rejected", reason=notes)

        elif decision == "request_info":
            request.add_note(
                approver_id,
                f"Additional information requested: {notes}",
                "info_request",
            )
            return ApprovalDecisionResult(
                status="info_requested", reason=notes
            )

        # Default: explicit failure on unknown decision strings.
        # Previously this branch silently returned ``None``, so a
        # typo'd decision (e.g. "approve" instead of "approved") would
        # propagate to the orchestrator as a missing result and
        # crash on attribute access.
        raise ApprovalError(
            f"Unknown decision '{decision}' for request "
            f"{request.id}; expected one of "
            f"approved/rejected/request_info"
        )

    async def _check_auto_approval(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> AutoApprovalResult:
        """Check if request can be auto-approved."""
        # Low-value office supplies
        if request.total_amount <= 500 and all(
            item.category == PurchaseCategory.OFFICE_SUPPLIES
            for item in request.items
        ):
            return AutoApprovalResult(
                eligible=True,
                reason="Low-value office supplies (policy AUTO-001)",
            )

        # Pre-approved vendor + budget available
        if request.total_amount <= 1000:
            all_preferred = True
            for item in request.items:
                if item.vendor_id:
                    vendor = await context.vendor_db.get(item.vendor_id)
                    if not (vendor and vendor.preferred):
                        all_preferred = False
                        break

            if all_preferred:
                # Check budget
                budget = await context.budget_service.get_remaining(
                    request.requester_department
                )
                if budget >= request.total_amount:
                    return AutoApprovalResult(
                        eligible=True,
                        reason="Preferred vendor with available budget (policy AUTO-002)",
                    )

        return AutoApprovalResult(eligible=False)

    def _registry(self, context: ProcessingContext) -> ApproverRegistry:
        """Return the active approver registry."""
        if self.approver_registry is None:
            self.approver_registry = context.approver_registry
        return self.approver_registry

    async def _build_approval_chain(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> list[Approver]:
        """Build the ordered human approval chain for a request."""
        registry = self._registry(context)
        required_level = request.required_approval_level
        approvers = [
            approver
            for approver in registry.approvers_for(required_level)
            if self._approver_can_handle_request(request, approver)
        ]
        if not approvers:
            raise ApprovalError(
                f"No approver registered for {required_level.value} "
                f"request {request.id}"
            )
        return approvers[:1]

    async def _apply_delegations(
        self, chain: list[Approver], context: ProcessingContext
    ) -> list[Approver]:
        """Apply active delegation rules to the approval chain."""
        registry = self._registry(context)
        now = datetime.now(timezone.utc)
        delegated = []
        for approver in chain:
            replacement = approver
            for rule in registry._delegations:
                if rule.primary_approver_id != approver.approver_id:
                    continue
                if rule.valid_until and rule.valid_until < now:
                    continue
                delegate = await registry.get(rule.delegate_approver_id)
                if delegate:
                    replacement = delegate
                    break
            delegated.append(replacement)
        return delegated

    async def _notify_approvers(
        self,
        request: PurchaseRequest,
        chain: list[Approver],
        context: ProcessingContext,
    ) -> None:
        """Send approval notifications.

        Production systems would deliver email, Slack, or workflow tasks.
        The listing records observable intent without depending on an
        external notification service.
        """
        logger = logging.getLogger(__name__)
        for approver in chain:
            logger.info(
                "approval requested",
                extra={
                    "request_id": request.id,
                    "approver_id": approver.approver_id,
                    "correlation_id": context.correlation_id,
                },
            )

    def _estimate_approval_time(self, chain: list[Approver]) -> float:
        """Estimate approval latency in seconds."""
        return 4 * 60 * 60 * max(1, len(chain))

    def _level_rank(self, level: Optional[ApprovalLevel]) -> int:
        order = {
            ApprovalLevel.AUTO: 0,
            ApprovalLevel.MANAGER: 1,
            ApprovalLevel.DIRECTOR: 2,
            ApprovalLevel.VP: 3,
            ApprovalLevel.EXECUTIVE: 4,
            ApprovalLevel.BOARD: 5,
        }
        return order.get(level, -1)

    def _approver_can_handle_request(
        self, request: PurchaseRequest, approver: Approver
    ) -> bool:
        if approver.level is None:
            return False
        if self._level_rank(approver.level) < self._level_rank(
            request.required_approval_level
        ):
            return False
        if approver.max_approval_amount and (
            request.total_amount > approver.max_approval_amount
        ):
            return False
        return (
            not approver.departments
            or request.requester_department in approver.departments
        )

    def _is_authorized_approver(
        self, request: PurchaseRequest, approver: Approver
    ) -> bool:
        """Check that the approver is eligible for the pending request."""
        pending_ids = {
            item.get("approver_id")
            for item in request.approval_chain
            if item.get("decision") == "pending"
        }
        if pending_ids and approver.approver_id not in pending_ids:
            return False
        return self._approver_can_handle_request(request, approver)

    def _all_approvals_complete(self, request: PurchaseRequest) -> bool:
        """Return True once every planned approver has approved."""
        pending = [
            item for item in request.approval_chain
            if item.get("decision") == "pending"
        ]
        if pending:
            return False
        return any(
            item.get("decision") == "approved"
            for item in request.approval_chain
        )

    def _get_next_approver(
        self, request: PurchaseRequest
    ) -> Optional[dict]:
        """Return the next pending approver entry, if any."""
        for item in request.approval_chain:
            if item.get("decision") == "pending":
                return item
        return None

# ============================================================================
# Block 8 (chapter listing #8)
# ============================================================================

class ProcurementOrchestrator:
    """
    Orchestrates the multi-agent procurement workflow.

    Coordinates:
    - Agent execution sequence
    - Event emission for observability
    - Error handling and recovery
    - Metrics collection
    """

    def __init__(self, max_in_memory_requests: int = 10_000):
        if max_in_memory_requests < 1:
            raise ValueError("max_in_memory_requests must be >= 1")
        self.intake = IntakeAgent()
        self.analysis = AnalysisAgent()
        self.approval = ApprovalAgent()
        self.fulfillment = FulfillmentAgent()

        # In-memory hot cache of recent requests. Bounded so a
        # long-running orchestrator cannot OOM on accumulated state;
        # the durable copy lives in your purchase-request store
        # (mentioned in the audit-log note on PurchaseRequest above).
        self.requests: "OrderedDict[str, PurchaseRequest]" = (
            __import__("collections").OrderedDict()
        )
        self._max_in_memory_requests = max_in_memory_requests
        # Per-request context map: submit_request stashes the
        # ProcessingContext under request.id so handle_approval_decision
        # can recover the same identity/budget/contract services
        # instead of constructing an empty ProcessingContext().
        self._contexts: dict[str, "ProcessingContext"] = {}
        self.event_bus = EventBus()
        self.metrics = OrchestratorMetrics()

        # Set up tracing
        self.tracer = trace.get_tracer("procurement-orchestrator")

    def _evict_oldest_request(self) -> None:
        """Drop the oldest in-memory request and its processing context.

        Used as the single entry point for hot-cache eviction so the
        ``self.requests`` OrderedDict and the ``self._contexts`` dict
        cannot drift out of sync; the durable purchase-request store
        retains the request itself.
        """
        if not self.requests:
            return
        evicted_id, _ = self.requests.popitem(last=False)
        self._contexts.pop(evicted_id, None)

    async def submit_request(
        self, request: PurchaseRequest, context: ProcessingContext
    ) -> SubmissionResult:
        """Submit a new purchase request for processing."""
        with self.tracer.start_as_current_span(
            "submit_request",
            attributes={
                "request.id": request.id,
                "request.amount": request.total_amount,
                "request.department": request.requester_department,
            },
        ) as span:

            # Idempotency: if the same request id is resubmitted
            # (network retry, doubled webhook delivery), short-circuit
            # rather than starting a second processing pipeline. The
            # caller can then observe the already-running submission
            # via the request status API.
            if request.id in self.requests:
                self.metrics.increment("requests_duplicate")
                return SubmissionResult(
                    status="duplicate",
                    request_id=request.id,
                    warnings=[
                        "duplicate submission; original still in flight",
                    ],
                )

            # Store request + processing context for later phases
            # (approval decisions arrive asynchronously and need the
            # same identity/budget/contract services that submitted).
            self.requests[request.id] = request
            self._contexts[request.id] = context
            self.metrics.increment("requests_submitted")
            # Bound the in-memory hot cache via the single
            # ``_evict_oldest_request`` helper so contexts and requests
            # are guaranteed to evict together; the request itself
            # remains durable in your purchase-request store.
            while len(self.requests) > self._max_in_memory_requests:
                self._evict_oldest_request()
            assert len(self._contexts) <= len(self.requests), (
                "_contexts/requests sync invariant violated"
            )

            await self.event_bus.emit(
                "request.submitted",
                {
                    "request_id": request.id,
                    "requester": request.requester_id,
                    "amount": request.total_amount,
                    "timestamp": time.time(),
                },
            )

            try:
                # Phase 1: Intake
                with self.tracer.start_span("intake") as intake_span:
                    intake_result = await self.intake.process(
                        request, context
                    )
                    intake_span.set_attribute("valid", intake_result.valid)

                if not intake_result.valid:
                    await self.event_bus.emit(
                        "request.validation_failed",
                        {
                            "request_id": request.id,
                            "errors": intake_result.errors,
                        },
                    )
                    return SubmissionResult(
                        status="validation_failed",
                        request_id=request.id,
                        errors=intake_result.errors,
                        warnings=intake_result.warnings,
                    )

                await self.event_bus.emit(
                    "request.validated", {"request_id": request.id}
                )

                # Phase 2: Analysis
                with self.tracer.start_span("analysis") as analysis_span:
                    analysis_result = await self.analysis.process(
                        request, context
                    )
                    analysis_span.set_attribute(
                        "risk_level", analysis_result.risk_level
                    )

                await self.event_bus.emit(
                    "request.analyzed",
                    {
                        "request_id": request.id,
                        "risk_level": analysis_result.risk_level,
                        "potential_savings": analysis_result.total_potential_savings,
                    },
                )

                # Phase 3: Approval routing
                with self.tracer.start_span("approval_routing"):
                    approval_result = await self.approval.process(
                        request, context
                    )

                if approval_result.status == "auto_approved":
                    await self.event_bus.emit(
                        "request.auto_approved",
                        {
                            "request_id": request.id,
                            "reason": approval_result.reason,
                        },
                    )

                    # Auto-approved requests go straight to fulfillment
                    fulfillment_result = await self.fulfillment.process(
                        request, context
                    )

                    await self.event_bus.emit(
                        "request.fulfilled",
                        {
                            "request_id": request.id,
                            "order_id": fulfillment_result.order_id,
                        },
                    )

                    return SubmissionResult(
                        status="completed",
                        request_id=request.id,
                        auto_approved=True,
                        order=fulfillment_result.to_dict(),
                    )

                # Request needs human approval
                await self.event_bus.emit(
                    "request.pending_approval",
                    {
                        "request_id": request.id,
                        "required_level": approval_result.required_level.value,
                        "estimated_time": approval_result.estimated_time,
                    },
                )

                self.metrics.increment("requests_pending_approval")

                return SubmissionResult(
                    status="pending_approval",
                    request_id=request.id,
                    approval_info=approval_result.to_dict(),
                    analysis_summary={
                        "risk_level": analysis_result.risk_level,
                        "recommendations": analysis_result.recommendations,
                    },
                )

            except (IntakeError, AnalysisError, ApprovalError) as e:
                span.record_exception(e)
                self.metrics.increment("requests_errored")

                await self.event_bus.emit(
                    "request.error",
                    {"request_id": request.id, "error": str(e)},
                )

                raise
            except (TimeoutError, RuntimeError, OSError, ValueError) as e:
                span.record_exception(e)
                self.metrics.increment("requests_errored")

                await self.event_bus.emit(
                    "request.error",
                    {"request_id": request.id, "error": str(e)},
                )

                raise ProcurementError(
                    f"Failed to process request {request.id}: {e}"
                ) from e

    async def process(
        self,
        request: PurchaseRequest,
        context: Optional[ProcessingContext] = None,
    ) -> SimpleNamespace:
        """Compatibility wrapper for older tests/listings.

        The production API is ``submit_request(request, context)``.
        This wrapper keeps extracted examples runnable by creating a
        default context and translating the result back to the older
        ``status``/``approval_level`` shape.
        """
        if context is None:
            context = ProcessingContext(request_id=request.id)
            context.identity_service.register_user(
                request.requester_id,
                permissions=["procurement:create"],
                spending_limit=max(request.total_amount, 1_000_000),
            )
            context.budget_service.set_cap(
                request.requester_department,
                daily=max(request.total_amount, 1_000_000),
                monthly=max(request.total_amount, 1_000_000),
            )
        required_level = request.required_approval_level
        if (
            required_level != ApprovalLevel.AUTO
            and not context.approver_registry.approvers_for(required_level)
        ):
            context.approver_registry.register(
                required_level,
                Approver(
                    approver_id=f"default-{required_level.value}",
                    name=f"Default {required_level.value.title()} Approver",
                    role="Approver",
                    max_approval_amount=max(request.total_amount, 1_000_000),
                    level=required_level,
                    departments=[request.requester_department],
                ),
            )

        result = await self.submit_request(request, context)
        if result.status == "completed":
            status = RequestStatus.APPROVED
            approver_name = "Auto-Approval"
        elif result.status == "pending_approval":
            status = RequestStatus.PENDING_APPROVAL
            approver_name = ""
        else:
            status = request.status
            approver_name = ""

        return SimpleNamespace(
            status=status,
            approval_level=required_level,
            approver_name=approver_name,
            submission=result,
        )

    async def handle_approval_decision(
        self,
        request_id: str,
        approver_id: str,
        decision: str,
        notes: str = "",
    ) -> ApprovalDecisionResult:
        """Handle an approval decision from a human approver."""
        if request_id not in self.requests:
            raise ProcurementError(f"Request not found: {request_id}")

        request = self.requests[request_id]

        with self.tracer.start_as_current_span(
            "handle_approval",
            attributes={
                "request.id": request_id,
                "approver.id": approver_id,
                "decision": decision,
            },
        ):
            result = await self.approval.submit_decision(
                request, approver_id, decision, notes
            )

            await self.event_bus.emit(
                "approval.decision",
                {
                    "request_id": request_id,
                    "approver_id": approver_id,
                    "decision": decision,
                    "result_status": result.status,
                },
            )

            if result.status == "fully_approved":
                # Proceed to fulfillment. Recover the ProcessingContext
                # stashed during submit_request so fulfillment runs
                # with the real identity/budget/contract services
                # instead of an empty context that would fail every
                # downstream lookup.
                context = self._contexts.get(request_id)
                if context is None:
                    raise ApprovalError(
                        f"Cannot fulfill request {request_id}: "
                        f"ProcessingContext was not preserved from intake"
                    )
                fulfillment_result = await self.fulfillment.process(
                    request, context
                )

                await self.event_bus.emit(
                    "request.fulfilled",
                    {
                        "request_id": request_id,
                        "order_id": fulfillment_result.order_id,
                    },
                )

                return ApprovalDecisionResult(
                    status="fulfilled", order=fulfillment_result.to_dict()
                )

            return result

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

class ApprovalInterface:
    """
    Interface for human approvers to review and decide on requests.
    """

    def __init__(self, orchestrator: ProcurementOrchestrator):
        self.orchestrator = orchestrator

    def get_pending_approvals(self, approver_id: str) -> list[ApprovalTask]:
        """Get pending approvals for a specific approver."""
        pending = []

        for request in self.orchestrator.requests.values():
            if request.status == RequestStatus.PENDING_APPROVAL:
                if self._is_approver_for(request, approver_id):
                    pending.append(
                        ApprovalTask(
                            request_id=request.id,
                            requester=request.requester_name,
                            department=request.requester_department,
                            amount=request.total_amount,
                            urgency=request.urgency,
                            categories=[c.value for c in request.categories],
                            risk_level=request.analysis_result.get(
                                "risk_level", "unknown"
                            ),
                            recommendations=request.analysis_result.get(
                                "recommendations", []
                            ),
                            compliance_issues=request.analysis_result.get(
                                "compliance_issues", []
                            ),
                            submitted_at=request.created_at,
                            waiting_time=time.time() - request.created_at,
                        )
                    )

        # Sort by urgency then waiting time
        urgency_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
        pending.sort(
            key=lambda t: (urgency_order[t.urgency], -t.waiting_time)
        )

        return pending

    def _is_approver_for(
        self, request: PurchaseRequest, approver_id: str
    ) -> bool:
        """Return True when approver_id has a pending task for request."""
        return any(
            item.get("approver_id") == approver_id
            and item.get("decision") == "pending"
            for item in request.approval_chain
        )

    def get_request_details(self, request_id: str) -> RequestDetails:
        """Get full details for a request."""
        request = self.orchestrator.requests.get(request_id)
        if not request:
            raise ValueError(f"Request not found: {request_id}")

        return RequestDetails(
            request=request,
            items=[item.to_dict() for item in request.items],
            validation=request.validation_result,
            analysis=request.analysis_result,
            approval_history=request.approval_chain,
            notes=request.notes,
            status_history=request.status_history,
        )

    async def approve(
        self, request_id: str, approver_id: str, notes: str = ""
    ) -> ApprovalDecisionResult:
        """Approve a request."""
        return await self.orchestrator.handle_approval_decision(
            request_id, approver_id, "approved", notes
        )

    async def reject(
        self, request_id: str, approver_id: str, reason: str
    ) -> ApprovalDecisionResult:
        """Reject a request."""
        return await self.orchestrator.handle_approval_decision(
            request_id, approver_id, "rejected", reason
        )

    async def request_info(
        self, request_id: str, approver_id: str, question: str
    ) -> ApprovalDecisionResult:
        """Request additional information before deciding."""
        return await self.orchestrator.handle_approval_decision(
            request_id, approver_id, "request_info", question
        )

# ============================================================================
# Block 10 (chapter listing #10)
# ============================================================================

class ProcurementDashboard:
    """
    Dashboard for procurement operations with event-driven updates.
    """

    def __init__(self, orchestrator: ProcurementOrchestrator):
        self.orchestrator = orchestrator
        self._recent_events: deque[dict] = deque(maxlen=1000)
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up event handlers for dashboard updates."""
        self.orchestrator.event_bus.subscribe(
            "request.*", self._handle_request_event
        )
        self.orchestrator.event_bus.subscribe(
            "approval.*", self._handle_approval_event
        )

    def get_summary(self) -> dict:
        """Get current system summary."""
        requests = list(self.orchestrator.requests.values())

        return {
            "total_requests": len(requests),
            "by_status": self._count_by_status(requests),
            "total_value": sum(r.total_amount for r in requests),
            "pending_approval_value": sum(
                r.total_amount
                for r in requests
                if r.status == RequestStatus.PENDING_APPROVAL
            ),
            "average_processing_time": self._calc_avg_processing_time(
                requests
            ),
            "auto_approval_rate": self._calc_auto_approval_rate(),
            "top_requesters": self._get_top_requesters(requests),
            "top_categories": self._get_top_categories(requests),
        }

    def get_approval_metrics(self) -> dict:
        """Get approval workflow metrics."""
        return {
            "pending_count": self.orchestrator.metrics.get(
                "requests_pending_approval"
            ),
            "approved_today": self._get_decisions_today("approved"),
            "rejected_today": self._get_decisions_today("rejected"),
            "average_approval_time": self._calc_avg_approval_time(),
            "oldest_pending": self._get_oldest_pending(),
            "by_level": self._get_approvals_by_level(),
        }

    async def _handle_request_event(self, envelope: dict) -> None:
        """Record request events for dashboard metrics."""
        self._recent_events.append(envelope)

    async def _handle_approval_event(self, envelope: dict) -> None:
        """Record approval events for dashboard metrics."""
        self._recent_events.append(envelope)

    def _count_by_status(
        self, requests: list[PurchaseRequest]
    ) -> dict[str, int]:
        return dict(Counter(request.status.value for request in requests))

    def _calc_avg_processing_time(
        self, requests: list[PurchaseRequest]
    ) -> float:
        completed = [
            request.updated_at - request.created_at
            for request in requests
            if request.status
            in {
                RequestStatus.ORDERED,
                RequestStatus.COMPLETED,
                RequestStatus.APPROVED,
                RequestStatus.REJECTED,
            }
        ]
        return sum(completed) / len(completed) if completed else 0.0

    def _calc_auto_approval_rate(self) -> float:
        requests = list(self.orchestrator.requests.values())
        if not requests:
            return 0.0
        auto_approved = 0
        for request in requests:
            if any(
                item.get("approver_id") == "system"
                and item.get("decision") == "approved"
                for item in request.approval_chain
            ):
                auto_approved += 1
        return auto_approved / len(requests)

    def _get_top_requesters(
        self, requests: list[PurchaseRequest], limit: int = 5
    ) -> list[dict]:
        counts = Counter(request.requester_name for request in requests)
        return [
            {"requester": requester, "count": count}
            for requester, count in counts.most_common(limit)
        ]

    def _get_top_categories(
        self, requests: list[PurchaseRequest], limit: int = 5
    ) -> list[dict]:
        counts: Counter[str] = Counter()
        for request in requests:
            for category in request.categories:
                counts[category.value] += 1
        return [
            {"category": category, "count": count}
            for category, count in counts.most_common(limit)
        ]

    def _get_decisions_today(self, decision: str) -> int:
        current_date = datetime.now(timezone.utc).date()
        total = 0
        for envelope in self._recent_events:
            if envelope.get("topic") != "approval.decision":
                continue
            try:
                event_date = datetime.fromisoformat(
                    envelope["ts"]
                ).date()
            except (KeyError, ValueError):
                continue
            if (
                event_date == current_date
                and envelope.get("event", {}).get("decision") == decision
            ):
                total += 1
        return total

    def _calc_avg_approval_time(self) -> float:
        durations = []
        for request in self.orchestrator.requests.values():
            if request.status not in {
                RequestStatus.APPROVED,
                RequestStatus.REJECTED,
                RequestStatus.ORDERED,
                RequestStatus.COMPLETED,
            }:
                continue
            durations.append(request.updated_at - request.created_at)
        return sum(durations) / len(durations) if durations else 0.0

    def _get_oldest_pending(self) -> float:
        pending_ages = [
            time.time() - request.created_at
            for request in self.orchestrator.requests.values()
            if request.status == RequestStatus.PENDING_APPROVAL
        ]
        return max(pending_ages) if pending_ages else 0.0

    def _get_approvals_by_level(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for request in self.orchestrator.requests.values():
            for approval in request.approval_chain:
                if approval.get("decision") == "pending":
                    counts[approval.get("level", "unknown")] += 1
        return dict(counts)

# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

async def main():
    """Demonstration of the procurement system."""

    # Initialize
    orchestrator = ProcurementOrchestrator()

    # Configure vendors
    orchestrator.analysis.vendor_db = VendorDatabase()
    orchestrator.analysis.vendor_db.add(
        Vendor(
            id="dell-001",
            name="Dell Technologies",
            categories=[PurchaseCategory.HARDWARE],
            rating=4.8,
            compliance_certified=True,
            preferred=True,
        )
    )

    # Configure approvers
    orchestrator.approval.approver_registry = ApproverRegistry()
    orchestrator.approval.approver_registry.register(
        Approver(
            id="mgr-001",
            name="Jane Smith",
            email="jane.smith@company.com",
            level=ApprovalLevel.MANAGER,
            departments=["Engineering", "Product"],
            max_approval_amount=5_000,
        )
    )
    orchestrator.approval.approver_registry.register(
        Approver(
            id="dir-001",
            name="Sam Lee",
            email="sam.lee@company.com",
            level=ApprovalLevel.DIRECTOR,
            departments=["Engineering", "Product"],
            max_approval_amount=25_000,
        )
    )

    # Set up observability
    dashboard = ProcurementDashboard(orchestrator)

    # Create approval interface
    approval_ui = ApprovalInterface(orchestrator)

    # Submit a request
    request = PurchaseRequest(
        id="REQ-2024-001",
        requester_id="emp-123",
        requester_name="John Doe",
        requester_email="john.doe@company.com",
        requester_department="Engineering",
        items=[
            PurchaseItem(
                name="Developer Laptop",
                category=PurchaseCategory.HARDWARE,
                quantity=5,
                unit_price=1800.00,
                vendor_id="dell-001",
                justification="New team members starting Q2",
            )
        ],
        business_justification="Expanding engineering team by 5 developers",
        urgency="high",
    )

    context = ProcessingContext(
        identity_service=IdentityService(),
        budget_service=BudgetService(),
        contract_service=ContractService(),
    )
    context.identity_service.register_user(
        "emp-123",
        permissions=["procurement:create"],
        spending_limit=25_000,
    )
    context.budget_service.set_cap(
        "Engineering",
        daily=50_000,
        monthly=250_000,
    )

    # Submit and process
    result = await orchestrator.submit_request(request, context)
    print(f"Submission result: {result.status}")

    if result.status == "pending_approval":
        # Show pending approvals
        pending = approval_ui.get_pending_approvals("dir-001")
        print(f"Pending approvals: {len(pending)}")

        # Approve the request
        decision = await approval_ui.approve(
            request.id, "dir-001", "Approved for Q2 expansion"
        )
        print(f"Decision result: {decision.status}")

    # Show dashboard summary
    summary = dashboard.get_summary()
    print(f"Total requests: {summary['total_requests']}")
    print(f"Total value: ${summary['total_value']:,.2f}")


if __name__ == "__main__":
    asyncio.run(main())

# ============================================================================
# Block 12 (chapter listing #12)
# ============================================================================

# tests/unit/test_analysis_agent.py
#
# In your project, the imports below resolve from your own
# src/procurement/... package. We guard them with try/except so this
# book listing parses cleanly when extracted as a single module: the
# ``as Name`` form rebinds successfully when the production layout is
# on the path; otherwise the except branch leaves the in-module class
# definitions (Mock LLM, PurchaseRequest, etc.) intact for tests to
# collect against.
import importlib
from unittest.mock import AsyncMock

try:
    pytest = importlib.import_module("pytest")
except ImportError:
    class _PytestMarkStub:
        def parametrize(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class _PytestStub:
        mark = _PytestMarkStub()

        def fixture(self, func=None, **kwargs):
            if func is None:
                def decorator(wrapped):
                    return wrapped
                return decorator
            return func

    pytest = _PytestStub()

try:
    # In a real project this imports from your src/procurement/ tree.
    # When the book listing is extracted as a single module these names
    # already exist (AnalysisAgent, PurchaseRequest, PurchaseItem, MockLLM,
    # MockResponse are defined elsewhere in the same module). We rebind
    # them through this import only if the real package is on the path;
    # the except branch leaves the in-module definitions intact rather
    # than shadowing them with None.
    from src.procurement.agents import AnalysisAgent as AnalysisAgent
    from src.procurement.models import (
        PurchaseRequest as PurchaseRequest,
        PurchaseItem as PurchaseItem,
    )
    from src.testing.mock_llm import (
        MockLLM as MockLLM,
        MockResponse as MockResponse,
    )
    _SRC_PROC_AVAILABLE = True
except ImportError:
    # Fall back to the in-module definitions above. Tests still collect
    # and run (no NameError); they exercise the in-module classes
    # instead of the production-layout ones.
    _SRC_PROC_AVAILABLE = False


@pytest.fixture
def mock_llm():
    llm = MockLLM()
    llm.add_response(
        MockResponse(
            content="Analysis complete. Found 15% savings opportunity "
            "by switching to preferred vendor TechSupply Inc."
        )
    )
    return llm


@pytest.fixture
def sample_request():
    # Field names match the PurchaseItem / PurchaseRequest dataclasses
    # defined earlier in this chapter; the earlier draft of this
    # fixture used ``description=``/string ``category=`` which never
    # actually instantiated cleanly.
    return PurchaseRequest(
        id="REQ-001",
        requester_id="user_123",
        requester_name="Test User",
        requester_email="test@example.com",
        requester_department="Engineering",
        items=[
            PurchaseItem(
                name="Laptop computers",
                category=PurchaseCategory.IT_EQUIPMENT,
                quantity=10,
                unit_price=1200.00,
            )
        ],
        business_justification="Equipment refresh for engineering team",
    )


async def test_analysis_agent_identifies_savings(mock_llm, sample_request):
    """AnalysisAgent should identify cost savings opportunities."""
    agent = AnalysisAgent()
    agent.vendor_db = VendorDatabase()

    context = ProcessingContext(
        request_id=sample_request.id,
        actor_agent_id=agent.agent_id,
    )
    result = await agent.process(sample_request, context)

    assert result.total_potential_savings >= 0


async def test_analysis_agent_flags_compliance_issues(mock_llm):
    """AnalysisAgent should flag requests missing required fields."""
    mock_llm.set_default_response(
        MockResponse(content="Compliance issue: Missing cost center code.")
    )
    agent = AnalysisAgent()

    incomplete_request = PurchaseRequest(
        id="REQ-002",
        requester_id="user_456",
        requester_name="Test User",
        requester_email="test@example.com",
        requester_department="Engineering",
        items=[],  # Empty items list
        business_justification="",
    )

    context = ProcessingContext(
        request_id=incomplete_request.id,
        actor_agent_id=agent.agent_id,
    )
    result = await agent.process(incomplete_request, context)

    assert len(result.compliance_issues) > 0

# ============================================================================
# Block 13 (chapter listing #13)
# ============================================================================

# tests/integration/test_orchestrator.py
#
# Imports below are guarded so this listing parses when extracted as a
# single module. In your project, they resolve from src/procurement/.
try:
    # See the unit-test block above for the rebind-via-``as`` rationale.
    # In-module fallbacks keep tests collectable in either layout.
    from src.procurement.orchestrator import (
        ProcurementOrchestrator as ProcurementOrchestrator,
    )
    from src.procurement.models import (
        PurchaseRequest as PurchaseRequest,
        RequestStatus as RequestStatus,
        ApprovalLevel as ApprovalLevel,
    )
    _SRC_PROC_ORCH_AVAILABLE = True
except ImportError:
    _SRC_PROC_ORCH_AVAILABLE = False


@pytest.fixture
def orchestrator():
    return ProcurementOrchestrator()


def create_test_request(amount: float, category: str) -> PurchaseRequest:
    """Helper to create test requests with specified amount."""
    category_map = {
        "software": PurchaseCategory.SOFTWARE,
        "hardware": PurchaseCategory.HARDWARE,
        "office_supplies": PurchaseCategory.OFFICE_SUPPLIES,
        "consulting": PurchaseCategory.SERVICES,
    }
    purchase_category = category_map.get(category, PurchaseCategory.SERVICES)
    return PurchaseRequest(
        id=f"REQ-{uuid.uuid4().hex[:8]}",
        requester_id="test_user",
        requester_name="Test User",
        requester_email="test@example.com",
        requester_department="Engineering",
        items=[
            PurchaseItem(
                name=f"Test {category} purchase",
                quantity=1,
                unit_price=amount,
                category=purchase_category,
            )
        ],
        business_justification=f"Test {category} purchase",
    )


async def test_low_value_request_auto_approved(orchestrator):
    """Requests under $500 should auto-approve without human intervention."""
    request = create_test_request(amount=100, category="office_supplies")

    result = await orchestrator.process(request)

    assert result.status == RequestStatus.APPROVED
    assert result.approval_level == ApprovalLevel.AUTO
    assert "Auto-Approval" in result.approver_name


async def test_high_value_request_requires_approval(orchestrator):
    """Requests over threshold should pause for human approval."""
    request = create_test_request(amount=50000, category="software")

    result = await orchestrator.process(request)

    assert result.status == RequestStatus.PENDING_APPROVAL
    assert result.approval_level == ApprovalLevel.VP


async def test_restricted_category_requires_review(orchestrator):
    """Restricted categories require additional review regardless of amount."""
    request = create_test_request(amount=200, category="consulting")

    result = await orchestrator.process(request)

    # Even small consulting purchases need manager approval
    assert result.approval_level != ApprovalLevel.AUTO

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

# tests/scenarios/test_approval_routing.py

@pytest.mark.parametrize(
    "amount,expected_level",
    [
        (100, "auto"),  # Under $500
        (499, "auto"),  # Just under threshold
        (500, "auto"),  # At threshold (inclusive)
        (501, "manager"),  # Just over threshold
        (3000, "manager"),  # Mid-range manager
        (5000, "manager"),  # At manager limit
        (5001, "director"),  # Just over manager limit
        (15000, "director"),  # Mid-range director
        (25000, "director"),  # At director limit
        (25001, "vp"),  # Just over director limit
        (75000, "vp"),  # Mid-range VP
        (100000, "vp"),  # At VP limit
        (100001, "executive"),  # Over VP limit
    ],
)
async def test_approval_routing_by_amount(
    orchestrator, amount, expected_level
):
    """Verify correct approval routing based on purchase amount."""
    request = create_test_request(amount=amount, category="office_supplies")

    result = await orchestrator.process(request)

    assert (
        result.approval_level.value == expected_level
    ), f"Amount ${amount} should route to {expected_level}, got {result.approval_level.value}"
