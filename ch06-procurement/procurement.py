from __future__ import annotations

"""
Complete Example: Procurement Automation

Code listings from Chapter 06, Book 3:
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

from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import time

class PurchaseCategory(Enum):
    """Categories of purchases with different handling rules."""
    SOFTWARE = "software"          # Requires security review above $10K
    HARDWARE = "hardware"          # Requires IT approval
    SERVICES = "services"          # Requires legal review above $25K
    OFFICE_SUPPLIES = "office_supplies"  # Low friction
    TRAVEL = "travel"              # Requires manager approval
    TRAINING = "training"          # HR must be notified

class ApprovalLevel(Enum):
    """
    Approval levels based on amount and risk.

    Note: Dollar thresholds below are illustrative. Organizations
    vary significantly by size, industry, and risk tolerance.
    Startups may use higher auto-approval limits for agility;
    regulated industries require lower thresholds with additional controls.
    """
    AUTO = "auto"            # System auto-approves (under $500 routine items)
    MANAGER = "manager"      # Up to $5,000
    DIRECTOR = "director"    # $5,001 - $25,000
    VP = "vp"               # $25,001 - $100,000
    EXECUTIVE = "executive"  # Over $100,000
    BOARD = "board"         # Over $1,000,000

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
            "budget_code": self.budget_code
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
    
    # Audit trail
    status_history: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    
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
        return (
            self.total_amount <= 500 and
            all(item.category in routine_categories for item in self.items)
        )
    
    def update_status(self, new_status: RequestStatus, agent_id: str, 
                     reason: str = ""):
        """Update status with audit trail."""
        self.status_history.append({
            "from_status": self.status.value,
            "to_status": new_status.value,
            "timestamp": time.time(),
            "agent_id": agent_id,
            "reason": reason
        })
        self.status = new_status
        self.updated_at = time.time()
    
    def add_note(self, author: str, content: str, note_type: str = "info"):
        """Add a note to the request."""
        self.notes.append({
            "author": author,
            "content": content,
            "type": note_type,
            "timestamp": time.time()
        })

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
Minimal but runnable; intended as integration points for the
production-grade services from Book 2.
"""

import asyncio
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Awaitable, Callable, Optional


# ------------------------------------------------------------
# Error hierarchy
# ------------------------------------------------------------

class ProcurementError(Exception):
    """Base class for procurement-flow errors."""

class IntakeError(ProcurementError): pass
class AnalysisError(ProcurementError): pass
class ApprovalError(ProcurementError): pass


# ------------------------------------------------------------
# Result types for each workflow stage
# ------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a single ValidationRule check."""
    passed: bool
    rule_id: str
    message: str = ""
    severity: str = "warning"  # 'warning' or 'error'


@dataclass
class ValidationRule:
    """A declarative rule the intake agent applies to a request."""
    rule_id: str
    description: str
    check: Callable[["PurchaseRequest"], ValidationResult]


@dataclass
class IntakeResult:
    """Outcome of the intake stage."""
    request: "PurchaseRequest"
    validations: list[ValidationResult] = field(default_factory=list)
    ready_for_analysis: bool = False
    notes: str = ""


@dataclass
class ItemAnalysis:
    """Per-item analysis result."""
    item_name: str
    estimated_unit_price: float
    estimated_total: float
    confidence: float  # 0.0 - 1.0
    risk_flags: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Aggregate risk view for a purchase request."""
    overall_score: float  # 0-100, lower is safer
    flags: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ComplianceIssue:
    """A specific compliance concern found by ComplianceRule."""
    rule_id: str
    severity: str  # 'low', 'medium', 'high'
    description: str


@dataclass
class ComplianceRule:
    """Declarative compliance rule."""
    rule_id: str
    description: str
    check: Callable[["PurchaseRequest"], list[ComplianceIssue]]


@dataclass
class AnalysisResult:
    """Output of the AnalysisAgent."""
    request_id: str
    item_analyses: list[ItemAnalysis]
    risk: RiskAssessment
    compliance_issues: list[ComplianceIssue] = field(default_factory=list)
    estimated_total: float = 0.0
    recommended_action: str = "approve"  # 'approve', 'review', 'reject'


@dataclass
class Approver:
    """Reference to a human or systemic approver."""
    approver_id: str
    name: str
    role: str
    max_approval_amount: float


@dataclass
class ApprovalTask:
    """A pending approval routed to a specific approver."""
    task_id: str
    request_id: str
    approver: Approver
    created_at: datetime
    timeout_at: datetime
    status: str = "pending"  # 'pending', 'approved', 'rejected', 'expired'
    decision_notes: str = ""


@dataclass
class ApprovalRoutingResult:
    """Routing decision from the ApprovalAgent."""
    request_id: str
    requires_approval: bool
    tasks: list[ApprovalTask] = field(default_factory=list)
    auto_approved: bool = False
    reason: str = ""


@dataclass
class ApprovalDecisionResult:
    """Final approval verdict after routing + responses."""
    request_id: str
    approved: bool
    decided_at: datetime
    approver_chain: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class AutoApprovalResult:
    """Outcome of the auto-approval fast path."""
    eligible: bool
    reason: str = ""


@dataclass
class DelegationRule:
    """When primary approver is unavailable, who can decide?"""
    primary_approver_id: str
    delegate_approver_id: str
    valid_until: Optional[datetime] = None


@dataclass
class AuthResult:
    """Result of authenticating an agent against the IdentityService."""
    success: bool
    agent_id: str
    scopes: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class SubmissionResult:
    """Result of submitting a request to the system."""
    success: bool
    request_id: str
    message: str = ""


@dataclass
class ProcessingContext:
    """Runtime context that flows through the orchestrator stages."""
    request_id: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    actor_agent_id: str = ""
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------
# IdentityService -- agent authentication and scope checks
# ------------------------------------------------------------

class IdentityService:
    """Minimal in-process identity service.

    For production, replace with the X.509 / OIDC implementation from
    Book 2 (Chapter 1).
    """

    def __init__(self):
        self._agents: dict[str, dict[str, Any]] = {}

    def register_agent(self, agent_id: str, scopes: list[str]) -> None:
        self._agents[agent_id] = {
            "scopes": list(scopes),
            "registered_at": datetime.now(timezone.utc),
        }

    async def authenticate(self, agent_id: str, token: str) -> AuthResult:
        # Demo-only token convention: token == "valid-token-for-<id>"
        if agent_id not in self._agents:
            return AuthResult(False, agent_id, [], "unknown agent")
        if token != f"valid-token-for-{agent_id}":
            return AuthResult(False, agent_id, [], "invalid token")
        return AuthResult(True, agent_id, self._agents[agent_id]["scopes"])

    def has_scope(self, agent_id: str, scope: str) -> bool:
        return scope in self._agents.get(agent_id, {}).get("scopes", [])


# ------------------------------------------------------------
# BudgetService -- spend limit checks
# ------------------------------------------------------------

class BudgetService:
    """Tracks daily / monthly spend caps per cost center.

    For production, replace with the BudgetManager from Book 2, Ch 3.
    """

    def __init__(self):
        self._caps: dict[str, dict[str, float]] = {}
        self._spent: dict[str, dict[str, float]] = defaultdict(
            lambda: {"daily": 0.0, "monthly": 0.0}
        )

    def set_cap(self, cost_center: str, daily: float, monthly: float) -> None:
        self._caps[cost_center] = {"daily": daily, "monthly": monthly}

    def can_spend(self, cost_center: str, amount: float) -> bool:
        cap = self._caps.get(cost_center)
        if not cap:
            return False
        spent = self._spent[cost_center]
        return (spent["daily"] + amount <= cap["daily"] and
                spent["monthly"] + amount <= cap["monthly"])

    def record_spend(self, cost_center: str, amount: float) -> None:
        self._spent[cost_center]["daily"] += amount
        self._spent[cost_center]["monthly"] += amount


# ------------------------------------------------------------
# EventBus -- coordination between agents
# ------------------------------------------------------------

class EventBus:
    """In-process pub/sub bus.

    For production, replace with Kafka, NATS, or Redis Streams.
    """

    def __init__(self):
        self._subscribers: dict[str, list[Callable[[dict], Awaitable[None]]]] = (
            defaultdict(list)
        )

    def subscribe(
        self,
        topic: str,
        handler: Callable[[dict], Awaitable[None]],
    ) -> None:
        self._subscribers[topic].append(handler)

    async def publish(self, topic: str, event: dict) -> None:
        envelope = {
            "topic": topic,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
        }
        for h in self._subscribers.get(topic, []):
            try:
                await h(envelope)
            except Exception:
                # Production: log + send to DLQ; never let one subscriber
                # block others.
                pass


# ------------------------------------------------------------
# Metrics collectors
# ------------------------------------------------------------

class AgentMetrics:
    """Per-agent counters and histograms.

    For production, replace with the OpenTelemetry/Prometheus
    instrumentation from Book 2, Chapter 6.
    """

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.counters: dict[str, int] = defaultdict(int)
        self.timings: dict[str, list[float]] = defaultdict(list)

    def increment(self, name: str, by: int = 1) -> None:
        self.counters[name] += by

    def record_timing(self, name: str, seconds: float) -> None:
        self.timings[name].append(seconds)


class OrchestratorMetrics(AgentMetrics):
    """Orchestrator-level metrics including stage durations and routing."""

    def record_stage(self, stage: str, seconds: float, success: bool) -> None:
        self.record_timing(f"stage.{stage}.seconds", seconds)
        self.increment(f"stage.{stage}.{'success' if success else 'failure'}")


# ------------------------------------------------------------
# ApproverRegistry -- look up who can approve what
# ------------------------------------------------------------

class ApproverRegistry:
    """Maps approval levels to eligible approvers."""

    def __init__(self):
        self._approvers: dict[ApprovalLevel, list[Approver]] = defaultdict(list)
        self._delegations: list[DelegationRule] = []

    def register(self, level: "ApprovalLevel", approver: Approver) -> None:
        self._approvers[level].append(approver)

    def add_delegation(self, rule: DelegationRule) -> None:
        self._delegations.append(rule)

    def approvers_for(self, level: "ApprovalLevel") -> list[Approver]:
        return list(self._approvers.get(level, []))


# ------------------------------------------------------------
# VendorDatabase -- supplier catalog
# ------------------------------------------------------------

class VendorDatabase:
    """In-memory vendor catalog. Replace with a real CRM/ERP integration."""

    def __init__(self):
        self._vendors: dict[str, "Vendor"] = {}

    def add(self, vendor: "Vendor") -> None:
        self._vendors[vendor.vendor_id] = vendor

    def get(self, vendor_id: str) -> Optional["Vendor"]:
        return self._vendors.get(vendor_id)

    def list_by_category(self, category: str) -> list["Vendor"]:
        return [v for v in self._vendors.values() if category in v.categories]


# ------------------------------------------------------------
# ContractService -- pricing and contract terms lookup
# ------------------------------------------------------------

class ContractService:
    """Looks up negotiated pricing for a vendor and item.

    Production version queries the contract-management system; this
    in-memory implementation is for examples and tests.
    """

    def __init__(self):
        self._contracts: dict[tuple[str, str], dict[str, Any]] = {}

    def set_contract(
        self,
        vendor_id: str,
        item_name: str,
        unit_price: float,
        valid_until: Optional[datetime] = None,
    ) -> None:
        self._contracts[(vendor_id, item_name)] = {
            "unit_price": unit_price,
            "valid_until": valid_until,
        }

    def get_unit_price(self, vendor_id: str, item_name: str) -> Optional[float]:
        c = self._contracts.get((vendor_id, item_name))
        if not c:
            return None
        if c["valid_until"] and c["valid_until"] < datetime.now(timezone.utc):
            return None
        return c["unit_price"]


# ------------------------------------------------------------
# MockLLM -- deterministic stand-in for unit tests
# ------------------------------------------------------------

@dataclass
class MockResponse:
    """Stand-in for an LLM response. Mirrors the shape the agent code expects."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)


class MockLLM:
    """A deterministic LLM stand-in for unit tests.

    Returns canned responses in order, falling back to a default if the
    queue is exhausted. For production, replace with the real
    Anthropic / OpenAI client.
    """

    def __init__(self, responses: Optional[list[MockResponse]] = None):
        self._responses = list(responses or [])
        self._default = MockResponse(content="(mock default response)")
        self.calls: list[dict] = []

    async def complete(self, messages: list[dict], **kwargs: Any) -> MockResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        if self._responses:
            return self._responses.pop(0)
        return self._default


# ------------------------------------------------------------
# FulfillmentAgent -- closes out an approved request
# ------------------------------------------------------------

class FulfillmentAgent:
    """Final stage: emits the purchase order to the vendor system.

    Production version integrates with your ERP / supplier portals.
    """

    def __init__(
        self,
        agent_id: str,
        identity: IdentityService,
        bus: EventBus,
        metrics: AgentMetrics,
    ):
        self.agent_id = agent_id
        self.identity = identity
        self.bus = bus
        self.metrics = metrics

    async def fulfill(
        self,
        request: "PurchaseRequest",
        approval: ApprovalDecisionResult,
        ctx: ProcessingContext,
    ) -> dict:
        if not approval.approved:
            raise ProcurementError("Cannot fulfill an unapproved request.")
        self.metrics.increment("fulfilled")
        po = {
            "po_id": f"PO-{request.request_id}",
            "vendor_id": getattr(request, "preferred_vendor_id", None),
            "items": [
                {"name": it.name, "qty": it.quantity, "unit_price": it.unit_price}
                for it in request.items
            ],
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        await self.bus.publish("procurement.fulfilled", {
            "request_id": request.request_id,
            "po": po,
            "correlation_id": ctx.correlation_id,
        })
        return po

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
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.validation_rules.append(rule)
    
    async def process(self, request: PurchaseRequest, 
                     context: ProcessingContext) -> IntakeResult:
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
                errors.append(f"Requester not authorized: {auth_result.reason}")
            
            # Validate budget codes
            budget_result = await self._validate_budgets(request, context)
            errors.extend(budget_result.errors)
            warnings.extend(budget_result.warnings)
            
            # Apply custom rules
            for rule in self.validation_rules:
                rule_result = await rule.evaluate(request, context)
                if not rule_result.passed:
                    if rule.severity == "error":
                        errors.append(rule_result.message)
                    else:
                        warnings.append(rule_result.message)
            
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
                processing_time=time.time() - start_time
            )
            
            # Update request
            request.validation_result = result.to_dict()
            if result.valid:
                request.update_status(
                    RequestStatus.PENDING_ANALYSIS, 
                    self.agent_id,
                    "Validation passed"
                )
            else:
                self.metrics.increment("requests_failed_validation")
            
            return result
            
        except Exception as e:
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
    
    async def _check_authorization(self, request: PurchaseRequest,
                                   context: ProcessingContext) -> AuthResult:
        """Check if requester is authorized to make purchases."""
        # In production, check against identity service
        user = await context.identity_service.get_user(request.requester_id)
        
        if not user:
            return AuthResult(authorized=False, reason="User not found")
        
        if not user.has_permission("procurement:create"):
            return AuthResult(authorized=False, reason="Missing procurement permission")
        
        # Check spending limit
        if request.total_amount > user.spending_limit:
            return AuthResult(
                authorized=False,
                reason=f"Amount ${request.total_amount:,.2f} exceeds "
                      f"spending limit ${user.spending_limit:,.2f}"
            )
        
        return AuthResult(authorized=True)

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
    
    async def process(self, request: PurchaseRequest,
                     context: ProcessingContext) -> AnalysisResult:
        """Analyze a validated purchase request."""
        self.metrics.increment("requests_analyzed")
        start_time = time.time()
        
        try:
            request.update_status(RequestStatus.ANALYZING, self.agent_id)
            
            # Analyze each item
            item_analyses = []
            total_potential_savings = 0.0
            compliance_issues = []
            
            for item in request.items:
                item_analysis = await self._analyze_item(item, context)
                item_analyses.append(item_analysis)
                total_potential_savings += item_analysis.potential_savings
                compliance_issues.extend(item_analysis.compliance_issues)
            
            # Overall risk assessment
            risk = await self._assess_risk(request, item_analyses, compliance_issues)
            
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
                processing_time=time.time() - start_time
            )
            
            # Update request
            request.analysis_result = result.to_dict()
            request.update_status(
                RequestStatus.PENDING_APPROVAL,
                self.agent_id,
                f"Analysis complete. Risk: {risk.level}"
            )
            
            return result
            
        except Exception as e:
            self.metrics.increment("requests_errored")
            raise AnalysisError(f"Analysis failed: {e}") from e
    
    async def _analyze_item(self, item: PurchaseItem,
                           context: ProcessingContext) -> ItemAnalysis:
        """Analyze a single item."""
        analysis = ItemAnalysis(item_name=item.name)
        
        # Find alternative vendors
        vendors = await self.vendor_db.find_by_category(item.category)
        preferred = [v for v in vendors if v.preferred and v.compliance_certified]
        
        if preferred:
            best = max(preferred, key=lambda v: v.rating)
            analysis.recommended_vendor = best
            
            # Estimate savings from preferred vendor pricing
            if best.id != item.vendor_id:
                analysis.potential_savings = item.total_price * 0.05  # 5% estimated
                analysis.savings_reason = f"Switch to preferred vendor {best.name}"
        
        # Check compliance
        if item.total_price > 10000:
            if item.vendor_id:
                vendor = await self.vendor_db.get(item.vendor_id)
                if vendor and not vendor.compliance_certified:
                    analysis.compliance_issues.append(ComplianceIssue(
                        severity="high",
                        rule="VENDOR_CERTIFICATION",
                        message=f"Vendor {vendor.name} not certified for "
                               f"purchases over $10,000"
                    ))
        
        # Check for contract pricing
        contract = await context.contract_service.find_contract(
            item.vendor_id, item.category
        )
        if contract and item.unit_price > contract.unit_price:
            savings = (item.unit_price - contract.unit_price) * item.quantity
            analysis.potential_savings += savings
            analysis.savings_reason = (
                f"Contract pricing available: ${contract.unit_price:.2f} "
                f"vs ${item.unit_price:.2f}"
            )
        
        return analysis
    
    async def _assess_risk(self, request: PurchaseRequest,
                          item_analyses: list[ItemAnalysis],
                          compliance_issues: list[ComplianceIssue]) -> RiskAssessment:
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
            high_severity = [i for i in compliance_issues if i.severity == "high"]
            if high_severity:
                factors.append(f"{len(high_severity)} high-severity compliance issues")
                score += 25 * len(high_severity)
        
        # New vendor risk
        for item in request.items:
            if item.vendor_id:
                vendor = await self.vendor_db.get(item.vendor_id)
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
    
    def _generate_recommendations(self, request: PurchaseRequest,
                                 item_analyses: list[ItemAnalysis],
                                 risk: RiskAssessment,
                                 compliance_issues: list[ComplianceIssue]) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Cost savings
        total_savings = sum(a.potential_savings for a in item_analyses)
        if total_savings > request.total_amount * 0.05:  # >5% savings available
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
        self.approver_registry: ApproverRegistry = None
        self.delegation_rules: list[DelegationRule] = []
        self.metrics = AgentMetrics()
    
    async def process(self, request: PurchaseRequest,
                     context: ProcessingContext) -> ApprovalRoutingResult:
        """Determine approval routing for a request."""
        self.metrics.increment("requests_processed")
        
        required_level = request.required_approval_level
        
        # Check for auto-approval eligibility
        auto_result = await self._check_auto_approval(request, context)
        if auto_result.eligible:
            request.approval_chain.append({
                "level": "auto",
                "approver_id": "system",
                "approver_name": "Auto-Approval System",
                "decision": "approved",
                "timestamp": time.time(),
                "reason": auto_result.reason
            })
            request.update_status(
                RequestStatus.APPROVED,
                self.agent_id,
                f"Auto-approved: {auto_result.reason}"
            )
            self.metrics.increment("requests_auto_approved")
            return ApprovalRoutingResult(
                status="auto_approved",
                reason=auto_result.reason
            )
        
        # Build approval chain
        chain = await self._build_approval_chain(request, context)
        
        # Check for delegations
        chain = await self._apply_delegations(chain, context)
        
        # Notify approvers
        await self._notify_approvers(request, chain, context)
        
        request.update_status(
            RequestStatus.PENDING_APPROVAL,
            self.agent_id,
            f"Awaiting {required_level.value} approval"
        )
        
        return ApprovalRoutingResult(
            status="pending_approval",
            required_level=required_level,
            approval_chain=chain,
            estimated_time=self._estimate_approval_time(chain)
        )
    
    async def submit_decision(self, request: PurchaseRequest,
                             approver_id: str,
                             decision: str,
                             notes: str = "") -> ApprovalDecisionResult:
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
        
        # Record decision
        request.approval_chain.append({
            "level": approver.level.value,
            "approver_id": approver_id,
            "approver_name": approver.name,
            "decision": decision,
            "timestamp": time.time(),
            "notes": notes
        })
        
        if decision == "approved":
            # Check if all required approvals are complete
            if self._all_approvals_complete(request):
                request.update_status(
                    RequestStatus.APPROVED,
                    self.agent_id,
                    f"Approved by {approver.name}"
                )
                self.metrics.increment("requests_approved")
                return ApprovalDecisionResult(
                    status="fully_approved",
                    next_step="fulfillment"
                )
            else:
                # More approvals needed
                return ApprovalDecisionResult(
                    status="partially_approved",
                    next_approver=self._get_next_approver(request)
                )
        
        elif decision == "rejected":
            request.update_status(
                RequestStatus.REJECTED,
                self.agent_id,
                f"Rejected by {approver.name}: {notes}"
            )
            self.metrics.increment("requests_rejected")
            return ApprovalDecisionResult(
                status="rejected",
                reason=notes
            )
        
        elif decision == "request_info":
            request.add_note(
                approver_id,
                f"Additional information requested: {notes}",
                "info_request"
            )
            return ApprovalDecisionResult(
                status="info_requested",
                message=notes
            )
    
    async def _check_auto_approval(self, request: PurchaseRequest,
                                   context: ProcessingContext) -> AutoApprovalResult:
        """Check if request can be auto-approved."""
        # Low-value office supplies
        if (request.total_amount <= 500 and
            all(item.category == PurchaseCategory.OFFICE_SUPPLIES 
                for item in request.items)):
            return AutoApprovalResult(
                eligible=True,
                reason="Low-value office supplies (policy AUTO-001)"
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
                        reason="Preferred vendor with available budget (policy AUTO-002)"
                    )
        
        return AutoApprovalResult(eligible=False)

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
    
    def __init__(self):
        self.intake = IntakeAgent()
        self.analysis = AnalysisAgent()
        self.approval = ApprovalAgent()
        self.fulfillment = FulfillmentAgent()
        
        self.requests: dict[str, PurchaseRequest] = {}
        self.event_bus = EventBus()
        self.metrics = OrchestratorMetrics()
        
        # Set up tracing
        self.tracer = trace.get_tracer("procurement-orchestrator")
    
    async def submit_request(self, request: PurchaseRequest,
                            context: ProcessingContext) -> SubmissionResult:
        """Submit a new purchase request for processing."""
        with self.tracer.start_as_current_span(
            "submit_request",
            attributes={
                "request.id": request.id,
                "request.amount": request.total_amount,
                "request.department": request.requester_department
            }
        ) as span:
            
            # Store request
            self.requests[request.id] = request
            self.metrics.increment("requests_submitted")
            
            await self.event_bus.emit("request.submitted", {
                "request_id": request.id,
                "requester": request.requester_id,
                "amount": request.total_amount,
                "timestamp": time.time()
            })
            
            try:
                # Phase 1: Intake
                with self.tracer.start_span("intake") as intake_span:
                    intake_result = await self.intake.process(request, context)
                    intake_span.set_attribute("valid", intake_result.valid)
                
                if not intake_result.valid:
                    await self.event_bus.emit("request.validation_failed", {
                        "request_id": request.id,
                        "errors": intake_result.errors
                    })
                    return SubmissionResult(
                        status="validation_failed",
                        request_id=request.id,
                        errors=intake_result.errors,
                        warnings=intake_result.warnings
                    )
                
                await self.event_bus.emit("request.validated", {
                    "request_id": request.id
                })
                
                # Phase 2: Analysis
                with self.tracer.start_span("analysis") as analysis_span:
                    analysis_result = await self.analysis.process(request, context)
                    analysis_span.set_attribute("risk_level", analysis_result.risk_level)
                
                await self.event_bus.emit("request.analyzed", {
                    "request_id": request.id,
                    "risk_level": analysis_result.risk_level,
                    "potential_savings": analysis_result.total_potential_savings
                })
                
                # Phase 3: Approval routing
                with self.tracer.start_span("approval_routing"):
                    approval_result = await self.approval.process(request, context)
                
                if approval_result.status == "auto_approved":
                    await self.event_bus.emit("request.auto_approved", {
                        "request_id": request.id,
                        "reason": approval_result.reason
                    })
                    
                    # Auto-approved requests go straight to fulfillment
                    fulfillment_result = await self.fulfillment.process(
                        request, context
                    )
                    
                    await self.event_bus.emit("request.fulfilled", {
                        "request_id": request.id,
                        "order_id": fulfillment_result.order_id
                    })
                    
                    return SubmissionResult(
                        status="completed",
                        request_id=request.id,
                        auto_approved=True,
                        order=fulfillment_result.to_dict()
                    )
                
                # Request needs human approval
                await self.event_bus.emit("request.pending_approval", {
                    "request_id": request.id,
                    "required_level": approval_result.required_level.value,
                    "estimated_time": approval_result.estimated_time
                })
                
                self.metrics.increment("requests_pending_approval")
                
                return SubmissionResult(
                    status="pending_approval",
                    request_id=request.id,
                    approval_info=approval_result.to_dict(),
                    analysis_summary={
                        "risk_level": analysis_result.risk_level,
                        "recommendations": analysis_result.recommendations
                    }
                )
                
            except Exception as e:
                span.record_exception(e)
                self.metrics.increment("requests_errored")
                
                await self.event_bus.emit("request.error", {
                    "request_id": request.id,
                    "error": str(e)
                })
                
                raise ProcurementError(
                    f"Failed to process request {request.id}: {e}"
                ) from e
    
    async def handle_approval_decision(self, request_id: str,
                                       approver_id: str,
                                       decision: str,
                                       notes: str = "") -> ApprovalDecisionResult:
        """Handle an approval decision from a human approver."""
        if request_id not in self.requests:
            raise ProcurementError(f"Request not found: {request_id}")
        
        request = self.requests[request_id]
        
        with self.tracer.start_as_current_span(
            "handle_approval",
            attributes={
                "request.id": request_id,
                "approver.id": approver_id,
                "decision": decision
            }
        ):
            result = await self.approval.submit_decision(
                request, approver_id, decision, notes
            )
            
            await self.event_bus.emit("approval.decision", {
                "request_id": request_id,
                "approver_id": approver_id,
                "decision": decision,
                "result_status": result.status
            })
            
            if result.status == "fully_approved":
                # Proceed to fulfillment
                context = ProcessingContext()  # Build context
                fulfillment_result = await self.fulfillment.process(request, context)
                
                await self.event_bus.emit("request.fulfilled", {
                    "request_id": request_id,
                    "order_id": fulfillment_result.order_id
                })
                
                return ApprovalDecisionResult(
                    status="fulfilled",
                    order=fulfillment_result.to_dict()
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
                    pending.append(ApprovalTask(
                        request_id=request.id,
                        requester=request.requester_name,
                        department=request.requester_department,
                        amount=request.total_amount,
                        urgency=request.urgency,
                        categories=[c.value for c in request.categories],
                        risk_level=request.analysis_result.get("risk_level", "unknown"),
                        recommendations=request.analysis_result.get("recommendations", []),
                        compliance_issues=request.analysis_result.get("compliance_issues", []),
                        submitted_at=request.created_at,
                        waiting_time=time.time() - request.created_at
                    ))
        
        # Sort by urgency then waiting time
        urgency_order = {"urgent": 0, "high": 1, "normal": 2, "low": 3}
        pending.sort(key=lambda t: (urgency_order[t.urgency], -t.waiting_time))
        
        return pending
    
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
            status_history=request.status_history
        )
    
    async def approve(self, request_id: str, approver_id: str,
                     notes: str = "") -> ApprovalDecisionResult:
        """Approve a request."""
        return await self.orchestrator.handle_approval_decision(
            request_id, approver_id, "approved", notes
        )
    
    async def reject(self, request_id: str, approver_id: str,
                    reason: str) -> ApprovalDecisionResult:
        """Reject a request."""
        return await self.orchestrator.handle_approval_decision(
            request_id, approver_id, "rejected", reason
        )
    
    async def request_info(self, request_id: str, approver_id: str,
                          question: str) -> ApprovalDecisionResult:
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
        self._setup_event_handlers()
    
    def _setup_event_handlers(self):
        """Set up event handlers for dashboard updates."""
        self.orchestrator.event_bus.subscribe(
            "request.*",
            self._handle_request_event
        )
        self.orchestrator.event_bus.subscribe(
            "approval.*",
            self._handle_approval_event
        )
    
    def get_summary(self) -> dict:
        """Get current system summary."""
        requests = list(self.orchestrator.requests.values())
        
        return {
            "total_requests": len(requests),
            "by_status": self._count_by_status(requests),
            "total_value": sum(r.total_amount for r in requests),
            "pending_approval_value": sum(
                r.total_amount for r in requests
                if r.status == RequestStatus.PENDING_APPROVAL
            ),
            "average_processing_time": self._calc_avg_processing_time(requests),
            "auto_approval_rate": self._calc_auto_approval_rate(),
            "top_requesters": self._get_top_requesters(requests),
            "top_categories": self._get_top_categories(requests)
        }
    
    def get_approval_metrics(self) -> dict:
        """Get approval workflow metrics."""
        return {
            "pending_count": self.orchestrator.metrics.get("requests_pending_approval"),
            "approved_today": self._get_decisions_today("approved"),
            "rejected_today": self._get_decisions_today("rejected"),
            "average_approval_time": self._calc_avg_approval_time(),
            "oldest_pending": self._get_oldest_pending(),
            "by_level": self._get_approvals_by_level()
        }

# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

async def main():
    """Demonstration of the procurement system."""
    
    # Initialize
    orchestrator = ProcurementOrchestrator()
    
    # Configure vendors
    orchestrator.analysis.vendor_db = VendorDatabase()
    orchestrator.analysis.vendor_db.add(Vendor(
        id="dell-001",
        name="Dell Technologies",
        categories=[PurchaseCategory.HARDWARE],
        rating=4.8,
        compliance_certified=True,
        preferred=True
    ))
    
    # Configure approvers
    orchestrator.approval.approver_registry = ApproverRegistry()
    orchestrator.approval.approver_registry.register(Approver(
        id="mgr-001",
        name="Jane Smith",
        email="jane.smith@company.com",
        level=ApprovalLevel.MANAGER,
        departments=["Engineering", "Product"]
    ))
    
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
                justification="New team members starting Q2"
            )
        ],
        business_justification="Expanding engineering team by 5 developers",
        urgency="high"
    )
    
    context = ProcessingContext(
        identity_service=IdentityService(),
        budget_service=BudgetService(),
        contract_service=ContractService()
    )
    
    # Submit and process
    result = await orchestrator.submit_request(request, context)
    print(f"Submission result: {result.status}")
    
    if result.status == "pending_approval":
        # Show pending approvals
        pending = approval_ui.get_pending_approvals("mgr-001")
        print(f"Pending approvals: {len(pending)}")
        
        # Approve the request
        decision = await approval_ui.approve(
            request.id,
            "mgr-001",
            "Approved for Q2 expansion"
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
import pytest
from unittest.mock import AsyncMock
from src.procurement.agents import AnalysisAgent
from src.procurement.models import PurchaseRequest, PurchaseItem
from src.testing.mock_llm import MockLLM, MockResponse


@pytest.fixture
def mock_llm():
    llm = MockLLM()
    llm.add_response(MockResponse(
        content="Analysis complete. Found 15% savings opportunity "
                "by switching to preferred vendor TechSupply Inc."
    ))
    return llm


@pytest.fixture
def sample_request():
    return PurchaseRequest(
        id="REQ-001",
        requester_id="user_123",
        items=[PurchaseItem(
            description="Laptop computers",
            quantity=10,
            unit_price=1200.00,
            category="hardware"
        )]
    )


async def test_analysis_agent_identifies_savings(mock_llm, sample_request):
    """AnalysisAgent should identify cost savings opportunities."""
    agent = AnalysisAgent(llm=mock_llm)

    result = await agent.analyze(sample_request)

    assert result.savings_identified > 0
    assert any("preferred vendor" in r.lower() for r in result.recommendations)


async def test_analysis_agent_flags_compliance_issues(mock_llm):
    """AnalysisAgent should flag requests missing required fields."""
    mock_llm.set_default_response(MockResponse(
        content="Compliance issue: Missing cost center code."
    ))
    agent = AnalysisAgent(llm=mock_llm)

    incomplete_request = PurchaseRequest(
        id="REQ-002",
        requester_id="user_456",
        items=[],  # Empty items list
        cost_center=None  # Missing required field
    )

    result = await agent.analyze(incomplete_request)

    assert len(result.compliance_issues) > 0

# ============================================================================
# Block 13 (chapter listing #13)
# ============================================================================

# tests/integration/test_orchestrator.py
import pytest
from src.procurement.orchestrator import ProcurementOrchestrator
from src.procurement.models import (
    PurchaseRequest, RequestStatus, ApprovalLevel
)


@pytest.fixture
def orchestrator():
    return ProcurementOrchestrator(config=test_config)


def create_test_request(amount: float, category: str) -> PurchaseRequest:
    """Helper to create test requests with specified amount."""
    return PurchaseRequest(
        id=f"REQ-{uuid.uuid4().hex[:8]}",
        requester_id="test_user",
        items=[PurchaseItem(
            description=f"Test {category} purchase",
            quantity=1,
            unit_price=amount,
            category=category
        )]
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
import pytest


@pytest.mark.parametrize("amount,expected_level", [
    (100, "auto"),           # Under $500
    (499, "auto"),           # Just under threshold
    (500, "auto"),           # At threshold (inclusive)
    (501, "manager"),        # Just over threshold
    (3000, "manager"),       # Mid-range manager
    (5000, "manager"),       # At manager limit
    (5001, "director"),      # Just over manager limit
    (15000, "director"),     # Mid-range director
    (25000, "director"),     # At director limit
    (25001, "vp"),           # Just over director limit
    (75000, "vp"),           # Mid-range VP
    (100000, "vp"),          # At VP limit
    (100001, "executive"),   # Over VP limit
])
async def test_approval_routing_by_amount(orchestrator, amount, expected_level):
    """Verify correct approval routing based on purchase amount."""
    request = create_test_request(amount=amount, category="office_supplies")

    result = await orchestrator.process(request)

    assert result.approval_level.value == expected_level, \
        f"Amount ${amount} should route to {expected_level}, got {result.approval_level.value}"
