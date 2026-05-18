"""
Caching Strategies for Agents

Code listings from Chapter 04, Book 3:
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
# Block 1 (chapter block #1) — Python fragment (incomplete, depends on surrounding context)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_1_listing = r"""
Daily requests: 10,000
Average tokens per request: 2,000 input + 500 output
Daily token usage: 25 million tokens
Daily cost (at $15/$45 per million): ~$52
Monthly cost: ~$1,560
"""

# ============================================================================
# Block 2 (chapter block #2) — Python fragment (incomplete, depends on surrounding context)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_2_listing = r"""
Cache hits: 6,000 (served from cache)
Cache misses: 4,000 (require LLM calls)
Daily cost: ~$21 (60% reduction)
Monthly cost: ~$624
Monthly savings: ~$936
"""

# ============================================================================
# Block 3 (chapter listing #3)
# ============================================================================

from typing import Optional, Any
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class CachedResponse:
    """Represents a cached LLM response with metadata."""
    response: str
    model: str
    created_at: datetime
    token_count: int
    cache_key: str
    
    def is_expired(self, ttl: timedelta) -> bool:
        return datetime.now(timezone.utc) - self.created_at > ttl


class LLMResponseCache:
    """
    Simple exact-match cache for LLM responses.
    
    This cache stores responses keyed by a hash of the prompt
    and relevant parameters. It does not handle semantic similarity;
    that is covered by SemanticCache later in this chapter.
    """
    
    def __init__(
        self,
        backend: "CacheBackend",
        default_ttl: timedelta = timedelta(hours=24)
    ):
        self.backend = backend
        self.default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def _generate_key(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a deterministic cache key from request parameters.
        
        Key includes model and temperature because different settings
        produce different outputs. System prompt is included if present.
        """
        key_data = {
            "prompt": prompt.strip(),
            "model": model,
            "temperature": temperature,
            "system_prompt": system_prompt.strip() if system_prompt else None
        }
        
        # Use canonical JSON serialization for deterministic hashing
        canonical = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        return f"llm:{hashlib.sha256(canonical.encode()).hexdigest()}"
    
    def get(
        self,
        prompt: str,
        model: str,
        temperature: float,
        system_prompt: Optional[str] = None
    ) -> Optional[CachedResponse]:
        """
        Retrieve a cached response if available and not expired.
        """
        key = self._generate_key(prompt, model, temperature, system_prompt)
        
        data = self.backend.get(key)
        if data is None:
            self._misses += 1
            return None
        
        cached = CachedResponse(**data)
        if cached.is_expired(self.default_ttl):
            self.backend.delete(key)
            self._misses += 1
            return None
        
        self._hits += 1
        return cached
    
    def set(
        self,
        prompt: str,
        model: str,
        temperature: float,
        response: str,
        token_count: int,
        system_prompt: Optional[str] = None,
        ttl: Optional[timedelta] = None
    ) -> str:
        """
        Store a response in the cache. Returns the cache key.
        """
        key = self._generate_key(prompt, model, temperature, system_prompt)
        
        cached = CachedResponse(
            response=response,
            model=model,
            created_at=datetime.now(timezone.utc),
            token_count=token_count,
            cache_key=key
        )
        
        effective_ttl = ttl or self.default_ttl
        self.backend.set(key, cached.__dict__, ttl=effective_ttl)
        
        return key
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

# ============================================================================
# Block 4 (chapter listing #4)
# ============================================================================

from typing import List
import numpy as np


class EmbeddingCache:
    """
    Cache for embedding vectors.
    
    Embeddings are deterministic and never change for a given model,
    making them perfect cache candidates. The only reason to expire
    cached embeddings is when the embedding model itself changes.
    """
    
    def __init__(
        self,
        backend: "CacheBackend",
        model_version: str,
        default_ttl: timedelta = timedelta(days=30)
    ):
        self.backend = backend
        self.model_version = model_version
        self.default_ttl = default_ttl
    
    def _generate_key(self, text: str) -> str:
        """
        Generate cache key from text and model version.
        
        Including model version ensures embeddings are invalidated
        when the embedding model changes.
        """
        text_hash = hashlib.sha256(text.strip().encode()).hexdigest()
        return f"emb:{self.model_version}:{text_hash}"
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding vector."""
        key = self._generate_key(text)
        data = self.backend.get(key)
        
        if data is None:
            return None
        
        return np.array(data["vector"], dtype=np.float32)
    
    def get_many(self, texts: List[str]) -> dict[str, Optional[np.ndarray]]:
        """
        Batch retrieve embeddings for multiple texts.
        
        Returns a dict mapping text to embedding (or None if not cached).
        This enables efficient batch lookups before computing missing embeddings.
        """
        keys = {text: self._generate_key(text) for text in texts}
        results = self.backend.get_many(list(keys.values()))
        
        output = {}
        for text, key in keys.items():
            if key in results and results[key] is not None:
                output[text] = np.array(results[key]["vector"], dtype=np.float32)
            else:
                output[text] = None
        
        return output
    
    def set(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding vector."""
        key = self._generate_key(text)
        self.backend.set(
            key,
            {"vector": embedding.tolist(), "model": self.model_version},
            ttl=self.default_ttl
        )
    
    def set_many(self, embeddings: dict[str, np.ndarray]) -> None:
        """Batch store multiple embeddings."""
        items = {}
        for text, vector in embeddings.items():
            key = self._generate_key(text)
            items[key] = {
                "vector": vector.tolist(),
                "model": self.model_version
            }
        
        self.backend.set_many(items, ttl=self.default_ttl)

# ============================================================================
# Block 5 (chapter listing #5)
# ============================================================================

from enum import Enum
from typing import Callable, Dict


class CachePolicy(Enum):
    """Cache policies for different tool types."""
    
    # Never cache - result changes constantly
    NO_CACHE = "no_cache"
    
    # Cache for seconds - result changes frequently
    SHORT_TTL = "short_ttl"  # 30-60 seconds
    
    # Cache for minutes - result relatively stable
    MEDIUM_TTL = "medium_ttl"  # 5-15 minutes
    
    # Cache for hours - result stable
    LONG_TTL = "long_ttl"  # 1-24 hours
    
    # Cache until explicit invalidation
    UNTIL_INVALIDATED = "until_invalidated"


TOOL_CACHE_POLICIES: Dict[str, CachePolicy] = {
    # Real-time data - never cache
    "get_stock_price": CachePolicy.NO_CACHE,
    "get_account_balance": CachePolicy.NO_CACHE,
    "check_service_status": CachePolicy.NO_CACHE,
    
    # Frequently updated - short TTL
    "get_weather": CachePolicy.SHORT_TTL,
    "get_exchange_rate": CachePolicy.SHORT_TTL,
    "list_active_orders": CachePolicy.SHORT_TTL,
    
    # Semi-stable data - medium TTL
    "search_products": CachePolicy.MEDIUM_TTL,
    "get_user_preferences": CachePolicy.MEDIUM_TTL,
    "list_team_members": CachePolicy.MEDIUM_TTL,
    
    # Stable data - long TTL
    "get_company_info": CachePolicy.LONG_TTL,
    "get_product_details": CachePolicy.LONG_TTL,
    "get_documentation": CachePolicy.LONG_TTL,
    
    # Static data - cache until invalidation
    "get_country_codes": CachePolicy.UNTIL_INVALIDATED,
    "get_timezone_info": CachePolicy.UNTIL_INVALIDATED,
    "get_schema_definition": CachePolicy.UNTIL_INVALIDATED,
}


class ToolResultCache:
    """
    Cache for tool execution results with policy-based TTL.
    """
    
    TTL_MAPPING = {
        CachePolicy.NO_CACHE: timedelta(seconds=0),
        CachePolicy.SHORT_TTL: timedelta(seconds=45),
        CachePolicy.MEDIUM_TTL: timedelta(minutes=10),
        CachePolicy.LONG_TTL: timedelta(hours=4),
        CachePolicy.UNTIL_INVALIDATED: timedelta(days=365),
    }
    
    def __init__(
        self,
        backend: "CacheBackend",
        policies: Dict[str, CachePolicy]
    ):
        self.backend = backend
        self.policies = policies
    
    def _generate_key(self, tool_name: str, arguments: dict) -> str:
        """Generate cache key from tool name and arguments."""
        args_canonical = json.dumps(arguments, sort_keys=True)
        args_hash = hashlib.sha256(args_canonical.encode()).hexdigest()[:16]
        return f"tool:{tool_name}:{args_hash}"
    
    def get(self, tool_name: str, arguments: dict) -> Optional[Any]:
        """Retrieve cached tool result if available."""
        policy = self.policies.get(tool_name, CachePolicy.NO_CACHE)
        
        if policy == CachePolicy.NO_CACHE:
            return None
        
        key = self._generate_key(tool_name, arguments)
        return self.backend.get(key)
    
    def set(self, tool_name: str, arguments: dict, result: Any) -> None:
        """Store tool result with appropriate TTL based on policy."""
        policy = self.policies.get(tool_name, CachePolicy.NO_CACHE)
        
        if policy == CachePolicy.NO_CACHE:
            return
        
        key = self._generate_key(tool_name, arguments)
        ttl = self.TTL_MAPPING[policy]
        self.backend.set(key, result, ttl=ttl)
    
    def invalidate(self, tool_name: str, arguments: Optional[dict] = None) -> int:
        """
        Invalidate cached results for a tool.
        
        If arguments provided, invalidates specific result.
        If arguments is None, invalidates all cached results for this tool.
        """
        if arguments is not None:
            key = self._generate_key(tool_name, arguments)
            self.backend.delete(key)
            return 1
        
        # Pattern-based deletion for all tool results
        pattern = f"tool:{tool_name}:*"
        return self.backend.delete_pattern(pattern)

# ============================================================================
# Block 6 (chapter listing #6)
# ============================================================================

from typing import Any, Optional, Tuple, FrozenSet
import hashlib
import json
from abc import ABC, abstractmethod


class CacheKeyGenerator:
    """
    Generates deterministic, collision-resistant cache keys.
    
    This class handles the complexity of creating consistent keys
    from varied input types, including nested structures, optional
    parameters, and non-serializable objects.
    """
    
    def __init__(self, namespace: str = ""):
        """
        Initialize with optional namespace prefix.
        
        Namespaces prevent key collisions between different cache
        domains (e.g., "llm" vs "embedding" vs "tool").
        """
        self.namespace = namespace
    
    def generate(
        self,
        *args,
        include_none: bool = False,
        **kwargs
    ) -> str:
        """
        Generate a cache key from arbitrary arguments.
        
        Args:
            *args: Positional values to include in key
            include_none: Whether to include None values (default False)
            **kwargs: Named values to include in key
        
        Returns:
            Deterministic cache key string
        """
        # Normalize all inputs to a canonical form
        canonical_args = tuple(self._normalize(arg) for arg in args)
        
        # Sort kwargs for deterministic ordering
        canonical_kwargs = {
            k: self._normalize(v)
            for k, v in sorted(kwargs.items())
            if include_none or v is not None
        }
        
        # Create canonical representation
        key_data = {
            "args": canonical_args,
            "kwargs": canonical_kwargs
        }
        
        # Serialize and hash
        serialized = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
        hash_value = hashlib.sha256(serialized.encode()).hexdigest()
        
        if self.namespace:
            return f"{self.namespace}:{hash_value}"
        return hash_value
    
    def _normalize(self, value: Any) -> Any:
        """
        Normalize a value to a canonical, JSON-serializable form.
        
        Handles strings, numbers, booleans, None, lists, dicts, sets,
        and common types like datetime and numpy arrays.
        """
        if value is None:
            return None
        
        if isinstance(value, (bool, int, float)):
            return value
        
        if isinstance(value, str):
            # Normalize whitespace
            return " ".join(value.split())
        
        if isinstance(value, bytes):
            return value.hex()
        
        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]
        
        if isinstance(value, dict):
            return {
                str(k): self._normalize(v)
                for k, v in sorted(value.items())
            }
        
        if isinstance(value, (set, frozenset)):
            return sorted(self._normalize(item) for item in value)
        
        if isinstance(value, datetime):
            return value.isoformat()
        
        # Handle numpy arrays
        if hasattr(value, 'tolist'):
            return self._normalize(value.tolist())
        
        # Fallback: convert to string
        return str(value)
    
    def generate_composite(
        self,
        components: dict[str, Any],
        version: str = "v1"
    ) -> str:
        """
        Generate a composite key with explicit component labeling.
        
        Useful when you want the key structure to be self-documenting
        and versioned for future changes.
        """
        versioned = {"_version": version, **components}
        return self.generate(**versioned)


class LLMKeyGenerator(CacheKeyGenerator):
    """
    Specialized key generator for LLM requests.
    
    Handles the specific parameters that affect LLM output:
    model, temperature, system prompt, and user prompt.
    """
    
    def __init__(self):
        super().__init__(namespace="llm")
    
    def for_completion(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None
    ) -> str:
        """
        Generate key for a completion request.
        
        Note: max_tokens is included because it can affect output
        (truncation). stop_sequences affect where generation stops.
        """
        return self.generate(
            prompt=prompt,
            model=model,
            temperature=round(temperature, 2),  # Normalize precision
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            stop_sequences=tuple(stop_sequences) if stop_sequences else None
        )
    
    def for_chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate key for a chat completion request.
        
        Messages are normalized to ensure consistent ordering
        of message fields.
        """
        normalized_messages = [
            {
                "role": msg["role"],
                "content": msg["content"]
            }
            for msg in messages
        ]
        
        return self.generate(
            messages=normalized_messages,
            model=model,
            temperature=round(temperature, 2),
            system_prompt=system_prompt
        )

# ============================================================================
# Block 7 (chapter listing #7)
# ============================================================================

class SemanticKeyGenerator(CacheKeyGenerator):
    """
    Key generator that groups semantically similar queries.
    
    Instead of hashing the raw query, this generator uses
    embedding-based bucketing to map similar queries to
    the same cache key.
    """
    
    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        similarity_threshold: float = 0.95,
        bucket_size: int = 1000
    ):
        super().__init__(namespace="semantic")
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.bucket_size = bucket_size
    
    def for_query(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Tuple[str, np.ndarray]:
        """
        Generate semantic key for a query.
        
        Returns both the key and the embedding vector, since
        the embedding is needed for similarity matching.
        """
        # Combine query and context for embedding
        text = f"{query}\n{context}" if context else query
        embedding = self.embedding_model.embed(text)
        
        # Quantize embedding to create bucket key
        # This is a simplified approach - production systems
        # use more sophisticated locality-sensitive hashing
        bucket = self._quantize_embedding(embedding)
        
        return (
            self.generate(bucket=bucket, context_hash=hash(context) if context else None),
            embedding
        )
    
    def _quantize_embedding(self, embedding: np.ndarray) -> tuple:
        """
        Quantize embedding to a discrete bucket.
        
        This is a naive implementation using sign-based quantization.
        Production systems should use proper LSH (locality-sensitive hashing).
        """
        # Simple sign-based quantization
        signs = tuple((embedding > 0).astype(int).tolist())
        return signs[:64]  # Use first 64 dimensions for bucketing

# ============================================================================
# Block 8 (chapter listing #8)
# ============================================================================

from collections import defaultdict
from typing import NamedTuple
import time


class AccessStats(NamedTuple):
    """Statistics for a cached item."""
    access_count: int
    last_access: float
    created_at: float
    invalidation_count: int


class AdaptiveTTLManager:
    """
    Dynamically adjusts TTL based on access patterns and invalidation rates.
    
    Items that are frequently accessed get longer TTLs (since cache hits
    are valuable). Items that are frequently invalidated get shorter TTLs
    (since they're likely to change again).
    """
    
    def __init__(
        self,
        min_ttl: timedelta = timedelta(seconds=30),
        max_ttl: timedelta = timedelta(hours=24),
        base_ttl: timedelta = timedelta(minutes=15)
    ):
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl
        self.base_ttl = base_ttl
        self._stats: Dict[str, AccessStats] = {}
    
    def record_access(self, key: str) -> None:
        """Record a cache access (hit or miss)."""
        now = time.time()
        
        if key in self._stats:
            stats = self._stats[key]
            self._stats[key] = AccessStats(
                access_count=stats.access_count + 1,
                last_access=now,
                created_at=stats.created_at,
                invalidation_count=stats.invalidation_count
            )
        else:
            self._stats[key] = AccessStats(
                access_count=1,
                last_access=now,
                created_at=now,
                invalidation_count=0
            )
    
    def record_invalidation(self, key: str) -> None:
        """Record when a cached item is invalidated."""
        if key in self._stats:
            stats = self._stats[key]
            self._stats[key] = AccessStats(
                access_count=stats.access_count,
                last_access=stats.last_access,
                created_at=stats.created_at,
                invalidation_count=stats.invalidation_count + 1
            )
    
    def get_ttl(self, key: str) -> timedelta:
        """
        Calculate recommended TTL for a key based on its access pattern.
        
        High access frequency + low invalidation rate = longer TTL
        Low access frequency + high invalidation rate = shorter TTL
        """
        if key not in self._stats:
            return self.base_ttl
        
        stats = self._stats[key]
        age = time.time() - stats.created_at
        
        if age < 60:  # Not enough data yet
            return self.base_ttl
        
        # Calculate access rate (accesses per hour)
        access_rate = (stats.access_count / age) * 3600
        
        # Calculate invalidation rate (invalidations per hour)
        invalidation_rate = (stats.invalidation_count / age) * 3600 if stats.invalidation_count > 0 else 0
        
        # Adjust TTL based on rates
        # High access rate -> extend TTL (more value from caching)
        # High invalidation rate -> reduce TTL (data changes often)
        
        access_multiplier = min(2.0, 1.0 + (access_rate / 100))  # Up to 2x for high access
        invalidation_multiplier = max(0.25, 1.0 - (invalidation_rate * 0.5))  # Down to 0.25x for high invalidation
        
        adjusted_ttl = self.base_ttl * access_multiplier * invalidation_multiplier
        
        # Clamp to min/max bounds
        return max(self.min_ttl, min(self.max_ttl, adjusted_ttl))


class TTLPolicy:
    """
    Declarative TTL policies for different content types.
    """
    
    def __init__(self):
        self._policies: Dict[str, Callable[[dict], timedelta]] = {}
    
    def register(
        self,
        pattern: str,
        ttl_func: Callable[[dict], timedelta]
    ) -> None:
        """
        Register a TTL policy for keys matching a pattern.
        
        ttl_func receives the cached data and returns appropriate TTL.
        """
        self._policies[pattern] = ttl_func
    
    def get_ttl(self, key: str, data: dict) -> timedelta:
        """Get TTL for a key based on registered policies."""
        for pattern, ttl_func in self._policies.items():
            if key.startswith(pattern):
                return ttl_func(data)
        
        return timedelta(hours=1)  # Default TTL


ttl_policy = TTLPolicy()

# Embeddings: long TTL, but include model version check
ttl_policy.register(
    "emb:",
    lambda data: timedelta(days=30)
)

# LLM responses: TTL based on temperature
ttl_policy.register(
    "llm:",
    lambda data: timedelta(hours=24) if data.get("temperature", 0) == 0 else timedelta(seconds=0)
)

# Tool results: TTL based on tool type (would look up policy)
ttl_policy.register(
    "tool:",
    lambda data: timedelta(minutes=10)  # Default for tools
)

# ============================================================================
# Block 9 (chapter listing #9)
# ============================================================================

from typing import Set, List, Callable
from abc import ABC, abstractmethod
import threading
import weakref


class InvalidationStrategy(ABC):
    """Base class for cache invalidation strategies."""
    
    @abstractmethod
    def should_invalidate(self, key: str, data: Any, metadata: dict) -> bool:
        """Check if a cache entry should be invalidated."""
        pass


class TTLInvalidation(InvalidationStrategy):
    """Time-based invalidation."""
    
    def should_invalidate(self, key: str, data: Any, metadata: dict) -> bool:
        if "expires_at" not in metadata:
            return False
        return time.time() > metadata["expires_at"]


class VersionInvalidation(InvalidationStrategy):
    """Version-based invalidation."""
    
    def __init__(self):
        self._versions: Dict[str, int] = {}
    
    def set_version(self, namespace: str, version: int) -> None:
        """Set current version for a namespace."""
        self._versions[namespace] = version
    
    def get_version(self, namespace: str) -> int:
        """Get current version for a namespace."""
        return self._versions.get(namespace, 0)
    
    def should_invalidate(self, key: str, data: Any, metadata: dict) -> bool:
        namespace = key.split(":")[0] if ":" in key else "default"
        current_version = self.get_version(namespace)
        cached_version = metadata.get("version", 0)
        return cached_version < current_version


class DependencyTracker:
    """
    Tracks dependencies between cache entries for cascading invalidation.
    
    When a dependency is invalidated, all dependents are also invalidated.
    """
    
    def __init__(self):
        self._dependencies: Dict[str, Set[str]] = defaultdict(set)
        self._dependents: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
    
    def add_dependency(self, key: str, depends_on: str) -> None:
        """
        Register that 'key' depends on 'depends_on'.
        
        When 'depends_on' is invalidated, 'key' should also be invalidated.
        """
        with self._lock:
            self._dependencies[key].add(depends_on)
            self._dependents[depends_on].add(key)
    
    def get_cascade(self, key: str) -> Set[str]:
        """
        Get all keys that should be invalidated when 'key' is invalidated.
        
        Returns the transitive closure of all dependent keys.
        """
        with self._lock:
            to_invalidate = set()
            queue = [key]
            
            while queue:
                current = queue.pop()
                if current in to_invalidate:
                    continue
                
                to_invalidate.add(current)
                queue.extend(self._dependents.get(current, set()))
            
            return to_invalidate
    
    def remove_key(self, key: str) -> None:
        """Remove a key and its dependency relationships."""
        with self._lock:
            # Remove from dependents of its dependencies
            for dep in self._dependencies.get(key, set()):
                self._dependents[dep].discard(key)
            
            # Remove from dependencies of its dependents
            for dependent in self._dependents.get(key, set()):
                self._dependencies[dependent].discard(key)
            
            # Remove the key itself
            self._dependencies.pop(key, None)
            self._dependents.pop(key, None)


class CacheInvalidator:
    """
    Coordinates cache invalidation across multiple strategies.
    """
    
    def __init__(
        self,
        backend: "CacheBackend",
        strategies: List[InvalidationStrategy],
        dependency_tracker: Optional[DependencyTracker] = None
    ):
        self.backend = backend
        self.strategies = strategies
        self.dependency_tracker = dependency_tracker or DependencyTracker()
        self._listeners: List[Callable[[str], None]] = []
    
    def on_invalidation(self, callback: Callable[[str], None]) -> None:
        """Register a callback for invalidation events."""
        self._listeners.append(callback)
    
    def check_and_invalidate(self, key: str) -> bool:
        """
        Check if a key should be invalidated based on registered strategies.
        
        Returns True if the key was invalidated.
        """
        data = self.backend.get(key)
        if data is None:
            return False
        
        metadata = self.backend.get_metadata(key) or {}
        
        for strategy in self.strategies:
            if strategy.should_invalidate(key, data, metadata):
                self.invalidate(key)
                return True
        
        return False
    
    def invalidate(self, key: str, cascade: bool = True) -> int:
        """
        Invalidate a key and optionally cascade to dependents.
        
        Returns the number of keys invalidated.
        """
        if cascade:
            keys_to_invalidate = self.dependency_tracker.get_cascade(key)
        else:
            keys_to_invalidate = {key}
        
        count = 0
        for k in keys_to_invalidate:
            if self.backend.delete(k):
                count += 1
                self.dependency_tracker.remove_key(k)
                
                # Notify listeners
                for listener in self._listeners:
                    try:
                        listener(k)
                    except Exception:
                        pass  # Don't let listener errors break invalidation
        
        return count
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching a pattern."""
        return self.backend.delete_pattern(pattern)

# ============================================================================
# Block 10 (chapter listing #10)
# ============================================================================

"""
Code Navigation (line numbers are approximate):
- CacheBackend ABC ... ~15
- InMemoryBackend ... ~115
- RedisBackend ... ~175
- TieredCacheBackend ... ~300
- CacheManager (main class) ... ~370
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List, Union
from datetime import timedelta
import redis
from redis.cluster import RedisCluster
import json
import threading
from functools import lru_cache


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value from the cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Store a value in the cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys. Override for efficiency."""
        return {key: self.get(key) for key in keys}
    
    def set_many(self, items: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool:
        """Batch set multiple keys. Override for efficiency."""
        return all(self.set(key, value, ttl) for key, value in items.items())
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a pattern. Override for efficiency."""
        raise NotImplementedError("Pattern deletion not supported")
    
    def get_metadata(self, key: str) -> Optional[dict]:
        """Get metadata for a cached item."""
        return None


class InMemoryBackend(CacheBackend):
    """
    Simple in-memory cache backend.
    
    Useful for local development and as L1 in a tiered cache.
    Not suitable for production multi-instance deployments.
    """
    
    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, tuple[Any, float]] = {}  # (value, expires_at)
        self._max_size = max_size
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._cache:
                return None
            
            value, expires_at = self._cache[key]
            if expires_at and time.time() > expires_at:
                del self._cache[key]
                return None
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        with self._lock:
            # Simple LRU eviction if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove oldest entry
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            
            expires_at = time.time() + ttl.total_seconds() if ttl else None
            self._cache[key] = (value, expires_at)
            return True
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def exists(self, key: str) -> bool:
        return self.get(key) is not None
    
    def delete_pattern(self, pattern: str) -> int:
        """Simple glob-style pattern matching."""
        import fnmatch
        
        with self._lock:
            keys_to_delete = [
                k for k in self._cache.keys()
                if fnmatch.fnmatch(k, pattern)
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return len(keys_to_delete)


class RedisBackend(CacheBackend):
    """
    Redis cache backend for distributed caching.
    
    Supports both standalone Redis and Redis Cluster.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        cluster_mode: bool = False,
        cluster_nodes: Optional[List[dict]] = None,
        key_prefix: str = "agent:"
    ):
        self.key_prefix = key_prefix
        
        if cluster_mode:
            if cluster_nodes:
                self._client = RedisCluster(
                    startup_nodes=cluster_nodes,
                    password=password,
                    decode_responses=False
                )
            else:
                self._client = RedisCluster(
                    host=host,
                    port=port,
                    password=password,
                    decode_responses=False
                )
        else:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False
            )
    
    def _prefixed_key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"
    
    def _serialize(self, value: Any) -> bytes:
        return json.dumps(value).encode('utf-8')
    
    def _deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))
    
    def get(self, key: str) -> Optional[Any]:
        data = self._client.get(self._prefixed_key(key))
        if data is None:
            return None
        return self._deserialize(data)
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        serialized = self._serialize(value)
        prefixed = self._prefixed_key(key)
        
        if ttl:
            return bool(self._client.setex(prefixed, ttl, serialized))
        return bool(self._client.set(prefixed, serialized))
    
    def delete(self, key: str) -> bool:
        return bool(self._client.delete(self._prefixed_key(key)))
    
    def exists(self, key: str) -> bool:
        return bool(self._client.exists(self._prefixed_key(key)))
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        if not keys:
            return {}
        
        prefixed_keys = [self._prefixed_key(k) for k in keys]
        values = self._client.mget(prefixed_keys)
        
        result = {}
        for key, value in zip(keys, values):
            if value is not None:
                result[key] = self._deserialize(value)
        
        return result
    
    def set_many(self, items: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool:
        if not items:
            return True
        
        pipe = self._client.pipeline()
        
        for key, value in items.items():
            prefixed = self._prefixed_key(key)
            serialized = self._serialize(value)
            
            if ttl:
                pipe.setex(prefixed, ttl, serialized)
            else:
                pipe.set(prefixed, serialized)
        
        results = pipe.execute()
        return all(results)
    
    def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching a pattern.
        
        Warning: SCAN can be slow on large datasets.
        Consider using Redis modules or different data structures
        for better pattern-based operations.
        """
        prefixed_pattern = self._prefixed_key(pattern)
        count = 0
        
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=prefixed_pattern, count=100)
            if keys:
                count += self._client.delete(*keys)
            if cursor == 0:
                break
        
        return count


class TieredCacheBackend(CacheBackend):
    """
    Two-tier cache with fast local L1 and distributed L2.
    
    Reads check L1 first, then L2. Writes go to both.
    L1 hits avoid network round-trip to L2.
    """
    
    def __init__(
        self,
        l1: CacheBackend,
        l2: CacheBackend,
        l1_ttl: timedelta = timedelta(minutes=5)
    ):
        self.l1 = l1
        self.l2 = l2
        self.l1_ttl = l1_ttl
        
        self._l1_hits = 0
        self._l2_hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        # Check L1 first
        value = self.l1.get(key)
        if value is not None:
            self._l1_hits += 1
            return value
        
        # Check L2
        value = self.l2.get(key)
        if value is not None:
            self._l2_hits += 1
            # Populate L1 for next time
            self.l1.set(key, value, self.l1_ttl)
            return value
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        # Write to both levels
        l1_result = self.l1.set(key, value, min(ttl, self.l1_ttl) if ttl else self.l1_ttl)
        l2_result = self.l2.set(key, value, ttl)
        
        return l1_result and l2_result
    
    def delete(self, key: str) -> bool:
        # Delete from both levels
        l1_result = self.l1.delete(key)
        l2_result = self.l2.delete(key)
        
        return l1_result or l2_result
    
    def exists(self, key: str) -> bool:
        return self.l1.exists(key) or self.l2.exists(key)
    
    @property
    def stats(self) -> dict:
        total = self._l1_hits + self._l2_hits + self._misses
        return {
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "misses": self._misses,
            "l1_hit_rate": self._l1_hits / total if total > 0 else 0,
            "l2_hit_rate": self._l2_hits / total if total > 0 else 0,
            "total_hit_rate": (self._l1_hits + self._l2_hits) / total if total > 0 else 0
        }


class CacheManager:
    """
    Unified cache manager for agent systems.
    
    Provides a single interface for caching LLM responses, embeddings,
    and tool results with appropriate TTLs and key generation.
    """
    
    def __init__(
        self,
        backend: CacheBackend,
        llm_ttl: timedelta = timedelta(hours=24),
        embedding_ttl: timedelta = timedelta(days=30),
        tool_ttl: timedelta = timedelta(minutes=10)
    ):
        self.backend = backend
        self.llm_ttl = llm_ttl
        self.embedding_ttl = embedding_ttl
        self.tool_ttl = tool_ttl
        
        self._llm_key_gen = LLMKeyGenerator()
        self._embedding_key_gen = CacheKeyGenerator(namespace="emb")
        self._tool_key_gen = CacheKeyGenerator(namespace="tool")
        
        # Statistics
        self._stats = {
            "llm": {"hits": 0, "misses": 0},
            "embedding": {"hits": 0, "misses": 0},
            "tool": {"hits": 0, "misses": 0}
        }
    
    def get_llm_response(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve cached LLM response."""
        # Don't cache non-deterministic responses
        if temperature > 0:
            return None
        
        key = self._llm_key_gen.for_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        result = self.backend.get(key)
        if result is not None:
            self._stats["llm"]["hits"] += 1
            return result.get("response")
        
        self._stats["llm"]["misses"] += 1
        return None
    
    def set_llm_response(
        self,
        prompt: str,
        model: str,
        response: str,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        token_count: int = 0
    ) -> str:
        """Cache an LLM response. Returns the cache key."""
        if temperature > 0:
            return ""  # Don't cache non-deterministic responses
        
        key = self._llm_key_gen.for_completion(
            prompt=prompt,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        self.backend.set(
            key,
            {
                "response": response,
                "model": model,
                "token_count": token_count,
                "cached_at": datetime.now(timezone.utc).isoformat()
            },
            self.llm_ttl
        )
        
        return key
    
    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve cached embedding vector."""
        key = self._embedding_key_gen.generate(text=text, model=model)
        
        result = self.backend.get(key)
        if result is not None:
            self._stats["embedding"]["hits"] += 1
            return result.get("vector")
        
        self._stats["embedding"]["misses"] += 1
        return None
    
    def set_embedding(
        self,
        text: str,
        model: str,
        vector: List[float]
    ) -> str:
        """Cache an embedding vector. Returns the cache key."""
        key = self._embedding_key_gen.generate(text=text, model=model)
        
        self.backend.set(
            key,
            {"vector": vector, "model": model},
            self.embedding_ttl
        )
        
        return key
    
    def get_tool_result(
        self,
        tool_name: str,
        arguments: dict
    ) -> Optional[Any]:
        """Retrieve cached tool result."""
        key = self._tool_key_gen.generate(tool=tool_name, args=arguments)
        
        result = self.backend.get(key)
        if result is not None:
            self._stats["tool"]["hits"] += 1
            return result.get("result")
        
        self._stats["tool"]["misses"] += 1
        return None
    
    def set_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        result: Any,
        ttl: Optional[timedelta] = None
    ) -> str:
        """Cache a tool result. Returns the cache key."""
        key = self._tool_key_gen.generate(tool=tool_name, args=arguments)
        
        self.backend.set(
            key,
            {"result": result, "tool": tool_name},
            ttl or self.tool_ttl
        )
        
        return key
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {}
        for category, counts in self._stats.items():
            total = counts["hits"] + counts["misses"]
            stats[category] = {
                **counts,
                "hit_rate": counts["hits"] / total if total > 0 else 0
            }
        return stats
    
    def clear_category(self, category: str) -> int:
        """Clear all cached items in a category."""
        return self.backend.delete_pattern(f"{category}:*")

# ============================================================================
# Block 11 (chapter listing #11)
# ============================================================================

from typing import Optional, Tuple, List, NamedTuple
import numpy as np
from dataclasses import dataclass, field
import threading
import heapq


@dataclass
class SemanticCacheEntry:
    """A single entry in the semantic cache."""
    query: str
    embedding: np.ndarray
    response: str
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    access_count: int = 0
    
    def similarity(self, other_embedding: np.ndarray) -> float:
        """Compute cosine similarity with another embedding."""
        dot_product = np.dot(self.embedding, other_embedding)
        norm_a = np.linalg.norm(self.embedding)
        norm_b = np.linalg.norm(other_embedding)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)


class SemanticCacheResult(NamedTuple):
    """Result from semantic cache lookup."""
    hit: bool
    response: Optional[str]
    similarity: float
    original_query: Optional[str]


class SemanticCache:
    """
    Cache that uses embedding similarity to match queries.
    
    Unlike exact-match caching, semantic caching can return cached
    responses for queries that are semantically similar but not
    identical to previously seen queries.
    """
    
    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        similarity_threshold: float = 0.95,
        max_entries: int = 10000,
        ttl: timedelta = timedelta(hours=24)
    ):
        """
        Initialize semantic cache.
        
        Args:
            embedding_model: Model to compute query embeddings
            similarity_threshold: Minimum similarity for cache hit (0.0-1.0)
            max_entries: Maximum number of entries to store
            ttl: Time-to-live for cache entries
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.ttl = ttl
        
        self._entries: List[SemanticCacheEntry] = []
        self._lock = threading.RLock()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._near_misses = 0  # Similarity > 0.8 but < threshold
    
    def get(
        self,
        query: str,
        context: Optional[str] = None
    ) -> SemanticCacheResult:
        """
        Look up a query in the semantic cache.
        
        Args:
            query: The user's query
            context: Optional context to include in embedding
        
        Returns:
            SemanticCacheResult with hit status, response, and similarity
        """
        # Compute embedding for the query
        text = f"{query}\n{context}" if context else query
        query_embedding = self.embedding_model.embed(text)
        
        with self._lock:
            # Remove expired entries
            self._evict_expired()
            
            if not self._entries:
                self._misses += 1
                return SemanticCacheResult(
                    hit=False,
                    response=None,
                    similarity=0.0,
                    original_query=None
                )
            
            # Find most similar entry
            best_entry = None
            best_similarity = 0.0
            
            for entry in self._entries:
                similarity = entry.similarity(query_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_entry = entry
            
            # Check if similarity exceeds threshold
            if best_similarity >= self.similarity_threshold:
                self._hits += 1
                best_entry.access_count += 1
                
                return SemanticCacheResult(
                    hit=True,
                    response=best_entry.response,
                    similarity=best_similarity,
                    original_query=best_entry.query
                )
            
            # Track near misses for threshold tuning
            if best_similarity >= 0.8:
                self._near_misses += 1
            
            self._misses += 1
            return SemanticCacheResult(
                hit=False,
                response=None,
                similarity=best_similarity,
                original_query=best_entry.query if best_entry else None
            )
    
    def set(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """
        Store a query-response pair in the cache.
        
        Args:
            query: The original query
            response: The response to cache
            context: Optional context used in the query
            metadata: Optional metadata to store
        """
        text = f"{query}\n{context}" if context else query
        embedding = self.embedding_model.embed(text)
        
        entry = SemanticCacheEntry(
            query=query,
            embedding=embedding,
            response=response,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Check if very similar entry already exists
            for existing in self._entries:
                if existing.similarity(embedding) > 0.99:
                    # Update existing entry instead of adding duplicate
                    existing.response = response
                    existing.metadata.update(metadata or {})
                    return
            
            # Evict if at capacity
            if len(self._entries) >= self.max_entries:
                self._evict_lru()
            
            self._entries.append(entry)
    
    def _evict_expired(self) -> int:
        """Remove expired entries. Returns count of removed entries."""
        cutoff = time.time() - self.ttl.total_seconds()
        original_count = len(self._entries)
        
        self._entries = [
            entry for entry in self._entries
            if entry.created_at > cutoff
        ]
        
        return original_count - len(self._entries)
    
    def _evict_lru(self) -> None:
        """Remove least recently used entries to make room."""
        if not self._entries:
            return
        
        # Remove 10% of entries, preferring low access count
        remove_count = max(1, len(self._entries) // 10)
        
        # Sort by access count (ascending) and age (oldest first)
        self._entries.sort(
            key=lambda e: (e.access_count, -e.created_at)
        )
        
        self._entries = self._entries[remove_count:]
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "near_misses": self._near_misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "entry_count": len(self._entries),
            "threshold": self.similarity_threshold
        }


class ProductionSemanticCache(SemanticCache):
    """
    Production-ready semantic cache with vector index for efficient lookup.
    
    Uses approximate nearest neighbor search for O(log n) lookups
    instead of O(n) linear scan.
    """
    
    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        similarity_threshold: float = 0.95,
        max_entries: int = 100000,
        ttl: timedelta = timedelta(hours=24),
        backend: Optional[CacheBackend] = None
    ):
        super().__init__(
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
            ttl=ttl
        )
        
        self.backend = backend
        self._index = None  # Would be FAISS, Annoy, or similar in production
        self._embedding_dim = None
    
    def _init_index(self, embedding_dim: int) -> None:
        """Initialize the vector index."""
        self._embedding_dim = embedding_dim
        
        # In production, you'd use FAISS, Annoy, or ScaNN
        # This is a placeholder showing the interface
        try:
            import faiss
            self._index = faiss.IndexFlatIP(embedding_dim)  # Inner product
        except ImportError:
            # Fallback to linear scan if FAISS not available
            self._index = None
    
    def get(
        self,
        query: str,
        context: Optional[str] = None,
        k: int = 5
    ) -> SemanticCacheResult:
        """
        Look up query using efficient ANN search.
        """
        text = f"{query}\n{context}" if context else query
        query_embedding = self.embedding_model.embed(text)
        
        # Normalize for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        if self._index is None:
            # Fallback to base implementation
            return super().get(query, context)
        
        with self._lock:
            if self._index.ntotal == 0:
                self._misses += 1
                return SemanticCacheResult(
                    hit=False, response=None,
                    similarity=0.0, original_query=None
                )
            
            # Search for k nearest neighbors
            similarities, indices = self._index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                min(k, self._index.ntotal)
            )
            
            best_similarity = similarities[0][0]
            best_idx = indices[0][0]
            
            if best_similarity >= self.similarity_threshold:
                self._hits += 1
                entry = self._entries[best_idx]
                entry.access_count += 1
                
                return SemanticCacheResult(
                    hit=True,
                    response=entry.response,
                    similarity=float(best_similarity),
                    original_query=entry.query
                )
            
            if best_similarity >= 0.8:
                self._near_misses += 1
            
            self._misses += 1
            return SemanticCacheResult(
                hit=False,
                response=None,
                similarity=float(best_similarity),
                original_query=self._entries[best_idx].query if best_idx >= 0 else None
            )
    
    def set(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> None:
        """Add entry to cache and index."""
        text = f"{query}\n{context}" if context else query
        embedding = self.embedding_model.embed(text)
        
        # Initialize index on first entry
        if self._embedding_dim is None:
            self._init_index(len(embedding))
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        entry = SemanticCacheEntry(
            query=query,
            embedding=embedding,
            response=response,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Add to index
            if self._index is not None:
                self._index.add(embedding.reshape(1, -1).astype(np.float32))
            
            self._entries.append(entry)
            
            # Persist to backend if available
            if self.backend:
                key = f"semantic:{hashlib.sha256(query.encode()).hexdigest()[:16]}"
                self.backend.set(key, {
                    "query": query,
                    "embedding": embedding.tolist(),
                    "response": response,
                    "metadata": metadata
                }, self.ttl)

# ============================================================================
# Block 12 (chapter listing #12)
# ============================================================================

class ThresholdTuner:
    """
    Helps tune the semantic cache similarity threshold.
    
    Collects examples of hits and misses, then analyzes them
    to recommend an appropriate threshold for your workload.
    """
    
    def __init__(self):
        self._samples: List[dict] = []
    
    def record_sample(
        self,
        query: str,
        cached_query: str,
        similarity: float,
        was_correct: bool
    ) -> None:
        """
        Record a sample for threshold analysis.
        
        Args:
            query: The incoming query
            cached_query: The query that was matched
            similarity: The computed similarity
            was_correct: Whether returning the cached response was correct
        """
        self._samples.append({
            "query": query,
            "cached_query": cached_query,
            "similarity": similarity,
            "was_correct": was_correct
        })
    
    def analyze(self) -> dict:
        """
        Analyze samples to recommend threshold.
        
        Returns analysis including recommended threshold and
        expected hit rate at various thresholds.
        """
        if len(self._samples) < 100:
            return {"error": "Need at least 100 samples for reliable analysis"}
        
        # Group by similarity buckets
        buckets = defaultdict(lambda: {"correct": 0, "incorrect": 0})
        
        for sample in self._samples:
            bucket = round(sample["similarity"], 2)
            if sample["was_correct"]:
                buckets[bucket]["correct"] += 1
            else:
                buckets[bucket]["incorrect"] += 1
        
        # Find threshold that maximizes correct hits while minimizing incorrect
        thresholds = sorted(buckets.keys(), reverse=True)
        
        best_threshold = 0.95  # Default
        best_f1 = 0
        
        for threshold in thresholds:
            # Count hits and correctness at this threshold
            hits_above = sum(
                buckets[t]["correct"] + buckets[t]["incorrect"]
                for t in thresholds if t >= threshold
            )
            correct_above = sum(
                buckets[t]["correct"]
                for t in thresholds if t >= threshold
            )
            
            if hits_above == 0:
                continue
            
            precision = correct_above / hits_above
            recall = correct_above / sum(s["was_correct"] for s in self._samples)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        return {
            "recommended_threshold": best_threshold,
            "expected_f1": best_f1,
            "sample_count": len(self._samples),
            "buckets": dict(buckets)
        }

# ============================================================================
# Block 13 (chapter listing #13)
# ============================================================================

@dataclass
class QualityAwareSemanticCacheEntry:
    """Cache entry with quality tracking for safer semantic matching."""
    query: str
    embedding: np.ndarray
    response: str
    quality_score: float  # 0.0-1.0, from evaluation or user feedback
    feedback_count: int = 0
    created_at: float = field(default_factory=time.time)


async def semantic_lookup_with_quality(
    query: str,
    cache: SemanticCache,
    min_quality: float = 0.8
) -> Optional[str]:
    """Only return cached responses that meet quality threshold."""
    result = await cache.get(query)

    if result.hit and result.entry.quality_score >= min_quality:
        return result.response
    elif result.hit and result.entry.quality_score < min_quality:
        # Log for analysis: we had a hit but quality was too low
        logger.info(
            f"Semantic cache hit rejected: quality={result.entry.quality_score}"
        )
    return None

# ============================================================================
# Block 14 (chapter listing #14)
# ============================================================================

from typing import List, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging


logger = logging.getLogger(__name__)


class CacheWarmer:
    """
    Pre-populates caches with expected queries.
    
    Supports multiple warming strategies:
    - Historical: Replay past queries
    - Synthetic: Generate expected queries
    - Scheduled: Warm during low-traffic periods
    """
    
    def __init__(
        self,
        cache_manager: CacheManager,
        llm_client: "LLMClient",
        embedding_model: "EmbeddingModel",
        max_workers: int = 4
    ):
        self.cache_manager = cache_manager
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        self.max_workers = max_workers
    
    def warm_from_history(
        self,
        query_log: Iterator[dict],
        limit: int = 1000
    ) -> dict:
        """
        Warm cache by replaying historical queries.
        
        Args:
            query_log: Iterator of past query records
            limit: Maximum number of queries to warm
        
        Returns:
            Statistics about the warming process
        """
        stats = {"processed": 0, "cached": 0, "errors": 0}
        
        queries = []
        for record in query_log:
            queries.append(record)
            if len(queries) >= limit:
                break
        
        logger.info(f"Warming cache with {len(queries)} historical queries")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for record in queries:
                future = executor.submit(
                    self._warm_single_query,
                    record.get("prompt", ""),
                    record.get("model", "gpt-4"),
                    record.get("system_prompt")
                )
                futures.append(future)
            
            for future in as_completed(futures):
                stats["processed"] += 1
                try:
                    if future.result():
                        stats["cached"] += 1
                except Exception as e:
                    logger.warning(f"Error warming query: {e}")
                    stats["errors"] += 1
        
        return stats
    
    def _warm_single_query(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str]
    ) -> bool:
        """Warm cache for a single query."""
        # Check if already cached
        cached = self.cache_manager.get_llm_response(
            prompt=prompt,
            model=model,
            temperature=0.0,
            system_prompt=system_prompt
        )
        
        if cached is not None:
            return False  # Already cached
        
        # Generate and cache response
        response = self.llm_client.complete(
            prompt=prompt,
            model=model,
            temperature=0.0,
            system_prompt=system_prompt
        )
        
        self.cache_manager.set_llm_response(
            prompt=prompt,
            model=model,
            response=response.text,
            temperature=0.0,
            system_prompt=system_prompt,
            token_count=response.token_count
        )
        
        return True
    
    def warm_embeddings(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002"
    ) -> dict:
        """
        Pre-compute and cache embeddings for a list of texts.
        """
        stats = {"processed": 0, "cached": 0, "skipped": 0}
        
        # Check which texts need embedding
        texts_to_embed = []
        for text in texts:
            cached = self.cache_manager.get_embedding(text, model)
            if cached is None:
                texts_to_embed.append(text)
            else:
                stats["skipped"] += 1
        
        logger.info(f"Computing {len(texts_to_embed)} embeddings ({stats['skipped']} already cached)")
        
        # Batch compute embeddings
        batch_size = 100
        for i in range(0, len(texts_to_embed), batch_size):
            batch = texts_to_embed[i:i + batch_size]
            
            try:
                embeddings = self.embedding_model.embed_batch(batch)
                
                for text, embedding in zip(batch, embeddings):
                    self.cache_manager.set_embedding(text, model, embedding.tolist())
                    stats["cached"] += 1
                
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}")
            
            stats["processed"] += len(batch)
        
        return stats
    
    def warm_semantic_cache(
        self,
        semantic_cache: SemanticCache,
        query_response_pairs: List[Tuple[str, str]],
        context: Optional[str] = None
    ) -> dict:
        """
        Pre-populate semantic cache with known query-response pairs.
        
        Useful for FAQ-style queries where you know the answers.
        """
        stats = {"added": 0, "errors": 0}
        
        for query, response in query_response_pairs:
            try:
                semantic_cache.set(
                    query=query,
                    response=response,
                    context=context
                )
                stats["added"] += 1
            except Exception as e:
                logger.warning(f"Error adding to semantic cache: {e}")
                stats["errors"] += 1
        
        return stats


class ScheduledCacheWarmer:
    """
    Runs cache warming on a schedule during low-traffic periods.
    """
    
    def __init__(
        self,
        warmer: CacheWarmer,
        schedule: str = "0 3 * * *"  # 3 AM daily
    ):
        self.warmer = warmer
        self.schedule = schedule
        self._running = False
    
    def start(self) -> None:
        """Start the scheduled warmer."""
        # In production, use a proper scheduler like APScheduler
        # This is a simplified example
        import schedule as sched
        
        sched.every().day.at("03:00").do(self._run_warming)
        self._running = True
        
        while self._running:
            sched.run_pending()
            time.sleep(60)
    
    def stop(self) -> None:
        """Stop the scheduled warmer."""
        self._running = False
    
    def _run_warming(self) -> None:
        """Execute the warming routine."""
        logger.info("Starting scheduled cache warming")
        
        # Get historical queries from your logging system
        # This is application-specific
        query_log = self._get_recent_queries()
        
        stats = self.warmer.warm_from_history(query_log, limit=5000)
        logger.info(f"Cache warming complete: {stats}")
    
    def _get_recent_queries(self) -> Iterator[dict]:
        """Retrieve recent queries from your logging system."""
        # Implement based on your logging infrastructure
        # Could be from Elasticsearch, CloudWatch, etc.
        raise NotImplementedError("Implement based on your logging system")

# ============================================================================
# Block 15 (chapter block #15) — Python fragment (incomplete, depends on surrounding context)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_15_listing = r"""
Daily Operations:
- Embedding computations: 50,000 x $0.0001 = $5
- LLM calls: 100,000 x $0.03 = $3,000
- Tool calls: 75,000 x $0.001 = $75

Daily total: $3,080
Monthly total: $92,400
"""

# ============================================================================
# Block 16 (chapter block #16) — Python fragment (incomplete, depends on surrounding context)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_16_listing = r"""
Cache Performance:
- Embedding cache hit rate: 85%
- LLM response cache hit rate: 45%
- Semantic cache hit rate: 20%
- Tool result cache hit rate: 70%

Effective LLM cache hit rate: 45% + (55% x 20%) = 56%

Daily Operations:
- Embedding computations: 50,000 x 0.15 x $0.0001 = $0.75
- LLM calls: 100,000 x 0.44 x $0.03 = $1,320
- Tool calls: 75,000 x 0.30 x $0.001 = $22.50

Daily total: $1,343.25
Monthly total: $40,298

Infrastructure costs:
- Redis cluster: $500/month
- Vector DB for semantic cache: $200/month

Net monthly total: $40,998
Monthly savings: $51,402 (56% reduction)
"""

# ============================================================================
# Block 17 (chapter block #17) — Python fragment (incomplete, depends on surrounding context)
# Preserved verbatim from the book. Not standalone-runnable.
# ============================================================================

_block_17_listing = r"""
Development time: 80 hours @ $150/hr = $12,000
Infrastructure setup: $2,000
Testing and tuning: $3,000

Total implementation: $17,000
Payback period: < 1 month
"""

# ============================================================================
# Block 18 (chapter listing #18)
# ============================================================================

class CacheMetrics:
    """
    Production metrics for cache monitoring.
    
    Exposes metrics in Prometheus format for dashboarding and alerting.
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        
        # Track costs
        self._cost_saved = 0.0
        self._cost_incurred = 0.0
        
        # Track latencies
        self._cache_latencies: List[float] = []
        self._llm_latencies: List[float] = []
    
    def record_cache_hit(
        self,
        category: str,
        latency_ms: float,
        estimated_cost_saved: float
    ) -> None:
        """Record a cache hit."""
        self._cost_saved += estimated_cost_saved
        self._cache_latencies.append(latency_ms)
    
    def record_cache_miss(
        self,
        category: str,
        latency_ms: float,
        cost_incurred: float
    ) -> None:
        """Record a cache miss."""
        self._cost_incurred += cost_incurred
        self._llm_latencies.append(latency_ms)
    
    def get_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        stats = self.cache_manager.get_stats()
        
        lines = []
        
        # Hit rates by category
        for category, data in stats.items():
            lines.append(
                f'cache_hit_rate{{category="{category}"}} {data["hit_rate"]:.4f}'
            )
            lines.append(
                f'cache_hits_total{{category="{category}"}} {data["hits"]}'
            )
            lines.append(
                f'cache_misses_total{{category="{category}"}} {data["misses"]}'
            )
        
        # Cost metrics
        lines.append(f'cache_cost_saved_dollars {self._cost_saved:.2f}')
        lines.append(f'cache_cost_incurred_dollars {self._cost_incurred:.2f}')
        
        # Latency metrics
        if self._cache_latencies:
            lines.append(
                f'cache_latency_p50_ms {np.percentile(self._cache_latencies, 50):.2f}'
            )
            lines.append(
                f'cache_latency_p99_ms {np.percentile(self._cache_latencies, 99):.2f}'
            )
        
        if self._llm_latencies:
            lines.append(
                f'llm_latency_p50_ms {np.percentile(self._llm_latencies, 50):.2f}'
            )
            lines.append(
                f'llm_latency_p99_ms {np.percentile(self._llm_latencies, 99):.2f}'
            )
        
        return '\n'.join(lines)
    
    def get_savings_report(self) -> dict:
        """Generate a cost savings report."""
        total_requests = sum(
            data["hits"] + data["misses"]
            for data in self.cache_manager.get_stats().values()
        )
        
        total_hits = sum(
            data["hits"]
            for data in self.cache_manager.get_stats().values()
        )
        
        return {
            "total_requests": total_requests,
            "total_hits": total_hits,
            "overall_hit_rate": total_hits / total_requests if total_requests > 0 else 0,
            "cost_saved": self._cost_saved,
            "cost_incurred": self._cost_incurred,
            "savings_percentage": (
                self._cost_saved / (self._cost_saved + self._cost_incurred)
                if (self._cost_saved + self._cost_incurred) > 0 else 0
            ),
            "avg_cache_latency_ms": np.mean(self._cache_latencies) if self._cache_latencies else 0,
            "avg_llm_latency_ms": np.mean(self._llm_latencies) if self._llm_latencies else 0,
            "latency_improvement_factor": (
                np.mean(self._llm_latencies) / np.mean(self._cache_latencies)
                if self._cache_latencies and self._llm_latencies else 0
            )
        }

# ============================================================================
# Block 19 (chapter listing #19)
# ============================================================================

class CachedAgentRuntime:
    """
    Agent runtime with integrated caching layer.
    
    Demonstrates how caching integrates with the agent execution loop.
    """
    
    def __init__(
        self,
        agent: "Agent",
        llm_client: "LLMClient",
        embedding_model: "EmbeddingModel",
        redis_url: str = "redis://localhost:6379"
    ):
        self.agent = agent
        self.llm_client = llm_client
        self.embedding_model = embedding_model
        
        # Initialize cache backends
        l1_cache = InMemoryBackend(max_size=1000)
        l2_cache = RedisBackend(host="localhost", port=6379)
        tiered_backend = TieredCacheBackend(l1_cache, l2_cache)
        
        # Initialize cache manager
        self.cache_manager = CacheManager(tiered_backend)
        
        # Initialize semantic cache for query matching
        self.semantic_cache = ProductionSemanticCache(
            embedding_model=embedding_model,
            similarity_threshold=0.95,
            backend=l2_cache
        )
        
        # Initialize tool result cache with policies
        self.tool_cache = ToolResultCache(
            backend=tiered_backend,
            policies=TOOL_CACHE_POLICIES
        )
        
        # Metrics
        self.metrics = CacheMetrics(self.cache_manager)
    
    async def process_query(
        self,
        query: str,
        context: Optional[dict] = None
    ) -> str:
        """
        Process a user query with caching at every layer.
        """
        start_time = time.time()
        
        # 1. Check semantic cache first
        semantic_result = self.semantic_cache.get(query)
        if semantic_result.hit:
            self.metrics.record_cache_hit(
                "semantic",
                latency_ms=(time.time() - start_time) * 1000,
                estimated_cost_saved=0.03
            )
            return semantic_result.response
        
        # 2. Run agent with caching hooks
        response = await self._run_agent_with_cache(query, context)
        
        # 3. Store in semantic cache for future similar queries
        self.semantic_cache.set(query, response)
        
        return response
    
    async def _run_agent_with_cache(
        self,
        query: str,
        context: Optional[dict]
    ) -> str:
        """Run agent loop with caching at each step."""
        
        # Create cached LLM wrapper
        async def cached_llm_call(
            prompt: str,
            system_prompt: Optional[str] = None
        ) -> str:
            # Check cache
            cached = self.cache_manager.get_llm_response(
                prompt=prompt,
                model=self.agent.model,
                temperature=0.0,
                system_prompt=system_prompt
            )
            
            if cached is not None:
                return cached
            
            # Call LLM
            response = await self.llm_client.complete_async(
                prompt=prompt,
                model=self.agent.model,
                temperature=0.0,
                system_prompt=system_prompt
            )
            
            # Cache response
            self.cache_manager.set_llm_response(
                prompt=prompt,
                model=self.agent.model,
                response=response.text,
                temperature=0.0,
                system_prompt=system_prompt,
                token_count=response.token_count
            )
            
            return response.text
        
        # Create cached tool wrapper
        async def cached_tool_call(
            tool_name: str,
            arguments: dict
        ) -> Any:
            # Check cache
            cached = self.tool_cache.get(tool_name, arguments)
            if cached is not None:
                return cached
            
            # Execute tool
            result = await self.agent.execute_tool(tool_name, arguments)
            
            # Cache result
            self.tool_cache.set(tool_name, arguments, result)
            
            return result
        
        # Run agent with cached wrappers
        return await self.agent.run(
            query=query,
            context=context,
            llm_call=cached_llm_call,
            tool_call=cached_tool_call
        )
