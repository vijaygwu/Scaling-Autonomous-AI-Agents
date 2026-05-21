# Scaling Autonomous AI Agents: Engineering for Production at Real-World Scale

**Companion Code Repository**

*Book 3 of the Agentic AI Series*

by Dr. Vijay Raghavan

---

## About This Repository

This repository contains the code listings from *Scaling Autonomous AI Agents: Engineering for Production at Real-World Scale* (Book 3 of the Agentic AI Series). Each chapter has its own directory; inside is a single Python module that reproduces every code listing from that chapter in book order, with section banners showing the block number.

The code is a **faithful extraction** of the in-book listings. Some listings define classes and functions that build incrementally across the chapter; others are illustrative fragments (log output, file trees, Dockerfile snippets, JSON examples) that have been preserved as docstrings so the chapter file always remains valid Python. To run a particular component, copy the relevant class or function into your own project and provide surrounding context (imports, dependencies, configuration) as needed.

If you are new to the series, see also:
- **Book 1: Agent Architectures** — design patterns (Orchestrator, Council, Swarm, Guardian, Hybrid). [github.com/vijaygwu/Agent-Architectures](https://github.com/vijaygwu/Agent-Architectures)
- **Book 2: Ship, Scale, and Govern Autonomous Systems** — governance and operations. [github.com/vijaygwu/Ship-Scale-and-Govern-Autonomous-Systems](https://github.com/vijaygwu/Ship-Scale-and-Govern-Autonomous-Systems)

## Book Overview

Book 3 takes the production foundation from Book 2 and pushes into scale, state, retrieval, and end-to-end systems. Part I (Infrastructure) covers horizontal/vertical scaling, durable state, vector databases for agent memory, and caching strategies. Part II (Complete Examples) presents two end-to-end production systems: a customer-service platform and a procurement automation system. Part III (Looking Ahead) closes with a survey of where agentic AI is headed.

## Repository Layout

```
book-3/code/
├── ch01-scaling/                # Horizontal/vertical scaling, queues, autoscaling
│   └── scaling.py
├── ch02-state/                  # Durable state, sessions, persistence patterns
│   └── state.py
├── ch03-vector-databases/       # Pinecone, Qdrant, Weaviate, pgvector, FAISS, RAG
│   └── vector_databases.py
├── ch04-caching/                # Semantic caching, prompt cache, eviction policies
│   └── caching.py
├── ch05-customer-service/       # End-to-end customer service platform
│   └── customer_service.py
├── ch06-procurement/            # End-to-end procurement automation
│   └── procurement.py
├── ch07-future/                 # Looking Ahead (prose-only chapter; no code)
│   └── future.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Chapter Code

| Chapter | Topic | Module |
|---|---|---|
| 1 | Scaling Agent Systems | [`ch01-scaling/scaling.py`](ch01-scaling/scaling.py) |
| 2 | State Management | [`ch02-state/state.py`](ch02-state/state.py) |
| 3 | Vector Databases and Agent Memory | [`ch03-vector-databases/vector_databases.py`](ch03-vector-databases/vector_databases.py) |
| 4 | Caching Strategies for Agents | [`ch04-caching/caching.py`](ch04-caching/caching.py) |
| 5 | Complete Example: Customer Service Platform | [`ch05-customer-service/customer_service.py`](ch05-customer-service/customer_service.py) |
| 6 | Complete Example: Procurement Automation | [`ch06-procurement/procurement.py`](ch06-procurement/procurement.py) |
| 7 | Looking Ahead: The Future of Agentic AI | [`ch07-future/future.py`](ch07-future/future.py) *(prose only — no code listings)* |

## Getting Started

```bash
# Create a virtual environment (Python 3.11+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Open any chapter module and read top-to-bottom alongside the book
$EDITOR ch01-scaling/scaling.py
```

## Conventions

- **Block banners**: Each listing is preceded by a banner showing its sequential position in the file (`Block N`) and the corresponding listing number from the chapter.
- **Wrapped listings**: Blocks that are not standalone Python (log samples, Dockerfile snippets, JSON examples, etc.) are wrapped in raw docstrings and labelled with a reason — they preserve the book content verbatim but are not meant to execute.
- **Future imports**: `from __future__ import annotations` is hoisted to the top of each file when used anywhere in the chapter.

## Production Safety Patterns

The chapter modules carry the production-readiness patterns advocated in the book. When you copy code into your own project, these patterns travel with it:

- **Bounded collections**: `deque(maxlen=...)`, OrderedDict-backed LRU caches, ring-buffer metrics. The `AdaptiveTTLManager` and `ThresholdTuner` in `ch04-caching` are bounded so high-cardinality keys do not leak memory. The `BackpressureConsumer` in `ch01-scaling` rolls back the dequeue's attempt-counter increment so rate limiting does not silently burn retry budget.
- **Explicit timeouts and retries**: Redis (`socket_timeout`, `socket_connect_timeout`, `health_check_interval`), DynamoDB (`BotoConfig` with `connect_timeout=2`, `read_timeout=5`, adaptive retries), LLM providers (per-call `timeout=30`), asyncpg connection pools with `command_timeout`. No caller hangs indefinitely on a flaky dependency.
- **Specific exception handling**: typed ladders for the Anthropic SDK (`APITimeoutError` / `RateLimitError` / `APIConnectionError` retry once; `AuthenticationError` / `BadRequestError` escalate without retry); `redis.RedisError` handlers that bump distinct error metrics so a Redis backend failure is observable separately from a normal cache miss.
- **Scale rails with loud warnings**: `BM25Retriever` and `KnowledgeBase` in `ch03-vector-databases` emit `warnings.warn(...)` past their in-memory operating envelope; `SemanticCache` in `ch04-caching` warns at the linear-scan cutover threshold and points callers at `ProductionSemanticCache` (FAISS-backed).
- **Idempotency**: `ProcurementOrchestrator.submit_request` short-circuits on duplicate request IDs and returns `SubmissionResult(status="duplicate")`. Customer-service refund tools fall back to `uuid4` idempotency keys when callers do not supply one, so two legitimate partial refunds of the same order do not collide.
- **PAN detection that does not false-positive on order IDs**: the compliance regex in `ch05-customer-service` is gated on a Luhn mod-10 check before docking the conversation's compliance score, so 13-16-digit tracking numbers no longer count as card-number exposure.
- **Cancellable async loops**: Redis pubsub watchers in `ch02-state` use timed `pubsub.get_message(timeout=1.0)` rather than open-ended `pubsub.listen()`, so the surrounding asyncio task can be cancelled cleanly when Redis drops the connection.
- **Vector-clock semantics**: the conflict-resolution path in `ch02-state` short-circuits identical clocks before invoking the additive merge, so concurrent updates that happen to land at the same causal version do not double-count fields like loyalty-point totals.
- **Per-conversation state on shared agents**: `BaseAgent` in `ch05-customer-service` keeps the turn counter on the `Conversation` object rather than on the agent instance, so a single shared agent serving multiple concurrent conversations does not leak its turn count across them.

These are not just illustrative; they are the patterns the surrounding prose advocates. If you copy a class out of one of these modules, you take the safety scaffolding with it.

## External Services Referenced

Book 3 reaches further across the infrastructure stack than Books 1 or 2. Code in this repository references the following services and libraries — provide your own deployments or mocks when running examples:

- **LLM providers**: Anthropic, OpenAI
- **Vector databases**: Pinecone, Qdrant, Weaviate, FAISS, pgvector
- **Embeddings**: sentence-transformers, InstructorEmbedding
- **Relational + cache**: PostgreSQL (with pgvector), Redis
- **Messaging**: RabbitMQ (via pika)
- **Cloud**: AWS (via boto3)

## Compatibility

- **Python**: 3.11 or higher (modern type hints, `match` statements, async/await)
- Some chapters use heavyweight ML libraries (FAISS, sentence-transformers) that are easiest to install on Linux or macOS

## Errata

Code in this repository corresponds to the manuscript as of the date the repository was last refreshed. If you find a discrepancy with the published book, open an issue.

## License

MIT — see `LICENSE` (to be added).

---

*Part of the Agentic AI Series.*
