# Ragify

A production-ready Retrieval-Augmented Generation (RAG) system with hybrid search, intelligent chunking, and reranking capabilities.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [System Components](#system-components)
- [Tech Stack](#tech-stack)
- [Features](#features)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Best Practices](#best-practices)

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Web UI    │  │  REST API  │  │  GraphQL   │  │   CLI Tool   │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘  │
└────────┼───────────────┼───────────────┼────────────────┼───────────┘
         │               │               │                │
         └───────────────┴───────────────┴────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                             │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    RAG Orchestrator                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │  │Query Analysis│  │  Retrieval   │  │    Generation    │   │  │
│  │  │  & Routing   │→ │  Pipeline    │→ │    Pipeline      │   │  │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────────┐
│  INGESTION PIPELINE  │  │  RETRIEVAL PIPELINE  │  │ GENERATION API  │
├──────────────────────┤  ├──────────────────────┤  ├─────────────────┤
│ 1. Document Loader   │  │ 1. Query Embedding   │  │ 1. Context Prep │
│ 2. Text Extraction   │  │ 2. Hybrid Search:    │  │ 2. Prompt Build │
│ 3. Smart Chunking    │  │    • Vector Search   │  │ 3. LLM Call     │
│ 4. Metadata Extract  │  │    • Keyword Search  │  │ 4. Response     │
│ 5. Embedding Gen     │  │    • Graph Search    │  │    Validation   │
│ 6. Store in DB       │  │ 3. Result Fusion     │  │ 5. Citation     │
│                      │  │ 4. Reranking         │  │    Injection    │
│                      │  │ 5. Context Selection │  │                 │
└──────────────────────┘  └──────────────────────┘  └─────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          STORAGE LAYER                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │   Vector   │  │  Document  │  │   Graph    │  │    Cache     │  │
│  │   Store    │  │  Metadata  │  │   Store    │  │   (Redis)    │  │
│  │ (PGVector/ │  │ (Postgres) │  │  (Neo4j)   │  │              │  │
│  │  Qdrant)   │  │            │  │  optional  │  │              │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
         │                          │                          │
         └──────────────────────────┴──────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     MONITORING & OBSERVABILITY                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Metrics   │  │   Traces   │  │    Logs    │  │   Feedback   │  │
│  │(Prometheus)│  │  (Jaeger)  │  │    (ELK)   │  │    Loop      │  │
│  └────────────┘  └────────────┘  └────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

## System Components

### 1. Ingestion Pipeline

The ingestion pipeline processes and indexes documents for retrieval:

**Document Loaders**
- PDF, DOCX, TXT, Markdown, HTML parsers
- Support for structured data (JSON, CSV, XML)
- Image and diagram extraction (OCR support)
- Code repository ingestion

**Text Processing**
- Smart chunking strategies:
  - Semantic chunking (sentence-transformers)
  - Recursive chunking with overlap
  - Document structure-aware chunking (respects headers, paragraphs)
  - Code-aware chunking (function/class boundaries)
- Metadata extraction:
  - Document source, timestamps, authors
  - Section headers, hierarchy
  - Entity extraction (NER)
  - Custom metadata tags

**Embedding Generation**
- Support for multiple embedding models:
  - OpenAI `text-embedding-3-small/large`
  - Sentence-Transformers (`all-MiniLM-L6-v2`, `e5-large`)
  - Cohere Embed v3
  - Custom fine-tuned models
- Batch processing for efficiency
- Embedding dimension: 384-1536 (configurable)

### 2. Storage Layer

**Vector Store**
- **Primary Option**: PostgreSQL with pgvector
  - ACID compliance
  - Mature ecosystem
  - HNSW indexing for fast ANN search
- **Alternative**: Qdrant
  - Purpose-built for vectors
  - Advanced filtering capabilities
  - Excellent performance at scale

**Document Metadata Store**
- PostgreSQL with JSONB columns
- Full-text search capabilities (tsvector)
- Relational structure for complex queries

**Graph Store (Optional)**
- Neo4j for knowledge graph relationships
- Entity linking and graph-based retrieval
- Enhanced context through relationship traversal

**Caching Layer**
- Redis for query result caching
- Embedding cache to avoid recomputation
- Session management

### 3. Retrieval Pipeline

**Query Processing**
- Query understanding and expansion
- Intent classification
- Multi-query generation for comprehensive retrieval

**Hybrid Search**
1. **Vector Search** (Semantic)
   - Cosine similarity on embeddings
   - Top-k retrieval (k=20-100)
   - Filters: metadata, date ranges, source types

2. **Keyword Search** (Lexical)
   - BM25 algorithm
   - Fuzzy matching
   - Boolean operators support

3. **Graph Search** (Optional)
   - Entity-based retrieval
   - Relationship traversal
   - Community detection

**Result Fusion**
- Reciprocal Rank Fusion (RRF)
- Weighted combination strategies
- Diversity-aware selection

**Reranking**
- Cross-encoder models (e.g., `ms-marco-MiniLM`)
- Relevance scoring
- Context window optimization
- Final top-k selection (k=3-10)

### 4. Generation Pipeline

**Context Preparation**
- Retrieved chunks formatting
- Context compression (if needed)
- Citation tracking

**Prompt Engineering**
- System prompts with instructions
- Context injection strategies
- Few-shot examples (optional)
- Chain-of-thought prompting

**LLM Integration**
- OpenAI GPT-4, GPT-4-turbo
- Anthropic Claude 3 (Opus, Sonnet, Haiku)
- Open-source: Llama 3, Mixtral
- Streaming responses
- Temperature and parameter tuning

**Response Processing**
- Citation injection
- Fact verification
- Answer validation
- Confidence scoring

### 5. Orchestration Layer

**Request Handling**
- Rate limiting and throttling
- Request validation
- Authentication & authorization
- Query routing based on complexity

**Pipeline Management**
- Async task processing (Celery/RQ)
- Retry logic with exponential backoff
- Error handling and fallbacks
- Pipeline composition

### 6. Monitoring & Observability

**Metrics**
- Query latency (p50, p95, p99)
- Retrieval accuracy
- LLM token usage
- Cache hit rates
- System resource utilization

**Tracing**
- Distributed tracing across pipeline stages
- Query journey visualization
- Performance bottleneck identification

**Feedback Loop**
- User feedback collection (thumbs up/down)
- Relevance judgments
- A/B testing framework
- Continuous evaluation

## Tech Stack

### Backend (Recommended: Python)

```python
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0

# RAG & ML
langchain==0.1.0
llama-index==0.9.0
sentence-transformers==2.3.0
openai==1.10.0
anthropic==0.8.0

# Vector & Storage
pgvector==0.2.4
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
redis==5.0.1

# Search & Retrieval
elasticsearch==8.11.0  # optional
rank-bm25==0.2.2

# Document Processing
pypdf==3.17.0
python-docx==1.1.0
beautifulsoup4==4.12.3
unstructured==0.11.0

# Monitoring
prometheus-client==0.19.0
opentelemetry-api==1.22.0

# Utils
pydantic==2.5.0
pydantic-settings==2.1.0
python-dotenv==1.0.0
```

### Frontend (Optional)

```typescript
// Framework
next.js 14+
react 18+
typescript

// UI Components
shadcn/ui
tailwindcss
radix-ui

// State Management
zustand / tanstack-query

// API Client
axios / fetch
```

### Infrastructure

- **Database**: PostgreSQL 15+ with pgvector extension
- **Cache**: Redis 7+
- **Message Queue**: RabbitMQ or Redis
- **Container**: Docker & Docker Compose
- **Orchestration**: Kubernetes (production)
- **CI/CD**: GitHub Actions / GitLab CI

## Features

- ✅ Hybrid search combining semantic and keyword retrieval
- ✅ Advanced document chunking with overlap and metadata
- ✅ Multi-stage retrieval with reranking
- ✅ Support for multiple LLM providers
- ✅ Citation tracking and source attribution
- ✅ Query result caching for performance
- ✅ Real-time streaming responses
- ✅ Multi-document question answering
- ✅ Conversation memory and context
- ✅ Document versioning and updates
- ✅ Fine-grained access control
- ✅ Comprehensive metrics and monitoring
- ✅ Extensible plugin architecture

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15+ with pgvector
- Redis 7+
- OpenAI API key (or other LLM provider)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragify.git
cd ragify

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python scripts/init_db.py

# Run migrations
alembic upgrade head
```

### Quick Start

```bash
# Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start the worker (for async tasks)
celery -A app.worker worker --loglevel=info
```

### Docker Setup

```bash
# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## API Reference

### Ingestion

**Upload Documents**
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

{
  "file": <binary>,
  "metadata": {
    "source": "internal_docs",
    "category": "engineering",
    "tags": ["api", "backend"]
  }
}
```

**Bulk Ingestion**
```bash
POST /api/v1/documents/ingest
Content-Type: application/json

{
  "documents": [
    {
      "content": "Document text...",
      "metadata": {...}
    }
  ],
  "chunking_strategy": "semantic",
  "chunk_size": 512,
  "chunk_overlap": 50
}
```

### Query

**Basic Query**
```bash
POST /api/v1/query
Content-Type: application/json

{
  "query": "What is the return policy?",
  "top_k": 5,
  "include_sources": true,
  "stream": false
}
```

**Advanced Query with Filters**
```bash
POST /api/v1/query
Content-Type: application/json

{
  "query": "Explain the authentication flow",
  "filters": {
    "category": "engineering",
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-12-31"
    }
  },
  "retrieval_config": {
    "hybrid_search": true,
    "rerank": true,
    "top_k": 10,
    "rerank_top_k": 3
  },
  "generation_config": {
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Streaming Response**
```bash
POST /api/v1/query/stream
Content-Type: application/json

{
  "query": "Summarize the Q4 report",
  "stream": true
}

# Server-Sent Events response
```

### Conversation

**Create Conversation**
```bash
POST /api/v1/conversations
Content-Type: application/json

{
  "title": "Product Questions",
  "metadata": {}
}
```

**Send Message**
```bash
POST /api/v1/conversations/{conversation_id}/messages
Content-Type: application/json

{
  "content": "What are the key features?"
}
```

## Configuration

### Environment Variables

```bash
# Application
APP_NAME=Ragify
APP_VERSION=1.0.0
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/ragify
REDIS_URL=redis://localhost:6379/0

# LLM Providers
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Embedding Model
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Retrieval Configuration
VECTOR_SEARCH_TOP_K=20
RERANK_TOP_K=5
ENABLE_RERANKING=true
ENABLE_HYBRID_SEARCH=true

# Generation
DEFAULT_LLM_MODEL=gpt-4-turbo
DEFAULT_TEMPERATURE=0.7
MAX_TOKENS=2000

# Caching
ENABLE_QUERY_CACHE=true
CACHE_TTL_SECONDS=3600

# Monitoring
ENABLE_METRICS=true
ENABLE_TRACING=true
```

### Chunking Strategies

```python
# config/chunking.py
CHUNKING_STRATEGIES = {
    "fixed": {
        "chunk_size": 512,
        "chunk_overlap": 50
    },
    "semantic": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "breakpoint_threshold": 0.5
    },
    "recursive": {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "separators": ["\n\n", "\n", " ", ""]
    }
}
```

## Best Practices

### 1. Document Preparation

- Clean and normalize text before ingestion
- Preserve document structure and metadata
- Use appropriate chunking strategy for content type
- Include relevant metadata for filtering

### 2. Embedding Strategy

- Use domain-specific embeddings when available
- Consider fine-tuning embeddings for specialized domains
- Maintain consistent embedding models across system
- Cache embeddings to reduce costs

### 3. Retrieval Optimization

- Start with hybrid search (vector + keyword)
- Always use reranking for better precision
- Tune top_k based on your use case (typically 20-50 for retrieval, 3-5 after reranking)
- Implement query expansion for short queries
- Use metadata filters to narrow search space

### 4. Generation Quality

- Craft clear, specific system prompts
- Include explicit instructions for citation
- Use appropriate temperature (0.0-0.3 for factual, 0.7-1.0 for creative)
- Implement streaming for better UX
- Validate generated responses

### 5. Performance

- Implement aggressive caching at multiple levels
- Use async processing for ingestion
- Batch embed documents
- Monitor and optimize slow queries
- Consider read replicas for high-traffic scenarios

### 6. Evaluation

- Track retrieval metrics (MRR, NDCG, precision@k)
- Monitor generation quality (BLEU, ROUGE, BERTScore)
- Collect user feedback systematically
- Run A/B tests for major changes
- Maintain evaluation datasets

### 7. Security

- Implement proper authentication and authorization
- Validate and sanitize all inputs
- Use rate limiting to prevent abuse
- Encrypt sensitive data at rest and in transit
- Audit document access patterns

### 8. Scalability

- Design for horizontal scalability
- Use connection pooling
- Implement circuit breakers
- Monitor resource utilization
- Plan for vector store growth

## Project Structure

```
ragify/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration management
│   ├── dependencies.py         # Dependency injection
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py    # Document endpoints
│   │   │   ├── query.py        # Query endpoints
│   │   │   └── conversations.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── ingestion.py        # Document processing
│   │   ├── retrieval.py        # Retrieval pipeline
│   │   ├── generation.py       # LLM integration
│   │   └── reranker.py         # Reranking logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy models
│   │   └── schemas.py          # Pydantic schemas
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── embedding.py        # Embedding service
│   │   ├── vector_store.py     # Vector DB operations
│   │   ├── cache.py            # Caching layer
│   │   └── llm.py              # LLM clients
│   │
│   └── utils/
│       ├── __init__.py
│       ├── chunking.py         # Chunking utilities
│       ├── logging.py          # Logging setup
│       └── metrics.py          # Metrics collection
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── scripts/
│   ├── init_db.py
│   ├── seed_data.py
│   └── benchmarks.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   └── evaluation.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── alembic.ini
└── README.md
```

## Roadmap

- [ ] Multi-modal RAG (images, tables, charts)
- [ ] Advanced query understanding with intent classification
- [ ] Graph-based retrieval integration
- [ ] Automated evaluation pipeline
- [ ] Fine-tuning pipeline for embeddings
- [ ] Support for additional vector stores (Milvus, Weaviate)
- [ ] Real-time document updates
- [ ] Multi-lingual support
- [ ] Federated search across multiple RAG systems
- [ ] RAG-as-a-Service deployment templates

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LlamaIndex](https://llamaindex.ai/)
- Inspired by best practices from industry leaders
- Community feedback and contributions

---

**Built with ❤️ for the RAG community**