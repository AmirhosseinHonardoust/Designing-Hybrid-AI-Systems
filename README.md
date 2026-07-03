<div align="center">
         
# Designing Hybrid AI Systems
### Blending Graphs, Vectors, and Language Models for Explainable Retrieval
  
*A design blueprint and set of engineering lessons for building a Graph-Powered RAG engine — systems that don't just answer, but **cite, reason, and explain**.*
  
![Status](https://img.shields.io/badge/status-conceptual%20blueprint-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Topic](https://img.shields.io/badge/topic-Graph%20RAG%20%7C%20Explainable%20AI-blue)
  
</div>

> [!NOTE]
> **What this repository is.** This is a **design article and reference architecture** — a detailed blueprint distilled from prototyping a hybrid retrieval system. It explains *how* to build a Graph-Powered RAG engine and *why* each choice matters. A runnable reference implementation is tracked in the [Roadmap](#roadmap); until it lands, treat the code snippets here as illustrative pseudocode, not a shipped library.

---

## Table of Contents

- [Why Hybrid Retrieval?](#why-hybrid-retrieval)
- [The Core Idea: Vectors + Graphs](#the-core-idea-vectors--graphs)
- [System Architecture](#system-architecture)
- [How It Works, Component by Component](#how-it-works-component-by-component)
  - [1. Ingestion](#1-ingestion)
  - [2. Graph Construction](#2-graph-construction)
  - [3. Vector Search](#3-vector-search)
  - [4. Graph Expansion & Hybrid Scoring](#4-graph-expansion--hybrid-scoring)
  - [5. Answer Composition & Traceability](#5-answer-composition--traceability)
  - [6. Serving Layer](#6-serving-layer)
- [Evaluating a Hybrid Retriever](#evaluating-a-hybrid-retriever)
- [Engineering Lessons](#engineering-lessons)
- [Where This Is Useful](#where-this-is-useful)
- [Proposed Repository Structure](#proposed-repository-structure)
- [Roadmap](#roadmap)
- [References](#references)
- [License](#license)

---

## Why Hybrid Retrieval?

Modern Retrieval-Augmented Generation (RAG) leans almost entirely on **semantic embeddings**: text is mapped into high-dimensional vectors so that semantically similar passages sit close together. This works well, but pure vector retrieval has three structural weaknesses:

<div align="center">

| Limitation | What goes wrong |
|---|---|
| **No structure** | Vectors encode similarity, not relationships. They can't express *"A cites B"* or *"C builds on D."* |
| **No explainability** | You can rank passages by cosine distance, but you can't easily justify *why* two items are related beyond "the model said so." |
| **Context collapse** | Items that are related but not textually similar (e.g., a definition and its distant application) are often missed entirely. |
</div>

Knowledge graphs address exactly these gaps. A graph represents information as **nodes** (entities) and **edges** (typed relationships), giving you hierarchy, explicit linkage, and an auditable trail.

## The Core Idea: Vectors + Graphs

The two representations are complementary rather than competing:

- **Vectors capture meaning** — fuzzy semantic proximity.
- **Graphs capture connections** — explicit, typed, traversable relationships.

Used together, they enable *reasoned retrieval*: the vector layer finds plausible candidates, and the graph layer expands, re-ranks, and — crucially — **explains** them. That explanation is the whole point; it is the difference between a system that returns an answer and one a domain expert is willing to trust.

## System Architecture

```
                       ┌──────────────────────┐
                       │      Documents       │
                       └──────────┬───────────┘
                                  ▼
                    ┌───────────────────────────┐
                    │  Chunking + Concept       │
                    │  Extraction (NER/KeyBERT) │
                    └──────────┬────────────────┘
                               ▼
        ┌───────────────┬──────────────┬────────────────┐
        ▼               ▼              ▼                │
 ┌────────────┐  ┌─────────────┐  ┌────────────┐        │
 │ Vector DB  │  │ Graph Store │  │  Metadata  │        │
 │  (FAISS)   │  │ (NetworkX / │  │ (docs, IDs)│        │
 │            │  │   Neo4j)    │  │            │        │
 └─────┬──────┘  └──────┬──────┘  └─────┬──────┘        │
       └───────────┬────┴───────────────┘               │
                   ▼                                    │
          ┌────────────────────┐                        │
          │  Hybrid Retriever  │◄───────────────────────┘
          │  (vector + graph)  │
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │  Answer Composer   │
          │ + citations + paths│
          └─────────┬──────────┘
                    ▼
        ┌──────────────────────────┐
        │  FastAPI service (/ask)  │  ── REST API layer
        └─────────┬────────────────┘
                  ▼
        ┌──────────────────────────┐
        │   Streamlit front-end    │  ── calls the API, renders
        │  (answer, cites, paths)  │     answers, citations, graph paths
        └──────────────────────────┘
```

> The API and UI are drawn as separate layers on purpose: a **FastAPI** service owns retrieval and exposes `/ask`, while the **Streamlit** app is a thin client that calls it. This split keeps the engine reusable outside the demo UI.

## How It Works, Component by Component

### 1. Ingestion

Ingestion turns raw documents into a structured, searchable knowledge base:

- **Chunking** — split long text into semantic units (~256–512 tokens works well for retrieval; larger chunks dilute relevance).
- **Concept extraction** — pull key entities/terms with an NER model (spaCy) or a keyword model (KeyBERT).
- **Embedding** — encode each chunk with a sentence transformer, e.g. `all-MiniLM-L6-v2` (384-dimensional output; fast and a strong baseline).
- **Storage** — write vectors to **FAISS** for approximate nearest-neighbor (ANN) search, and materialize nodes/edges in the **graph store**.

Example chunk record:

```json
{
  "id": "doc1_chunk_0",
  "text": "FAISS is a library for efficient similarity search and clustering of dense vectors.",
  "concepts": ["faiss", "similarity search", "clustering"],
  "doc_id": "doc1",
  "url": "file://docs/faiss_notes.md"
}
```

### 2. Graph Construction

Edges are what make retrieval explainable. A minimal, expressive schema:

- `(:Doc)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Concept)`
- `(:Concept)-[:RELATED_TO]->(:Concept)`
- `(:Author)-[:WROTE]->(:Doc)`

You can then use graph algorithms to inform ranking. To estimate document authority with **PageRank in Neo4j**, use the Graph Data Science (GDS) library — note this is a genuine PageRank computation, not a mention count:

```cypher
// Project a subgraph, then run PageRank over it
CALL gds.graph.project(
  'docGraph',
  ['Doc', 'Concept'],
  { MENTIONS: { orientation: 'UNDIRECTED' } }
)

CALL gds.pageRank.stream('docGraph')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).title AS title, score
ORDER BY score DESC;
```

> On a small local graph you can skip Neo4j entirely and call `networkx.pagerank(G)` — same algorithm, one line.

### 3. Vector Search

FAISS handles dense similarity efficiently. One detail matters for correctness: to get **cosine** similarity you must L2-normalize vectors and use an inner-product index, not the default L2 index.

```python
import faiss, numpy as np

# Cosine similarity = inner product on L2-normalized vectors
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.normalize_L2(query_vector)
scores, ids = index.search(query_vector, k=10)  # higher score = more similar
```

These candidates are only the starting point — the graph layer refines them next.

### 4. Graph Expansion & Hybrid Scoring

After the initial ANN retrieval:

1. Take the top-*k* chunks.
2. Expand along their graph neighborhood (sibling chunks, shared concepts, related documents).
3. Re-rank the merged candidate set with a **hybrid score**.

The critical subtlety the naive version gets wrong: the three signals live on **different scales** (cosine ∈ [0,1], concept overlap as Jaccard ∈ [0,1], but raw PageRank values sum to 1 across the whole graph and are therefore tiny). Blend them only *after* normalizing each into a common range:

```python
def hybrid_score(sim, overlap, pr, norms):
    # norms holds min/max (or mean/std) collected over the candidate set
    sim_n     = minmax(sim,     norms.sim)      # semantic similarity  -> [0,1]
    overlap_n = minmax(overlap, norms.overlap)  # symbolic overlap     -> [0,1]
    pr_n      = minmax(pr,      norms.pr)        # graph authority      -> [0,1]
    return 0.60 * sim_n + 0.25 * overlap_n + 0.15 * pr_n
```

Without that normalization, the PageRank term contributes essentially nothing and the "authority" weight is a no-op. Treat `0.60 / 0.25 / 0.15` as a starting point to tune against a labeled set, not a universal constant.

### 5. Answer Composition & Traceability

The composer assembles the top passages into a response. Two modes:

- **Extractive** (no LLM): stitch retrieved passages together with citations — deterministic, cheap, fully grounded.
- **Generative** (with an LLM): pass the passages as context for a fluent answer. *This* is the "G" in RAG; the extractive mode is retrieval-plus-extraction and is worth distinguishing explicitly.

Either way, every answer ships with its provenance:

> **Answer.** FAISS is a library for efficient similarity search and clustering of dense vectors, enabling approximate nearest-neighbor retrieval at scale.
>
> **Sources.** `faiss_notes.md`
>
> **Why these sources.** Query concept *similarity* → linked via *embedding* → mentioned in `faiss_notes.md`.

That "why" trail — the actual graph path from query to evidence — is the feature most pure-LLM pipelines can't produce.

### 6. Serving Layer

A **FastAPI** service exposes the engine (`POST /ask`), and a **Streamlit** front-end calls it to let users:

- ask a question and read the answer with inline citations,
- expand a *"Why these?"* panel to inspect the graph path,
- browse document-similarity recommendations.

Keeping the engine behind an API (rather than inside the Streamlit script) means the same retrieval logic can back a UI, a batch job, or another service without change.

## Evaluating a Hybrid Retriever

If explainability is the selling point, **measurement is the proof**. Track at minimum:

- **Recall@k** — fraction of relevant chunks retrieved in the top *k*.
- **Grounding rate** — fraction of answer claims traceable to a cited source.
- **Latency** — end-to-end p50/p95, since graph expansion adds cost.

> [!IMPORTANT]
> The table below is an **illustrative template**, not measured results — it shows the shape of an ablation you should run, so vector-only can be compared against the hybrid design on your own corpus. Fill it with real numbers before citing any improvement.

<div align="center">

| Configuration | Recall@10 | Grounding rate | p95 latency |
|---|---|---|---|
| Vector-only (FAISS) | *your baseline* | *your baseline* | *your baseline* |
| + Graph expansion | *measure* | *measure* | *measure* |
| + Hybrid re-ranking | *measure* | *measure* | *measure* |
</div>

Publishing this table with real numbers is what turns "graphs help" from a claim into a result.

## Engineering Lessons

**Graphs and vectors are complementary, not interchangeable.** The vector layer maximizes recall of plausible candidates; the graph layer adds precision, connective context, and an audit trail. Neither alone gives you both.

**Design for swappability.** Keep the embedding model, vector store, graph store, and UI behind clean interfaces. You *will* want to swap MiniLM for a larger encoder, or NetworkX for Neo4j, without touching the rest.

**Explainability is a UX feature, not a footnote.** A visible graph path is what converts a skeptical domain expert into a user. Build the "why" surface as a first-class part of the product.

**Start local, scale later.** NetworkX + FAISS + Streamlit runs on a laptop and is enough to validate the idea. Move to Neo4j, a managed vector DB, and hosted LLMs only once the design earns it.

**Measure before you claim.** "Higher recall" and "better answers" mean nothing without the ablation above. Instrument recall@k, grounding, and latency from day one.

## Where This Is Useful

<div align="center">

| Domain | Use case |
|---|---|
| Enterprise knowledge | Explainable Q&A over internal docs with citations |
| Healthcare & law | Traceable reasoning where provenance is mandatory |
| Education | Tutors that show *why* an answer follows from sources |
| Research | Graph-enhanced literature exploration |
| Recommendation | Content- and author-graph suggestions |
</div>

## Proposed Repository Structure

The layout the reference implementation will follow:

```
.
├── src/
│   ├── ingest/          # chunking, concept extraction, embedding
│   ├── graph/           # graph construction + PageRank
│   ├── retrieve/        # FAISS search, graph expansion, hybrid scoring
│   ├── compose/         # extractive + generative answer composition
│   └── api/             # FastAPI service (/ask)
├── app/                 # Streamlit front-end
├── eval/                # recall@k, grounding, latency harness
├── data/                # sample corpus
├── requirements.txt
└── README.md
```

## Roadmap

- [ ] Reference implementation of the ingestion → retrieval → compose pipeline
- [ ] FastAPI `/ask` service + Streamlit client
- [ ] Evaluation harness with a sample corpus and a filled-in ablation table
- [ ] Optional Neo4j GDS backend for PageRank at scale
- [ ] Screenshots / demo GIF of the "Why these?" graph-path panel

## References

1. Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. — [FAISS](https://github.com/facebookresearch/faiss)
2. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
3. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP. — [SentenceTransformers](https://www.sbert.net/)
4. Page, L., et al. (1999). *The PageRank Citation Ranking.* — implemented in [Neo4j GDS](https://neo4j.com/docs/graph-data-science/current/) and [NetworkX](https://networkx.org/)
5. [Streamlit](https://streamlit.io/) · [FastAPI](https://fastapi.tiangolo.com/) · [KeyBERT](https://github.com/MaartenGr/KeyBERT)

## License

Released under the [MIT License](LICENSE).
