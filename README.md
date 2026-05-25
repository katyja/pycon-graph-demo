# GraphRAG Mini Lab

A local demo app that builds both retrieval stores from editable source documents:

- Vector retrieval reads `data/source_docs`, creates OpenAI embeddings, stores chunks in Chroma, and answers from retrieved chunks. By default, runtime queries use a local lexical search over the ingested chunks; set `RUNTIME_EMBEDDINGS=true` to query Chroma with OpenAI embeddings on each request.
- GraphRAG reads the same source documents, uses OpenAI structured outputs to extract entities and relationships, stores them in Neo4j, and answers from generated graph paths or one-hop neighborhoods. The app keeps an in-memory graph snapshot for request-time traversal and uses Neo4j as the persisted graph store and visual inspection target.

The example avoids real sponsor brands by using the ambiguity of `Python`.

## Run it

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY="..."
```

Start Neo4j and the app:

```bash
npm run neo4j
npm run dev
```

Then open:

```text
http://127.0.0.1:5173
```

If you run the backend directly, use:

```bash
.venv/bin/python server.py
```

## Configuration

Required:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_EXTRACTION_MODEL`, default `gpt-5.4-nano`
- `OPENAI_EMBEDDING_MODEL`, default `text-embedding-3-small`
- `SYNTHESIZE_ANSWERS`, default `false`; set to `true` to use an LLM to synthesize vector answers on each request
- `LLM_ENTITY_LINKING`, default `false`; set to `true` to use an LLM to map each question to graph entities
- `RUNTIME_EMBEDDINGS`, default `false`; set to `true` to use OpenAI embeddings on each request instead of local search over ingested chunks
- `REQUEST_TIMEOUT_SECONDS`, default `10`; maximum time allowed for an `/api/answer` request
- `PORT`, default `5173`
- `NEO4J_URL`, default `http://127.0.0.1:7474`
- `NEO4J_USER`, default `neo4j`
- `NEO4J_PASSWORD`, default `password`
- `NEO4J_DATABASE`, default `neo4j`

Neo4j runs from `docker-compose.yml` on:

- Browser: `http://127.0.0.1:7474`
- Username: `neo4j`
- Password: `password`

## How Generation Works

On startup, `server.py` rebuilds the demo stores:

1. Reads non-empty `.md` and `.txt` files from `data/source_docs`.
2. Splits the documents into chunks.
3. Calls OpenAI embeddings and stores the vectors in Chroma under `data/chroma`.
4. Calls OpenAI structured outputs to extract graph entities and relationships.
5. Adds a few deterministic demo relationships for the Python examples, such as Python use cases and Maya/Django/backend-service links.
6. Normalizes extracted relation labels into Cypher-safe Neo4j relationship types.
7. Recreates Neo4j `Entity` nodes and typed relationships from the extracted graph.
8. Warms the cache for the preset questions in the UI.

This means editing a source document and restarting the server changes both retrieval paths.

## Demo Flow

Try:

- `What is Python?`
- `How is Maya connected to data analysis?`
- `How is Maya connected to backend services?`
- `What is Python used for?`

Vector retrieval answers from retrieved chunks. Broad questions, such as `What is Python?`, can combine evidence from several top chunks; narrower questions return the best matching sentence unless `SYNTHESIZE_ANSWERS=true`.

GraphRAG resolves entities from the question, then either finds a path between multiple matched entities or returns the outgoing neighborhood for a single matched entity. It currently verbalizes answers by concatenating the graph edge facts, so path answers may look like `Maya works with Django. Django is used to build backend services.` rather than a synthesized sentence.
