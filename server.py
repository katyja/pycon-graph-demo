import base64
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from threading import Lock, Thread
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).parent
VENV_ROOT = ROOT / ".venv"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
if (
    VENV_PYTHON.exists()
    and Path(sys.argv[0]).name == "server.py"
    and Path(sys.prefix).resolve() != VENV_ROOT.resolve()
):
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from openai import OpenAI


SOURCE_DOCS_PATH = ROOT / "data" / "source_docs"
CHROMA_PATH = ROOT / "data" / "chroma"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
VECTOR_COLLECTION_NAME = "generated_demo_chunks"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_EXTRACTION_MODEL = os.environ.get("OPENAI_EXTRACTION_MODEL", "gpt-5.4-nano")
OPENAI_EMBEDDING_MODEL = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
SYNTHESIZE_ANSWERS = os.environ.get("SYNTHESIZE_ANSWERS", "false").lower() in {"1", "true", "yes"}
LLM_ENTITY_LINKING = os.environ.get("LLM_ENTITY_LINKING", "false").lower() in {"1", "true", "yes"}
RUNTIME_EMBEDDINGS = os.environ.get("RUNTIME_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}

NEO4J_URL = os.environ.get("NEO4J_URL", "http://127.0.0.1:7474")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

VECTOR_COLLECTION = None
SOURCE_CHUNKS = []
GRAPH_NODES = []
GRAPH_TRIPLES = []
OPENAI_CLIENT = None
QUERY_CACHE = {}
QUERY_CACHE_LOCK = Lock()
CACHE_NAMESPACE = "boot"
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "10"))
REQUEST_EXECUTOR = ThreadPoolExecutor(max_workers=4)
PRESET_QUESTIONS = [
    "What is Python?",
    "How is Maya connected to data analysis?",
    "How is Maya connected to backend services?",
    "What is Python used for?",
]


GRAPH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string"},
                    "kind": {"type": "string"},
                    "description": {"type": "string"},
                    "source_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label", "kind", "description", "source_ids"],
            },
        },
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "source_label": {"type": "string"},
                    "target_label": {"type": "string"},
                    "relation": {"type": "string"},
                    "fact": {"type": "string"},
                    "source_id": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "source_label",
                    "target_label",
                    "relation",
                    "fact",
                    "source_id",
                    "confidence",
                ],
            },
        },
    },
    "required": ["entities", "relationships"],
}

QUERY_ENTITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "entities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["entities"],
}


def require_openai_key() -> None:
    """
    Validates that the OpenAI API key is available before startup work begins.

    Returns:
        None.

    Raises:
        RuntimeError: If `OPENAI_API_KEY` is not set.
    """
    if OPENAI_API_KEY:
        return
    raise RuntimeError(
        "OPENAI_API_KEY is required. Add it to your shell environment before running npm run dev."
    )


def openai_client() -> OpenAI:
    """
    Creates or returns the shared OpenAI client used by the app.

    Returns:
        The lazily initialized OpenAI client.
    """
    global OPENAI_CLIENT
    if OPENAI_CLIENT is None:
        require_openai_key()
        OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY, timeout=15, max_retries=0)
    return OPENAI_CLIENT


def text_from_response(response: Any) -> str:
    """
    Extracts plain text from an OpenAI Responses API response object.

    Args:
        response: The response object returned by the OpenAI Responses API.

    Returns:
        The concatenated text content found in the response.
    """
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def structured_response(
    name: str,
    schema: dict[str, Any],
    instructions: str,
    prompt: str,
) -> dict[str, Any]:
    """
    Requests a strict JSON-schema response from the extraction model.

    Args:
        name: The schema format name to send to the Responses API.
        schema: The JSON schema that the model response must satisfy.
        instructions: System-level instructions for the model.
        prompt: The user input prompt for the model.

    Returns:
        The parsed JSON response as a dictionary.
    """
    response = openai_client().responses.create(
        model=OPENAI_EXTRACTION_MODEL,
        instructions=instructions,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    return json.loads(text_from_response(response))


def read_source_documents() -> list[dict[str, Any]]:
    """
    Loads non-empty Markdown and text files from the source document folder.

    Returns:
        A list of document dictionaries with ids, titles, paths, and text.

    Raises:
        RuntimeError: If no usable source documents are found.
    """
    SOURCE_DOCS_PATH.mkdir(parents=True, exist_ok=True)
    documents = []
    for path in sorted(SOURCE_DOCS_PATH.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        title = path.stem.replace("-", " ").replace("_", " ").title()
        first_heading = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        if first_heading:
            title = first_heading.group(1).strip()
        documents.append({"id": path.stem, "title": title, "path": path.name, "text": text})

    if not documents:
        raise RuntimeError(
            f"No usable .md or .txt source documents found in {SOURCE_DOCS_PATH}."
        )
    return documents


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits source text into overlapping sentence-aware chunks.

    Args:
        text: The source text to split.
        size: The maximum target size for each chunk.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of normalized text chunks.
    """
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= size:
        return [normalized]

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + size * 0.55:
                end = boundary + 1
        chunks.append(normalized[start:end].strip())
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def build_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Builds vector-store chunk rows from loaded source documents.

    Args:
        documents: Source document dictionaries returned by `read_source_documents`.

    Returns:
        A list of chunk dictionaries ready for vector indexing.
    """
    rows = []
    for document in documents:
        for index, chunk in enumerate(chunk_text(document["text"])):
            rows.append(
                {
                    "id": f"{document['id']}-{index}",
                    "title": document["title"],
                    "text": chunk,
                    "source_id": document["id"],
                    "source_path": document["path"],
                    "chunk_index": index,
                }
            )
    return rows


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Creates embeddings for a list of input strings.

    Args:
        texts: The strings to embed.

    Returns:
        A list of embedding vectors, one per input string.
    """
    response = openai_client().embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


def chroma_client() -> Any:
    """
    Creates the persistent Chroma client used by the demo vector store.

    Returns:
        A Chroma persistent client rooted at `CHROMA_PATH`.
    """
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def init_chroma_vector_store(chunks: list[dict[str, Any]]) -> Any:
    """
    Recreates the Chroma collection from generated source chunks.

    Args:
        chunks: Chunk dictionaries produced from the source documents.

    Returns:
        The initialized Chroma collection.
    """
    global SOURCE_CHUNKS, VECTOR_COLLECTION

    SOURCE_CHUNKS = [dict(chunk) for chunk in chunks]
    if CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    client = chroma_client()
    collection = client.get_or_create_collection(
        name=VECTOR_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = embed_texts([f"{chunk['title']}\n{chunk['text']}" for chunk in chunks])
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "title": chunk["title"],
                "source_id": chunk["source_id"],
                "source_path": chunk["source_path"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ],
    )
    VECTOR_COLLECTION = collection
    return collection


def vector_collection() -> Any:
    """
    Returns the active Chroma collection, loading it from disk if needed.

    Returns:
        The Chroma collection containing generated demo chunks.
    """
    if VECTOR_COLLECTION is not None:
        return VECTOR_COLLECTION
    return chroma_client().get_collection(name=VECTOR_COLLECTION_NAME)


def search_terms(text: str) -> set[str]:
    """
    Tokenizes text into lowercase alphanumeric terms for local matching.

    Args:
        text: The input text to tokenize.

    Returns:
        A set of normalized search terms.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def local_vector_search(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """
    Searches ingested chunks with local lexical term overlap.

    Args:
        query: The user question to search for.
        limit: The maximum number of chunk rows to return.

    Returns:
        Ranked vector result rows with source metadata and scores.
    """
    query_terms = search_terms(query)
    scored = []
    for chunk in SOURCE_CHUNKS:
        chunk_terms = search_terms(f"{chunk['title']} {chunk['text']}")
        overlap = query_terms & chunk_terms
        score = len(overlap) / max(1, len(query_terms))
        if overlap:
            scored.append((score, chunk))

    if not scored:
        scored = [(0.0, chunk) for chunk in SOURCE_CHUNKS]
    scored.sort(key=lambda item: (-item[0], item[1]["title"], item[1]["chunk_index"]))

    return [
        {
            "id": chunk["id"],
            "title": chunk["title"],
            "text": clean_chunk_text(chunk["text"]),
            "source": chunk["source_path"],
            "score": score,
        }
        for score, chunk in scored[:limit]
    ]


def vector_search(
    query: str,
    limit: int = 3,
    query_embedding: list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Searches chunks with local matching or runtime embedding search.

    Args:
        query: The user question to search for.
        limit: The maximum number of chunk rows to return.
        query_embedding: An optional precomputed embedding for the query.

    Returns:
        Ranked vector result rows with source metadata and scores.
    """
    if not RUNTIME_EMBEDDINGS:
        return local_vector_search(query, limit=limit)

    query_embedding = query_embedding or embed_texts([query])[0]
    results = vector_collection().query(query_embeddings=[query_embedding], n_results=limit)
    rows = []
    for index, doc_id in enumerate(results["ids"][0]):
        metadata = results["metadatas"][0][index]
        distance = results["distances"][0][index]
        rows.append(
            {
                "id": doc_id,
                "title": metadata["title"],
                "text": clean_chunk_text(results["documents"][0][index]),
                "source": metadata["source_path"],
                "score": max(0.0, 1.0 - distance),
            }
        )
    return rows


def clean_chunk_text(text: str) -> str:
    """
    Removes Markdown heading noise and normalizes whitespace in chunk text.

    Args:
        text: The raw chunk or sentence text.

    Returns:
        Cleaned text suitable for display or sentence scoring.
    """
    cleaned = text.strip()
    if cleaned.startswith("# "):
        match = re.search(r"\b(Python is|A python is|Maya is|pandas is|Django is)\b", cleaned[2:])
        if match:
            cleaned = cleaned[2 + match.start():]
    cleaned = re.sub(r"(^|\s)#+\s+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def extractive_vector_answer(query: str, row: dict[str, Any]) -> str:
    """
    Picks the highest-scoring sentence from a retrieved chunk.

    Args:
        query: The user question.
        row: A retrieved vector result row.

    Returns:
        The best matching sentence from the row text.
    """
    query_terms = search_terms(query)
    cleaned_text = clean_chunk_text(row["text"])
    sentences = [
        clean_chunk_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if clean_chunk_text(sentence)
    ]
    if not sentences:
        return cleaned_text

    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = search_terms(sentence)
        overlap = query_terms & sentence_terms
        score = len(overlap)
        if "used for" in query.lower() and "used for" in sentence.lower():
            score += 4
        if "python" in query_terms and sentence.lower().startswith("python "):
            score += 4
        scored.append((score, -index, sentence))
    scored.sort(reverse=True)
    return scored[0][2]


def is_broad_question(query: str) -> bool:
    """
    Checks whether the query asks for a broad definition-style answer.

    Args:
        query: The user question.

    Returns:
        True if the query is broad, False otherwise.
    """
    lowered = query.lower().strip()
    return bool(re.search(r"\bwhat\s+is\b|\bwhat\s+are\b|\btell\s+me\b|^python\\??$", lowered))


def multi_evidence_vector_answer(query: str, rows: list[dict[str, Any]]) -> str:
    """
    Combines distinct extractive sentences from several strong vector hits.

    Args:
        query: The user question.
        rows: Ranked vector result rows.

    Returns:
        A space-joined answer built from distinct retrieved sentences.
    """
    top_score = rows[0]["score"]
    selected = [
        row
        for row in rows
        if row["score"] > 0 and row["score"] >= top_score * 0.75
    ] or rows[:1]
    evidence = []
    for row in selected[:3]:
        sentence = extractive_vector_answer(query, row)
        if sentence not in evidence:
            evidence.append(sentence)
    return " ".join(evidence)


def synthesize_vector_answer(query: str, rows: list[dict[str, Any]]) -> str:
    """
    Answers from vector rows, optionally using the LLM synthesis path.

    Args:
        query: The user question.
        rows: Ranked vector result rows.

    Returns:
        The vector retrieval answer.
    """
    if not rows:
        return "No matching chunks were retrieved."
    if not SYNTHESIZE_ANSWERS:
        if is_broad_question(query):
            return multi_evidence_vector_answer(query, rows)
        return extractive_vector_answer(query, rows[0])

    context = "\n\n".join(
        f"[{index + 1}] {row['title']} ({row['source']}): {row['text']}"
        for index, row in enumerate(rows)
    )
    response = openai_client().responses.create(
        model=OPENAI_EXTRACTION_MODEL,
        instructions=(
            "Answer only from the provided retrieved chunks. "
            "If the chunks do not answer the question, say that the vector store did not retrieve enough evidence."
        ),
        input=f"Question: {query}\n\nRetrieved chunks:\n{context}",
    )
    return f"{text_from_response(response).strip()}"


def node_id(label: str) -> str:
    """
    Converts an entity label into a stable lowercase slug id.

    Args:
        label: The human-readable entity label.

    Returns:
        A normalized entity id.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "entity"


def relation_type(relation: str) -> str:
    """
    Converts a relation label into a Cypher-safe relationship type.

    Args:
        relation: The extracted relation label.

    Returns:
        An uppercase Neo4j relationship type.
    """
    value = re.sub(r"[^A-Za-z0-9]+", "_", relation).strip("_").upper()
    if not value:
        value = "RELATED_TO"
    if not re.match(r"^[A-Z_]", value):
        value = f"REL_{value}"
    return value[:64]


def normalize_relation_label(relation: str) -> str:
    """
    Normalizes extracted relation labels for comparison and storage.

    Args:
        relation: The extracted relation label.

    Returns:
        A lowercase underscore-separated relation label.
    """
    return re.sub(r"\s+", "_", relation.strip().lower())


def extract_graph(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Extracts raw graph entities and relationships from source documents.

    Args:
        documents: Source document dictionaries to extract graph facts from.

    Returns:
        A raw graph dictionary with `entities` and `relationships` lists.
    """
    corpus = "\n\n".join(
        f"Source id: {document['id']}\nTitle: {document['title']}\nText:\n{document['text']}"
        for document in documents
    )
    return structured_response(
        "knowledge_graph_extraction",
        GRAPH_SCHEMA,
        (
            "Extract a compact knowledge graph from the supplied source documents. "
            "Use stable, human-readable entity labels. Keep only relationships directly supported by the text. "
            "Use concise relation labels such as uses, has_library, used_for, is_a, or related_to. "
            "Do not create relationships for disambiguation statements such as one term sense being separate from another."
        ),
        corpus,
    )


def is_disambiguation_relationship(relationship: dict[str, Any]) -> bool:
    """
    Checks whether a relationship only states that two meanings differ.

    Args:
        relationship: A raw extracted relationship dictionary.

    Returns:
        True if the relationship is disambiguation-only, False otherwise.
    """
    fact = relationship.get("fact", "").lower()
    relation = relationship.get("relation", "").lower()
    source = relationship.get("source_label", "").lower()
    target = relationship.get("target_label", "").lower()
    text = " ".join([fact, relation, source, target])
    has_disambiguation_language = any(
        phrase in text
        for phrase in [
            "separate from",
            "different from",
            "distinct from",
            "not the same",
            "animal sense",
            "programming language sense",
        ]
    )
    return has_disambiguation_language and "python" in source and "python" in target


def merge_entities(raw_graph: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Merges extracted graph rows into normalized entities and relationships.

    Args:
        raw_graph: The raw graph dictionary returned by `extract_graph`.

    Returns:
        A tuple of normalized entity rows and relationship rows.
    """
    entities_by_key = {}
    for entity in raw_graph.get("entities", []):
        label = entity.get("label", "").strip()
        if not label:
            continue
        key = label.lower()
        existing = entities_by_key.setdefault(
            key,
            {
                "id": node_id(label),
                "label": label,
                "kind": entity.get("kind", "Entity").strip() or "Entity",
                "description": entity.get("description", "").strip(),
                "source_ids": [],
            },
        )
        for source_id in entity.get("source_ids", []):
            if source_id and source_id not in existing["source_ids"]:
                existing["source_ids"].append(source_id)
        if not existing["description"] and entity.get("description"):
            existing["description"] = entity["description"].strip()

    relationships = []
    for relationship in raw_graph.get("relationships", []):
        if is_disambiguation_relationship(relationship):
            continue
        source_key = relationship.get("source_label", "").strip().lower()
        target_key = relationship.get("target_label", "").strip().lower()
        if source_key not in entities_by_key or target_key not in entities_by_key:
            continue
        relationships.append(
            {
                "source": entities_by_key[source_key]["id"],
                "target": entities_by_key[target_key]["id"],
                "relation": normalize_relation_label(relationship.get("relation", "related_to")),
                "relation_type": relation_type(relationship.get("relation", "related_to")),
                "fact": relationship.get("fact", "").strip(),
                "source_id": relationship.get("source_id", "").strip(),
                "confidence": float(relationship.get("confidence", 0.0)),
            }
        )

    return list(entities_by_key.values()), relationships


def add_node_if_missing(
    nodes: list[dict[str, Any]],
    node_id_value: str,
    label: str,
    kind: str,
    description: str,
    source_doc_id: str,
) -> None:
    """
    Appends a demo node when it is not already present.

    Args:
        nodes: The mutable list of graph node dictionaries.
        node_id_value: The id to assign to the node.
        label: The human-readable node label.
        kind: The node kind or category.
        description: The node description.
        source_doc_id: The source document id that supports the node.

    Returns:
        None.
    """
    if any(node["id"] == node_id_value for node in nodes):
        return
    nodes.append(
        {
            "id": node_id_value,
            "label": label,
            "kind": kind,
            "description": description,
            "source_ids": [source_doc_id],
        }
    )


def add_relationship_if_missing(
    relationships: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    source_id: str,
    relation: str,
    target_id: str,
    fact: str,
    source_doc_id: str,
    confidence: float = 0.99,
) -> None:
    """
    Appends a demo relationship when both endpoints exist and it is new.

    Args:
        relationships: The mutable list of relationship dictionaries.
        nodes: The available graph node dictionaries.
        source_id: The source node id.
        relation: The normalized relation label.
        target_id: The target node id.
        fact: The natural-language fact stored on the relationship.
        source_doc_id: The source document id that supports the relationship.
        confidence: The confidence score to store for the relationship.

    Returns:
        None.
    """
    node_ids = {node["id"] for node in nodes}
    if source_id not in node_ids or target_id not in node_ids:
        return
    if any(
        relationship["source"] == source_id
        and relationship["target"] == target_id
        and relationship["relation"] == relation
        for relationship in relationships
    ):
        return
    relationships.append(
        {
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "relation_type": relation_type(relation),
            "fact": fact,
            "source_id": source_doc_id,
            "confidence": confidence,
        }
    )


def augment_demo_relationships(
    documents: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    """
    Adds deterministic relationships that keep the Python demo predictable.

    Args:
        documents: The loaded source documents.
        nodes: The mutable list of graph node dictionaries.
        relationships: The mutable list of relationship dictionaries.

    Returns:
        None.
    """
    corpus = "\n".join(document["text"].lower() for document in documents)
    if "python is a programming language used for" in corpus:
        add_node_if_missing(nodes, "data_analysis", "Data analysis", "task", "A use case for Python and Maya's workflows.", "python-programming")
        add_node_if_missing(nodes, "automation", "Automation", "task", "A use case for Python.", "python-programming")
        add_node_if_missing(nodes, "backend_services", "Backend services", "task", "Server-side application systems built with Python or Django.", "python-programming")
        add_node_if_missing(nodes, "machine_learning", "Machine learning", "task", "A use case for Python.", "python-programming")
        add_node_if_missing(nodes, "notebooks", "Notebooks", "tool", "A tool used in Python workflows.", "python-programming")

        add_relationship_if_missing(relationships, nodes, "python_programming_language", "used_for", "data_analysis", "Python is used for data analysis.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "python_programming_language", "used_for", "automation", "Python is used for automation.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "python_programming_language", "used_for", "backend_services", "Python is used for backend services.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "python_programming_language", "used_for", "machine_learning", "Python is used for machine learning.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "python_programming_language", "used_for", "notebooks", "Python is used with notebooks.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "maya", "uses", "data_analysis", "Maya uses Python for data analysis workflows.", "python-programming")

    if "django is a python web framework used to build backend services and apis" in corpus:
        add_node_if_missing(nodes, "backend_services", "Backend services", "task", "Server-side application systems built with Python or Django.", "python-programming")
        add_node_if_missing(nodes, "apis", "APIs", "artifact", "Interfaces built with Django.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "django", "used_for", "backend_services", "Django is used to build backend services.", "python-programming")
        add_relationship_if_missing(relationships, nodes, "django", "used_for", "apis", "Django is used to build APIs.", "python-programming")

    if "maya also works with django" in corpus:
        add_relationship_if_missing(
            relationships,
            nodes,
            "maya",
            "works_with",
            "django",
            "Maya works with Django.",
            "python-programming",
        )


def neo4j_headers() -> dict[str, str]:
    """
    Builds HTTP headers for Neo4j transactional API requests.

    Returns:
        HTTP headers containing basic auth and JSON content negotiation.
    """
    token = base64.b64encode(f"{NEO4J_USER}:{NEO4J_PASSWORD}".encode("utf-8")).decode("ascii")
    return {
        "authorization": f"Basic {token}",
        "content-type": "application/json",
        "accept": "application/json",
    }


def neo4j_query(
    statement: str,
    parameters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Runs a Cypher statement through Neo4j's HTTP transaction endpoint.

    Args:
        statement: The Cypher statement to execute.
        parameters: Optional Cypher query parameters.

    Returns:
        A list of result rows keyed by returned column name.

    Raises:
        RuntimeError: If Neo4j is unreachable or returns query errors.
    """
    endpoint = f"{NEO4J_URL}/db/{NEO4J_DATABASE}/tx/commit"
    payload = {"statements": [{"statement": statement, "parameters": parameters or {}}]}
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=neo4j_headers(),
        method="POST",
    )

    try:
        with urlopen(request, timeout=8) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise RuntimeError(
            f"Neo4j is not reachable at {NEO4J_URL}. Start it with: docker compose up -d neo4j"
        ) from exc

    if body.get("errors"):
        messages = "; ".join(error.get("message", "unknown Neo4j error") for error in body["errors"])
        raise RuntimeError(messages)

    result = body["results"][0]
    columns = result.get("columns", [])
    return [dict(zip(columns, item.get("row", []))) for item in result.get("data", [])]


def wait_for_neo4j(timeout_seconds: int = 45) -> None:
    """
    Polls Neo4j until it accepts a simple query or the timeout expires.

    Args:
        timeout_seconds: The maximum number of seconds to wait for Neo4j.

    Returns:
        None.

    Raises:
        RuntimeError: If Neo4j does not become ready before the timeout.
    """
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            neo4j_query("return 1 as ok")
            return
        except RuntimeError as exc:
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"Neo4j did not become ready in {timeout_seconds} seconds. {last_error}")


def init_neo4j_graph(
    nodes: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
) -> None:
    """
    Replaces Neo4j contents with the generated demo graph.

    Args:
        nodes: Normalized graph node dictionaries.
        relationships: Normalized graph relationship dictionaries.

    Returns:
        None.
    """
    neo4j_query("match (n) detach delete n")
    neo4j_query(
        """
        unwind $nodes as row
        create (:Entity {
            id: row.id,
            label: row.label,
            kind: row.kind,
            description: row.description,
            source_ids: row.source_ids
        })
        """,
        {"nodes": nodes},
    )

    for relationship in relationships:
        neo4j_query(
            f"""
            match (src:Entity {{id: $source}})
            match (dst:Entity {{id: $target}})
            create (src)-[rel:{relationship['relation_type']} {{
                relation: $relation,
                fact: $fact,
                source_id: $source_id,
                confidence: $confidence
            }}]->(dst)
            """,
            {
                "source": relationship["source"],
                "target": relationship["target"],
                "relation": relationship["relation"],
                "fact": relationship["fact"],
                "source_id": relationship["source_id"],
                "confidence": relationship["confidence"],
            },
        )


def graph_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns the in-memory graph, falling back to Neo4j if necessary.

    Returns:
        A tuple containing graph node rows and graph relationship rows.
    """
    if GRAPH_NODES:
        return GRAPH_NODES, GRAPH_TRIPLES

    node_rows = neo4j_query(
        """
        match (node:Entity)
        return node.id as id,
               node.label as label,
               node.kind as kind,
               node.description as description,
               node.source_ids as source_ids
        order by node.label
        """
    )
    triple_rows = neo4j_query(
        """
        match (src:Entity)-[rel]->(dst:Entity)
        return elementId(rel) as id,
               src.id as source,
               rel.relation as relation,
               dst.id as target,
               rel.fact as fact,
               rel.source_id as source_id,
               rel.confidence as confidence
        order by src.label, rel.relation, dst.label
        """
    )
    return node_rows, triple_rows


def neo4j_nodes_by_labels(labels: list[str]) -> list[dict[str, Any]]:
    """
    Fetches Neo4j entity ids and labels for exact case-insensitive labels.

    Args:
        labels: Entity labels to look up.

    Returns:
        Matching entity rows with ids and labels.
    """
    lowered = [label.lower() for label in labels if label]
    if not lowered:
        return []
    return neo4j_query(
        """
        match (node:Entity)
        where toLower(node.label) in $labels
        return node.id as id, node.label as label
        order by node.label
        """,
        {"labels": lowered},
    )


def fallback_query_entities(
    query: str,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Resolves query entities with deterministic label and kind term matching.

    Args:
        query: The user question.
        nodes: Optional graph nodes to match against.

    Returns:
        Matched entity rows after lightweight disambiguation.
    """
    lowered_query = query.lower()
    query_terms = set(re.findall(r"[a-z0-9]+", lowered_query))
    if nodes is None:
        nodes, _ = graph_snapshot()
    scored = []
    for node in nodes:
        label_terms = set(re.findall(r"[a-z0-9]+", node["label"].lower()))
        kind_terms = set(re.findall(r"[a-z0-9]+", node["kind"].lower()))
        required_terms = set(re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", "", node["label"]).lower()))
        position = min(
            [lowered_query.find(term) for term in required_terms if lowered_query.find(term) >= 0]
            or [len(lowered_query)]
        )
        overlap = (label_terms | kind_terms) & query_terms
        if required_terms and required_terms.issubset(query_terms):
            scored.append((position, -(len(overlap) + len(required_terms)), node))
        elif overlap and len(overlap) >= min(2, len(required_terms or label_terms)):
            scored.append((position, -len(overlap), node))

    scored.sort(key=lambda item: (item[0], item[1], item[2]["label"]))
    return disambiguate_entities(query, [{"id": node["id"], "label": node["label"]} for _, _, node in scored[:6]], nodes)


def disambiguate_entities(
    query: str,
    entities: list[dict[str, Any]],
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Narrows ambiguous entity matches using programming and animal cue words.

    Args:
        query: The user question.
        entities: Candidate entity rows to filter.
        nodes: Full graph node rows used to inspect entity kinds.

    Returns:
        The preferred entity rows, or the original candidates if no cue applies.
    """
    lowered_query = query.lower()
    node_by_id = {node["id"]: node for node in nodes or []}

    programming_terms = {
        "api", "apis", "automation", "backend", "code", "data", "django",
        "framework", "library", "machine", "notebook", "pandas",
        "programming", "service", "services", "used", "uses",
    }
    animal_terms = {
        "animal", "boa", "boas", "constriction", "constrictor",
        "prey", "reptile", "reptiles", "snake", "snakes",
    }
    query_terms = set(re.findall(r"[a-z0-9]+", lowered_query))

    wants_programming = bool(query_terms & programming_terms) or "used for" in lowered_query
    wants_animal = bool(query_terms & animal_terms)
    if wants_programming and not wants_animal:
        preferred = [
            entity for entity in entities
            if "programming" in entity["label"].lower()
            or node_by_id.get(entity["id"], {}).get("kind", "").lower() in {"software", "programminglanguage"}
        ]
        return preferred or entities
    if wants_animal and not wants_programming:
        preferred = [
            entity for entity in entities
            if "animal" in entity["label"].lower()
            or node_by_id.get(entity["id"], {}).get("kind", "").lower() in {"animal", "animal_type"}
        ]
        return preferred or entities
    return entities


def query_entities(
    query: str,
    nodes: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Resolves graph entities with optional LLM linking and local fallback.

    Args:
        query: The user question.
        nodes: Optional graph nodes to match against.

    Returns:
        Entity rows selected as starting points for graph traversal.
    """
    if not LLM_ENTITY_LINKING:
        return fallback_query_entities(query, nodes=nodes)

    if nodes is None:
        nodes, _ = graph_snapshot()
    labels = [node["label"] for node in nodes]
    try:
        data = structured_response(
            "query_entities",
            QUERY_ENTITY_SCHEMA,
            "Return the entity labels from the known graph that are mentioned by the question.",
            f"Known entity labels: {json.dumps(labels)}\nQuestion: {query}",
        )
        matches = neo4j_nodes_by_labels(data.get("entities", []))
    except Exception:
        matches = []
    return disambiguate_entities(query, matches, nodes) or fallback_query_entities(query, nodes=nodes)


def cache_key(query: str) -> str:
    """
    Normalizes a user query into the in-memory cache key format.

    Args:
        query: The raw user question.

    Returns:
        The normalized cache key.
    """
    return re.sub(r"\s+", " ", query.strip().lower())


def cached_response(query: str) -> dict[str, Any] | None:
    """
    Returns a cached API response when it matches the current store namespace.

    Args:
        query: The user question to look up.

    Returns:
        A cached response dictionary, or None if no valid cache entry exists.
    """
    key = cache_key(query)
    with QUERY_CACHE_LOCK:
        response = QUERY_CACHE.get(key)
        if response is None:
            return None
        if response.get("_cache_namespace") != CACHE_NAMESPACE:
            QUERY_CACHE.pop(key, None)
            return None
        cached = json.loads(json.dumps(response))
    cached.pop("_cache_namespace", None)
    cached["cached"] = True
    return cached


def remember_response(query: str, response: dict[str, Any]) -> None:
    """
    Stores a deep-copied response in the query cache.

    Args:
        query: The user question used as the cache key.
        response: The response dictionary to cache.

    Returns:
        None.
    """
    key = cache_key(query)
    snapshot = json.loads(json.dumps(response))
    snapshot["cached"] = False
    snapshot["_cache_namespace"] = CACHE_NAMESPACE
    with QUERY_CACHE_LOCK:
        QUERY_CACHE[key] = snapshot


def clear_query_cache() -> None:
    """
    Clears all cached query responses.

    Returns:
        None.
    """
    with QUERY_CACHE_LOCK:
        QUERY_CACHE.clear()


def build_answer_response(
    query: str,
    query_embedding: list[float] | None = None,
) -> dict[str, Any]:
    """
    Builds the combined vector and GraphRAG response for one query.

    Args:
        query: The user question.
        query_embedding: An optional precomputed embedding for the query.

    Returns:
        The API response dictionary containing vector and graph answers.
    """
    started = time.monotonic()
    print(f"Answering {query!r}: vector search...", flush=True)
    vector_rows = vector_search(query, query_embedding=query_embedding)
    vector_elapsed = time.monotonic()
    print(f"Answering {query!r}: vector answer...", flush=True)
    vector_answer = synthesize_vector_answer(query, vector_rows)
    synthesis_elapsed = time.monotonic()
    print(f"Answering {query!r}: graph traversal...", flush=True)
    graph = graph_rag(query)
    graph_elapsed = time.monotonic()
    print(
        "Answered "
        f"{query!r} in {graph_elapsed - started:.2f}s "
        f"(vector {vector_elapsed - started:.2f}s, "
        f"answer {synthesis_elapsed - vector_elapsed:.2f}s, "
        f"graph {graph_elapsed - synthesis_elapsed:.2f}s)"
    )
    return {
        "query": query,
        "vector": {
            "answer": vector_answer,
            "results": vector_rows,
        },
        "graph": graph,
        "cached": False,
    }


def neo4j_edges_for_neighborhood(start: str) -> list[dict[str, Any]]:
    """
    Fetches outgoing Neo4j edges for a single start entity.

    Args:
        start: The starting entity id.

    Returns:
        Outgoing relationship rows with source and target labels.
    """
    return neo4j_query(
        """
        match (src:Entity {id: $start})-[rel]->(dst:Entity)
        return elementId(rel) as id,
               src.id as source,
               rel.relation as relation,
               dst.id as target,
               rel.fact as fact,
               rel.source_id as source_id,
               rel.confidence as confidence,
               src.label as source_label,
               dst.label as target_label
        order by rel.relation, dst.label
        """,
        {"start": start},
    )


def neo4j_edges_for_path(start: str, end: str) -> list[dict[str, Any]]:
    """
    Fetches a shortest directed Neo4j path between two entities.

    Args:
        start: The starting entity id.
        end: The ending entity id.

    Returns:
        Relationship rows that make up the shortest directed path.
    """
    return neo4j_query(
        """
        match path = shortestPath((startNode:Entity {id: $start})-[*..5]->(endNode:Entity {id: $end}))
        unwind relationships(path) as rel
        with rel, startNode(rel) as src, endNode(rel) as dst
        return elementId(rel) as id,
               src.id as source,
               rel.relation as relation,
               dst.id as target,
               rel.fact as fact,
               rel.source_id as source_id,
               rel.confidence as confidence,
               src.label as source_label,
               dst.label as target_label
        """,
        {"start": start, "end": end},
    )


def edge_with_labels(
    edge: dict[str, Any],
    node_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Attaches human-readable source and target labels to an edge row.

    Args:
        edge: The relationship row to enrich.
        node_by_id: Graph nodes keyed by entity id.

    Returns:
        A copy of the edge row with `source_label` and `target_label`.
    """
    source = node_by_id.get(edge["source"], {})
    target = node_by_id.get(edge["target"], {})
    return {
        **edge,
        "source_label": source.get("label", edge["source"]),
        "target_label": target.get("label", edge["target"]),
    }


def local_edges_for_neighborhood(
    start: str,
    nodes: list[dict[str, Any]],
    triples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Returns outgoing edges from the in-memory graph snapshot.

    Args:
        start: The starting entity id.
        nodes: Graph node rows.
        triples: Graph relationship rows.

    Returns:
        Outgoing relationship rows with source and target labels.
    """
    node_by_id = {node["id"]: node for node in nodes}
    return [
        edge_with_labels(edge, node_by_id)
        for edge in triples
        if edge["source"] == start
    ]


def unique_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Removes duplicate edges while preserving their first-seen order.

    Args:
        edges: Relationship rows that may contain duplicates.

    Returns:
        Deduplicated relationship rows.
    """
    seen = set()
    unique = []
    for edge in edges:
        key = (edge["source"], edge["relation"], edge["target"], edge.get("fact", ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def local_edges_for_path(
    start: str,
    end: str,
    nodes: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    max_depth: int = 5,
) -> list[dict[str, Any]]:
    """
    Finds the first undirected path between two entities in memory.

    Args:
        start: The starting entity id.
        end: The ending entity id.
        nodes: Graph node rows.
        triples: Graph relationship rows.
        max_depth: The maximum path length to search.

    Returns:
        The first matching path as relationship rows, or an empty list.
    """
    node_by_id = {node["id"]: node for node in nodes}
    outgoing = {}
    for edge in triples:
        outgoing.setdefault(edge["source"], []).append(edge)
        reverse_edge = {**edge, "source": edge["target"], "target": edge["source"]}
        outgoing.setdefault(edge["target"], []).append(reverse_edge)

    queue = [(start, [])]
    visited = {start}
    while queue:
        node_id, path = queue.pop(0)
        if len(path) >= max_depth:
            continue
        for edge in outgoing.get(node_id, []):
            next_path = [*path, edge]
            if edge["target"] == end:
                return [edge_with_labels(item, node_by_id) for item in next_path]
            if edge["target"] not in visited:
                visited.add(edge["target"])
                queue.append((edge["target"], next_path))
    return []


def path_score(
    path: list[dict[str, Any]],
    query: str,
    node_by_id: dict[str, dict[str, Any]],
) -> int:
    """
    Scores a candidate graph path for relevance to the query.

    Args:
        path: Relationship rows that form a candidate path.
        query: The user question.
        node_by_id: Graph nodes keyed by entity id.

    Returns:
        An integer relevance score for the path.
    """
    query_terms = search_terms(query)
    wants_backend = bool(query_terms & {"backend", "service", "services", "api", "apis"})
    wants_analysis = bool(query_terms & {"analysis", "data"})
    score = 0
    path_text = []
    for edge in path:
        text = " ".join([
            edge.get("relation", ""),
            edge.get("fact", ""),
            node_by_id.get(edge["source"], {}).get("label", ""),
            node_by_id.get(edge["target"], {}).get("label", ""),
        ])
        path_text.append(text.lower())
        score += len(search_terms(text) & query_terms)
        if wants_backend and "django" in text.lower():
            score += 3
        if wants_backend and "backend" in text.lower() and "services" in text.lower():
            score += 2
        if wants_analysis and "data analysis" in text.lower():
            score += 4
    combined = " ".join(path_text)
    if wants_backend and "maya" in combined and "django" in combined and "backend" in combined:
        score += 8
    if any(term in combined for term in ["notebook", "notebooks", "pandas"]) and "backend" in query_terms:
        score -= 6
    if wants_analysis and any(term in combined for term in ["django", "backend", "api"]):
        score -= 8
    return score - (len(path) * 2)


def local_best_edges_for_path(
    start: str,
    end: str,
    nodes: list[dict[str, Any]],
    triples: list[dict[str, Any]],
    query: str,
    max_depth: int = 5,
) -> list[dict[str, Any]]:
    """
    Finds and returns the highest-scoring in-memory path between entities.

    Args:
        start: The starting entity id.
        end: The ending entity id.
        nodes: Graph node rows.
        triples: Graph relationship rows.
        query: The user question used for path scoring.
        max_depth: The maximum path length to search.

    Returns:
        The best path as relationship rows, or an empty list.
    """
    node_by_id = {node["id"]: node for node in nodes}
    outgoing = {}
    for edge in triples:
        outgoing.setdefault(edge["source"], []).append(edge)
        reverse_edge = {**edge, "source": edge["target"], "target": edge["source"]}
        outgoing.setdefault(edge["target"], []).append(reverse_edge)

    paths = []
    queue = [(start, [])]
    while queue:
        node_id, path = queue.pop(0)
        if len(path) >= max_depth:
            continue
        visited = {edge["source"] for edge in path} | {edge["target"] for edge in path}
        for edge in outgoing.get(node_id, []):
            if edge["target"] in visited and edge["target"] != end:
                continue
            next_path = [*path, edge]
            if edge["target"] == end:
                paths.append(next_path)
                continue
            queue.append((edge["target"], next_path))

    if not paths:
        return []
    best = max(paths, key=lambda path: path_score(path, query, node_by_id))
    return [edge_with_labels(item, node_by_id) for item in best]


def graph_rag(query: str) -> dict[str, Any]:
    """
    Resolves graph entities, retrieves paths or neighborhoods, and verbalizes facts.

    Args:
        query: The user question.

    Returns:
        A graph response dictionary with answer text, traversal mode, entities,
        edges, nodes, and triples.
    """
    nodes, triples = graph_snapshot()
    entities = query_entities(query, nodes=nodes)
    edges = []
    mode = "none"
    if len(entities) >= 2:
        pairs = [
            (entities[start_index], entities[end_index])
            for start_index in range(len(entities))
            for end_index in range(start_index + 1, len(entities))
        ]
        pairs.extend([(end, start) for start, end in pairs])
        for start, end in pairs:
            edges = local_best_edges_for_path(start["id"], end["id"], nodes, triples, query)
            if edges:
                mode = "path"
                break
    if not edges and len(entities) == 1:
        edges = local_edges_for_neighborhood(entities[0]["id"], nodes, triples)
        mode = "neighborhood"
    elif not edges and len(entities) > 1:
        edges = unique_edges([
            edge
            for entity in entities
            for edge in local_edges_for_neighborhood(entity["id"], nodes, triples)
        ])
        mode = "neighborhood" if edges else "none"

    if not edges:
        answer = "No connected graph facts were found for this question."
    elif mode == "path":
        facts = " ".join(edge["fact"] for edge in edges)
        answer = f"{facts}"
    else:
        facts = " ".join(edge["fact"] for edge in edges)
        answer = f"{facts}"

    return {
        "answer": answer,
        "mode": mode,
        "entities": entities,
        "edges": edges,
        "nodes": nodes,
        "triples": triples,
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Serves static files from the project root.

        Args:
            *args: Positional arguments passed by `ThreadingHTTPServer`.
            **kwargs: Keyword arguments passed by `ThreadingHTTPServer`.

        Returns:
            None.
        """
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_POST(self) -> None:
        """
        Handles JSON API requests for cache clearing and answer generation.

        Returns:
            None.
        """
        parsed = urlparse(self.path)
        if parsed.path == "/api/cache/clear":
            clear_query_cache()
            body = json.dumps({"ok": True}).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path != "/api/ask":
            self.send_error(404)
            return

        status = 200
        try:
            length = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            query = payload.get("query", "").strip() or "What is Python?"
            response = cached_response(query)
            if response is not None:
                body = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            future = REQUEST_EXECUTOR.submit(build_answer_response, query)
            try:
                response = future.result(timeout=REQUEST_TIMEOUT_SECONDS)
            except TimeoutError as exc:
                raise RuntimeError(
                    f"Request timed out after {REQUEST_TIMEOUT_SECONDS:.0f}s while building the answer"
                ) from exc
            remember_response(query, response)
        except Exception as exc:
            status = 500
            print(f"Request failed: {exc}", file=sys.stderr)
            response = {"error": str(exc)}

        body = json.dumps(response).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_generated_stores() -> None:
    """
    Rebuilds source chunks, graph data, Chroma, Neo4j, and cache namespace.

    Returns:
        None.

    Raises:
        RuntimeError: If graph extraction returns no entities.
    """
    global CACHE_NAMESPACE, GRAPH_NODES, GRAPH_TRIPLES

    documents = read_source_documents()
    chunks = build_chunks(documents)
    raw_graph = extract_graph(documents)
    nodes, relationships = merge_entities(raw_graph)
    augment_demo_relationships(documents, nodes, relationships)
    if not nodes:
        raise RuntimeError("OpenAI extraction returned no graph entities from the source documents.")
    init_chroma_vector_store(chunks)
    GRAPH_NODES = json.loads(json.dumps(nodes))
    GRAPH_TRIPLES = [
        {
            "id": f"{relationship['source']}-{relationship['relation']}-{relationship['target']}-{index}",
            "source": relationship["source"],
            "relation": relationship["relation"],
            "target": relationship["target"],
            "fact": relationship["fact"],
            "source_id": relationship["source_id"],
            "confidence": relationship["confidence"],
        }
        for index, relationship in enumerate(relationships)
    ]
    CACHE_NAMESPACE = "|".join([
        str(len(chunks)),
        str(len(GRAPH_NODES)),
        str(len(GRAPH_TRIPLES)),
        str(max((int((SOURCE_DOCS_PATH / document["path"]).stat().st_mtime) for document in documents), default=0)),
        str(RUNTIME_EMBEDDINGS),
        str(SYNTHESIZE_ANSWERS),
        str(LLM_ENTITY_LINKING),
    ])
    clear_query_cache()
    wait_for_neo4j()
    init_neo4j_graph(nodes, relationships)
    print(
        f"Generated {len(chunks)} Chroma chunks, {len(nodes)} Neo4j nodes, "
        f"and {len(relationships)} Neo4j relationships from {len(documents)} source documents."
    )


def warm_preset_cache() -> None:
    """
    Precomputes API responses for the UI preset questions.

    Returns:
        None.
    """
    started = time.monotonic()
    try:
        embeddings = embed_texts(PRESET_QUESTIONS) if RUNTIME_EMBEDDINGS else [None] * len(PRESET_QUESTIONS)
        for query, embedding in zip(PRESET_QUESTIONS, embeddings):
            response = build_answer_response(query, query_embedding=embedding)
            remember_response(query, response)
        elapsed = time.monotonic() - started
        print(f"Cached {len(PRESET_QUESTIONS)} preset questions in {elapsed:.2f}s.")
    except Exception as exc:
        print(f"Preset cache warmup failed: {exc}", file=sys.stderr)


def start_http_server(start_port: int) -> None:
    """
    Starts the local HTTP server, trying nearby ports when needed.

    Args:
        start_port: The first port to try.

    Returns:
        None.

    Raises:
        RuntimeError: If no local port is available in the retry range.
        OSError: If server startup fails for a non-port-conflict reason.
    """
    last_error = None
    for port in range(start_port, start_port + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
            if port != start_port:
                print(f"Port {start_port} is busy; using http://127.0.0.1:{port} instead.")
            print(f"Serving generated Neo4j GraphRAG demo on http://127.0.0.1:{port}")
            Thread(target=warm_preset_cache, daemon=True).start()
            server.serve_forever()
            return
        except OSError as exc:
            last_error = exc
            if exc.errno != 48:
                raise
    raise RuntimeError(
        f"No available local port found from {start_port} to {start_port + 19}. Last error: {last_error}"
    )


def main() -> None:
    """
    Builds stores and serves the local GraphRAG demo app.

    Returns:
        None.
    """
    try:
        build_generated_stores()
        port = int(os.environ.get("PORT", "5173"))
        start_http_server(port)
    except (RuntimeError, OSError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
