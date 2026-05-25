let selectedNode = null;
let graphData = { nodes: [], triples: [], edges: [] };

const queryInput = document.getElementById("queryInput");
const vectorAnswer = document.getElementById("vectorAnswer");
const vectorResults = document.getElementById("vectorResults");
const graphAnswer = document.getElementById("graphAnswer");
const pathSteps = document.getElementById("pathSteps");
const graphSvg = document.getElementById("graphSvg");
const selectedNodeLabel = document.getElementById("selectedNodeLabel");
const inspectorTitle = document.getElementById("inspectorTitle");
const inspectorDescription = document.getElementById("inspectorDescription");
const inspectorFacts = document.getElementById("inspectorFacts");
const presetButtons = document.querySelectorAll(".preset");

/**
 * Escapes text before inserting it into HTML markup.
 *
 * @param {*} value - The value to convert to a safe HTML string.
 * @returns {string} The escaped HTML string.
 */
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Formats a retrieval score as a whole-number percentage.
 *
 * @param {number} score - The normalized score returned by the backend.
 * @returns {string} The score formatted as a percentage label.
 */
function scoreLabel(score) {
  return `${Math.max(0, Math.round(score * 100))}%`;
}

/**
 * Sends a question to the backend comparison endpoint.
 *
 * @param {string} query - The user question to answer.
 * @returns {Promise<object>} The parsed API response containing vector and graph results.
 * @throws {Error} When the backend returns a non-OK response.
 */
async function ask(query) {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query })
  });
  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(data.error || `Request failed with ${response.status}`);
  }

  return data;
}

/**
 * Renders the vector retrieval answer and ranked source chunks.
 *
 * @param {object} vector - The vector response payload.
 * @param {string} vector.answer - The vector retrieval answer text.
 * @param {Array<object>} vector.results - The retrieved chunk rows.
 * @returns {void}
 */
function renderVector(vector) {
  vectorAnswer.innerHTML = `
    <strong>Answer from vector DB</strong>
    <span>${escapeHtml(vector.answer)}</span>
  `;

  vectorResults.innerHTML = vector.results.map((result, index) => `
    <div class="result">
      <div class="result-top">
        <span class="result-title">${index + 1}. ${escapeHtml(result.title)}</span>
        <span class="score">${scoreLabel(result.score)}</span>
      </div>
      <p>${escapeHtml(result.text)}</p>
    </div>
  `).join("");
}

/**
 * Renders the GraphRAG answer and traversed graph facts.
 *
 * @param {object} graph - The graph response payload.
 * @param {string} graph.answer - The GraphRAG answer text.
 * @param {Array<object>} graph.edges - The traversed graph edges.
 * @returns {void}
 */
function renderGraph(graph) {
  graphAnswer.innerHTML = `
    <strong>Answer from GraphRAG</strong>
    <span>${escapeHtml(graph.answer)}</span>
  `;

  pathSteps.innerHTML = graph.edges.length
    ? graph.edges.map((edge, index) => `
      <div class="step">
        <div class="step-top">
          <span class="step-title">${escapeHtml(edge.source_label)} -> ${escapeHtml(edge.target_label)}</span>
          <span class="score">${index + 1}</span>
        </div>
        <p><strong>${escapeHtml(edge.relation)}</strong>: ${escapeHtml(edge.fact)}</p>
      </div>
    `).join("")
    : `
      <div class="step">
        <div class="step-top">
          <span class="step-title">No traversed facts</span>
          <span class="score">0</span>
        </div>
        <p>The graph only answers from explicit relationships stored in Neo4j.</p>
      </div>
    `;
}

/**
 * Finds a graph node by its id in the active graph data.
 *
 * @param {string|null} id - The graph node id to look up.
 * @returns {object|undefined} The matching node, if one exists.
 */
function nodeById(id) {
  return graphData.nodes.find((node) => node.id === id);
}

/**
 * Builds a stable edge identifier from source, relation, and target ids.
 *
 * @param {object} edge - The graph edge to identify.
 * @param {string} edge.source - The source node id.
 * @param {string} edge.relation - The relationship label.
 * @param {string} edge.target - The target node id.
 * @returns {string} The stable edge identifier.
 */
function edgeId(edge) {
  return `${edge.source}-${edge.relation}-${edge.target}`;
}

/**
 * Calculates circular SVG positions for the current graph nodes.
 *
 * @returns {Object<string, {x: number, y: number}>} Node positions keyed by node id.
 */
function graphPositions() {
  const width = 780;
  const height = 430;
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = 290;
  const radiusY = 150;
  const positions = {};

  graphData.nodes.forEach((node, index) => {
    const angle = graphData.nodes.length === 1
      ? 0
      : (Math.PI * 2 * index / graphData.nodes.length) - Math.PI / 2;
    positions[node.id] = {
      x: centerX + Math.cos(angle) * radiusX,
      y: centerY + Math.sin(angle) * radiusY
    };
  });

  return positions;
}

/**
 * Renders the interactive SVG graph and binds node click handlers.
 *
 * @returns {void}
 */
function renderGraphSvg() {
  const positions = graphPositions();
  const hotEdges = new Set(graphData.edges.map(edgeId));
  const hotNodes = new Set(graphData.edges.flatMap((edge) => [edge.source, edge.target]));

  const edgeMarkup = graphData.triples.map((edge) => {
    const from = positions[edge.source];
    const to = positions[edge.target];
    if (!from || !to) {
      return "";
    }
    const midX = (from.x + to.x) / 2;
    const midY = (from.y + to.y) / 2 - 10;
    const hot = hotEdges.has(edgeId(edge)) ? " hot" : "";

    return `
      <line class="edge${hot}" x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}"></line>
      <text class="edge-label" x="${midX}" y="${midY}">${escapeHtml(edge.relation)}</text>
    `;
  }).join("");

  const nodeMarkup = graphData.nodes.map((node) => {
    const position = positions[node.id];
    if (!position) {
      return "";
    }
    const classes = [
      "node",
      hotNodes.has(node.id) ? "hot" : "",
      selectedNode === node.id ? "selected" : ""
    ].filter(Boolean).join(" ");
    const radius = Math.max(38, Math.min(68, node.label.length * 5.5));

    return `
      <g class="${classes}" data-node="${node.id}" transform="translate(${position.x} ${position.y})">
        <circle r="${radius}"></circle>
        <text>${escapeHtml(node.label)}</text>
      </g>
    `;
  }).join("");

  graphSvg.innerHTML = `
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L9,3 z" fill="currentColor"></path>
      </marker>
    </defs>
    ${edgeMarkup}
    ${nodeMarkup}
  `;

  graphSvg.querySelectorAll(".node").forEach((element) => {
    element.addEventListener("click", () => {
      selectedNode = element.dataset.node;
      renderInspector();
      renderGraphSvg();
    });
  });
}

/**
 * Renders details and outgoing facts for the selected graph node.
 *
 * @returns {void}
 */
function renderInspector() {
  const node = nodeById(selectedNode) || graphData.nodes[0];
  if (!node) {
    return;
  }
  selectedNode = node.id;

  const facts = graphData.triples.filter((edge) => edge.source === node.id);
  selectedNodeLabel.textContent = `${node.label} selected`;
  inspectorTitle.textContent = node.label;
  inspectorDescription.textContent = `${node.kind}. ${node.description}`;

  inspectorFacts.innerHTML = facts.length
    ? facts.map((edge) => {
      const target = nodeById(edge.target);
      return `
        <div class="fact">
          <strong>${escapeHtml(edge.relation)} -> ${escapeHtml(target.label)}</strong>
          <p>${escapeHtml(edge.fact)}</p>
        </div>
      `;
    }).join("")
    : `<div class="fact"><strong>No outgoing triples</strong><p>This node is an endpoint in the graph store.</p></div>`;
}

/**
 * Toggles the loading style on the query input.
 *
 * @param {boolean} isLoading - Whether the UI should show a loading state.
 * @returns {void}
 */
function setLoading(isLoading) {
  queryInput.classList.toggle("is-loading", isLoading);
}

/**
 * Runs the vector-vs-GraphRAG comparison for the current query.
 *
 * @returns {Promise<void>}
 */
async function runComparison() {
  const query = queryInput.value.trim() || "What is Python?";
  setLoading(true);
  queryInput.classList.add("is-loading");

  try {
    const data = await ask(query);
    if (data.error) {
      throw new Error(data.error);
    }
    graphData = {
      nodes: data.graph.nodes,
      triples: data.graph.triples,
      edges: data.graph.edges
    };
    if (!selectedNode || !nodeById(selectedNode)) {
      selectedNode = graphData.edges[0]?.source || graphData.nodes[0]?.id || null;
    }
    renderVector(data.vector);
    renderGraph(data.graph);
    renderInspector();
    renderGraphSvg();
  } catch (error) {
    vectorAnswer.innerHTML = `<strong>Request failed</strong><span>${escapeHtml(error.message)}</span>`;
    graphAnswer.innerHTML = `<strong>Request failed</strong><span>Check that server.py is running.</span>`;
  } finally {
    setLoading(false);
    queryInput.classList.remove("is-loading");
  }
}

queryInput.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    runComparison();
  }
});

presetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    presetButtons.forEach((candidate) => candidate.classList.remove("active"));
    button.classList.add("active");
    queryInput.value = button.dataset.query;
    runComparison();
  });
});

runComparison();
