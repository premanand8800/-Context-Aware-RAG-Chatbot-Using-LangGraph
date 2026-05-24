const uploadForm = document.querySelector("#upload-form");
const chatForm = document.querySelector("#chat-form");
const fileInput = document.querySelector("#pdf-file");
const documentSelect = document.querySelector("#document-select");
const questionInput = document.querySelector("#question");
const messages = document.querySelector("#messages");
const statusLine = document.querySelector("#status");
const documentCount = document.querySelector("#document-count");
const activeDocument = document.querySelector("#active-document");
const uploadButton = document.querySelector("#upload-button");
const sendButton = document.querySelector("#send-button");
const sessionId = crypto.randomUUID();
let documentsCache = [];

function setStatus(text) {
  statusLine.textContent = text;
}

function setBusy(isBusy) {
  uploadButton.disabled = isBusy;
  sendButton.disabled = isBusy;
}

function updateActiveDocument() {
  const selected = documentsCache.find((item) => item.document_id === documentSelect.value);
  activeDocument.textContent = selected
    ? `${selected.filename} · ${selected.chunks} chunks`
    : "No document selected";
}

function renderEmptyState() {
  messages.innerHTML = "";
  const node = document.createElement("article");
  node.className = "empty-state";
  node.textContent = "Select an indexed document and ask a question about the paper.";
  messages.appendChild(node);
}

function addMessage(role, text, citations = []) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.textContent = text;

  if (citations.length > 0) {
    const citeNode = document.createElement("div");
    citeNode.className = "citations";
    for (const item of citations.slice(0, 4)) {
      const citation = document.createElement("span");
      citation.className = "citation";
      citation.textContent = `${item.filename} · chunk ${item.chunk_index}`;
      citeNode.appendChild(citation);
    }
    node.appendChild(citeNode);
  }

  messages.appendChild(node);
  messages.scrollTop = messages.scrollHeight;
  return node;
}

function setMessageText(node, text) {
  node.textContent = text;
}

function setMessageCitations(node, citations = []) {
  const existing = node.querySelector(".citations");
  if (existing) existing.remove();
  if (citations.length === 0) return;

  const citeNode = document.createElement("div");
  citeNode.className = "citations";
  for (const item of citations.slice(0, 4)) {
    const citation = document.createElement("span");
    citation.className = "citation";
    citation.textContent = `${item.filename} · chunk ${item.chunk_index}`;
    citeNode.appendChild(citation);
  }
  node.appendChild(citeNode);
  messages.scrollTop = messages.scrollHeight;
}

function parseSseChunk(buffer, onEvent) {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop();
  for (const part of parts) {
    const line = part
      .split("\n")
      .find((item) => item.startsWith("data: "));
    if (!line) continue;
    onEvent(JSON.parse(line.slice(6)));
  }
  return remainder;
}

async function loadDocuments() {
  const response = await fetch("/api/documents");
  documentsCache = await response.json();
  documentSelect.innerHTML = "";
  documentCount.textContent = `${documentsCache.length} ${
    documentsCache.length === 1 ? "document" : "documents"
  }`;

  if (documentsCache.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No documents uploaded";
    documentSelect.appendChild(option);
    updateActiveDocument();
    return;
  }

  for (const documentInfo of documentsCache) {
    const option = document.createElement("option");
    option.value = documentInfo.document_id;
    option.textContent = `${documentInfo.filename} (${documentInfo.chunks} chunks)`;
    documentSelect.appendChild(option);
  }
  updateActiveDocument();
  setStatus("Indexed paper ready. Ask a question or ingest another PDF.");
}

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!fileInput.files.length) {
    setStatus("Choose a PDF first.");
    return;
  }

  const body = new FormData();
  body.append("file", fileInput.files[0]);
  setStatus("Uploading and ingesting. This can take a minute for long papers.");
  setBusy(true);

  const response = await fetch("/api/documents/upload", {
    method: "POST",
    body,
  });
  const payload = await response.json();
  setBusy(false);

  if (!response.ok) {
    setStatus(payload.detail || "Upload failed.");
    return;
  }

  await loadDocuments();
  documentSelect.value = payload.document_id;
  setStatus(`Ingested ${payload.filename} with ${payload.chunks} chunks.`);
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const documentId = documentSelect.value || null;
  addMessage("user", question);
  questionInput.value = "";
  setStatus("Thinking...");
  setBusy(true);

  const assistantNode = addMessage("assistant", "");
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      question,
      document_id: documentId,
      session_id: sessionId,
    }),
  });

  if (!response.ok) {
    const payload = await response.json();
    addMessage("assistant", payload.detail || "Chat failed.");
    assistantNode.remove();
    setBusy(false);
    setStatus("");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseChunk(buffer, (payload) => {
      if (payload.type === "status") {
        setStatus(payload.message);
      } else if (payload.type === "answer") {
        setMessageText(assistantNode, payload.answer);
      } else if (payload.type === "done") {
        setMessageText(assistantNode, payload.answer);
        setMessageCitations(assistantNode, payload.citations);
        setStatus("");
      } else if (payload.type === "error") {
        setMessageText(assistantNode, payload.message);
        setStatus("");
      }
    });
  }
  setBusy(false);
});

documentSelect.addEventListener("change", updateActiveDocument);

renderEmptyState();
loadDocuments().catch(() => setStatus("Could not load documents."));
