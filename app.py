from __future__ import annotations

import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from rag.config import Settings
from rag.logging_config import configure_logging
from rag.retriever import RAGRetriever


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RAG ChatBot</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7fb;
      --panel: #ffffff;
      --line: #dde2ea;
      --text: #172033;
      --muted: #667085;
      --accent: #1769e0;
      --accent-soft: #e8f1ff;
      --ok: #087443;
      --bad: #9b1c1c;
      --warn: #8a4b00;
      --warn-bg: #fff4dc;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
    }
    .app {
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 24px;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      min-height: 100vh;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.88);
      padding: 20px 28px;
    }
    h1, h2, p { margin: 0; }
    h1 { font-size: 24px; line-height: 1.2; }
    h2 { font-size: 15px; margin-bottom: 14px; }
    .subtitle { color: var(--muted); margin-top: 6px; font-size: 14px; }
    .status {
      display: grid;
      gap: 12px;
      margin-top: 18px;
    }
    .status-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcff;
    }
    .label { color: var(--muted); font-size: 12px; }
    .value { font-weight: 700; margin-top: 4px; }
    .components {
      display: grid;
      gap: 8px;
      margin-top: 18px;
    }
    .component {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--line);
      padding: 7px 0;
      color: var(--muted);
      font-size: 13px;
    }
    .component:last-child { border-bottom: 0; }
    .badge {
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .badge.ok {
      background: #e7f6ef;
      color: var(--ok);
    }
    .badge.bad {
      background: #fdecec;
      color: var(--bad);
    }
    .warning {
      display: none;
      margin-top: 14px;
      color: var(--warn);
      background: var(--warn-bg);
      border: 1px solid #ffd58c;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      line-height: 1.5;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      padding: 11px 14px;
    }
    button:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }
    .secondary {
      width: 100%;
      margin-top: 18px;
      background: var(--accent-soft);
      color: var(--accent);
    }
    .hint {
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    #messages {
      overflow-y: auto;
      padding: 24px 28px;
    }
    .message {
      max-width: 900px;
      margin: 0 auto 18px;
      display: flex;
    }
    .message.user { justify-content: flex-end; }
    .bubble {
      max-width: min(720px, 100%);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: var(--panel);
      white-space: pre-wrap;
      line-height: 1.65;
    }
    .user .bubble {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }
    details {
      max-width: 900px;
      margin: -6px auto 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 10px 14px;
    }
    summary { cursor: pointer; font-weight: 700; }
    .source {
      border-top: 1px solid var(--line);
      margin-top: 10px;
      padding-top: 10px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 13px;
    }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 28px;
    }
    textarea {
      width: 100%;
      min-height: 46px;
      max-height: 150px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      font: inherit;
    }
    @media (max-width: 780px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      main { min-height: 70vh; }
      form { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h2>知识库状态</h2>
      <div class="status">
        <div class="status-item">
          <div class="label">运行模式</div>
          <div class="value" id="mode">加载中</div>
        </div>
        <div class="status-item">
          <div class="label">文档切片</div>
          <div class="value" id="chunks">0</div>
        </div>
        <div class="status-item">
          <div class="label">向量索引</div>
          <div class="value" id="vectors">0</div>
        </div>
      </div>
      <div class="components" id="components"></div>
      <div class="warning" id="warning"></div>
      <button class="secondary" id="rebuild" type="button">重建知识库索引</button>
      <p class="hint">把 .txt 或 .md 文件放入 data/documents 后，点击重建知识库索引即可重新生成向量库。</p>
    </aside>
    <main>
      <header>
        <h1>RAG ChatBot</h1>
        <p class="subtitle">先用 Embedding 检索 ChromaDB 中的知识库上下文，再交给 LLM 生成带来源的回答。</p>
      </header>
      <section id="messages"></section>
      <form id="chat-form">
        <textarea id="prompt" placeholder="输入你的问题..." required></textarea>
        <button id="send" type="submit">发送</button>
      </form>
    </main>
  </div>
  <script>
    const messages = document.querySelector("#messages");
    const form = document.querySelector("#chat-form");
    const promptInput = document.querySelector("#prompt");
    const sendButton = document.querySelector("#send");
    const rebuildButton = document.querySelector("#rebuild");
    const componentLabels = {
      documents: "文档切片",
      embedding_model: "Embedding 模型",
      vector_database: "向量数据库",
      retriever: "检索器",
      context_builder: "上下文构建",
      llm: "LLM"
    };

    function appendMessage(role, text, sources = []) {
      const wrapper = document.createElement("div");
      wrapper.className = `message ${role}`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      wrapper.appendChild(bubble);
      messages.appendChild(wrapper);

      if (sources.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "参考来源";
        details.appendChild(summary);
        sources.forEach((source, index) => {
          const item = document.createElement("div");
          item.className = "source";
          const score = typeof source.score === "number" ? ` · 相关度 ${source.score}` : "";
          item.textContent = `${index + 1}. ${source.source || "unknown"}${score}\n${source.text || ""}`;
          details.appendChild(item);
        });
        messages.appendChild(details);
      }

      messages.scrollTop = messages.scrollHeight;
    }

    function renderComponents(components) {
      const container = document.querySelector("#components");
      container.textContent = "";
      Object.entries(componentLabels).forEach(([key, label]) => {
        const row = document.createElement("div");
        row.className = "component";
        const name = document.createElement("span");
        name.textContent = label;
        const badge = document.createElement("span");
        const enabled = Boolean(components && components[key]);
        badge.className = `badge ${enabled ? "ok" : "bad"}`;
        badge.textContent = enabled ? "已就绪" : "未就绪";
        row.appendChild(name);
        row.appendChild(badge);
        container.appendChild(row);
      });
    }

    async function refreshStatus() {
      const response = await fetch("/api/status");
      const status = await response.json();
      document.querySelector("#mode").textContent = status.mode;
      document.querySelector("#chunks").textContent = status.chunk_count;
      document.querySelector("#vectors").textContent = status.vector_count;
      renderComponents(status.components || {});
      const warning = document.querySelector("#warning");
      if (status.warning) {
        warning.style.display = "block";
        warning.textContent = status.warning;
      } else {
        warning.style.display = "none";
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const message = promptInput.value.trim();
      if (!message) return;

      appendMessage("user", message);
      promptInput.value = "";
      sendButton.disabled = true;
      sendButton.textContent = "生成中";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({message})
        });
        const data = await response.json();
        appendMessage("assistant", data.answer || "没有生成回答。", data.sources || []);
      } catch (error) {
        appendMessage("assistant", `请求失败：${error}`);
      } finally {
        sendButton.disabled = false;
        sendButton.textContent = "发送";
      }
    });

    rebuildButton.addEventListener("click", async () => {
      rebuildButton.disabled = true;
      rebuildButton.textContent = "重建中";
      try {
        const response = await fetch("/api/rebuild", {method: "POST"});
        const data = await response.json();
        if (!response.ok) {
          appendMessage("assistant", data.error || "重建索引失败。");
        }
      } finally {
        await refreshStatus();
        rebuildButton.disabled = false;
        rebuildButton.textContent = "重建知识库索引";
      }
    });

    appendMessage("assistant", "你好，我是一个基于知识库的 RAG ChatBot。你可以问我关于项目、RAG 流程或 data/documents 中资料的问题。");
    refreshStatus();
  </script>
</body>
</html>
"""


settings = Settings.from_env()
configure_logging(settings.log_file)
retriever = RAGRetriever(settings)
app = FastAPI(title="RAG ChatBot")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML_PAGE


@app.get("/api/status")
def api_status() -> dict:
    return status_payload()


@app.post("/api/chat")
async def api_chat(request: Request):
    try:
        payload = await request.json()
    except ValueError:
        payload = {}
    payload = payload if isinstance(payload, dict) else {}
    message = str(payload.get("message", "")).strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)

    result = retriever.answer(message)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "mode": result.mode,
    }


@app.post("/api/rebuild")
def api_rebuild():
    try:
        retriever.rebuild_index()
    except Exception as exc:
        return JSONResponse(
            {"error": f"重建索引失败：{exc}", **status_payload()},
            status_code=500,
        )
    return status_payload()


def status_payload() -> dict:
    warning = retriever.startup_warning
    try:
        vector_count = retriever.vector_count
    except Exception as exc:
        vector_count = 0
        status_warning = f"向量库状态读取失败：{exc}"
        warning = f"{warning}\n{status_warning}" if warning else status_warning

    return {
        "mode": retriever.mode_label,
        "chunk_count": retriever.chunk_count,
        "vector_count": vector_count,
        "components": retriever.components,
        "warning": warning,
    }


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"RAG ChatBot running at http://{display_host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
