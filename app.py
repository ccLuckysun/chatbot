from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

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
      grid-template-columns: 280px minmax(0, 1fr);
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
      background: rgba(255, 255, 255, 0.86);
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
      </div>
      <div class="warning" id="warning"></div>
      <button class="secondary" id="rebuild" type="button">重建知识库索引</button>
      <p class="hint">把 .txt 或 .md 文件放入 data/documents 后，点击重建知识库索引即可检索新内容。</p>
    </aside>
    <main>
      <header>
        <h1>RAG ChatBot</h1>
        <p class="subtitle">基于本地知识库检索上下文，再生成带来源的回答。</p>
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

    async function refreshStatus() {
      const response = await fetch("/api/status");
      const status = await response.json();
      document.querySelector("#mode").textContent = status.mode;
      document.querySelector("#chunks").textContent = status.chunk_count;
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
      await fetch("/api/rebuild", {method: "POST"});
      await refreshStatus();
      rebuildButton.disabled = false;
      rebuildButton.textContent = "重建知识库索引";
    });

    appendMessage("assistant", "你好，我是一个基于知识库的 RAG ChatBot。你可以问我关于这个项目、RAG 流程或部署方式的问题。");
    refreshStatus();
  </script>
</body>
</html>
"""


settings = Settings.from_env()
configure_logging(settings.log_file)
retriever = RAGRetriever(settings)


class ChatbotHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML_PAGE)
            return
        if path == "/api/status":
            self._send_json(status_payload())
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/chat":
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                self._send_json({"error": "message is required"}, HTTPStatus.BAD_REQUEST)
                return

            result = retriever.answer(message)
            self._send_json(
                {
                    "answer": result.answer,
                    "sources": result.sources,
                    "mode": result.mode,
                }
            )
            return

        if path == "/api/rebuild":
            retriever.rebuild_index()
            self._send_json(status_payload())
            return

        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length).decode("utf-8")
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _send_html(self, content: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = content.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def status_payload() -> dict:
    return {
        "mode": retriever.mode_label,
        "chunk_count": retriever.chunk_count,
        "warning": retriever.startup_warning,
    }


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), ChatbotHandler)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"RAG ChatBot running at http://{display_host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
