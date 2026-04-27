# RAG ChatBot

一个最小但完整的中文 RAG 问答助手。项目主链路已经迁移为 LangChain 标准 RAG：

```text
文档加载 -> LangChain Document -> RecursiveCharacterTextSplitter -> OpenAIEmbeddings -> Chroma -> ChatPromptTemplate -> ChatOpenAI -> 前端展示答案与来源
```

未配置外部 API 时，应用会自动降级到本地关键词检索兜底模式，方便演示文档读取和来源展示；完整 RAG 能力需要同时配置 LLM 和 Embedding 服务。

## 功能

- FastAPI 提供网页聊天界面和 JSON API，并自动生成接口文档。
- 支持读取 `data/documents/` 下的 `.md` 与 `.txt` 文档。
- 使用 LangChain `RecursiveCharacterTextSplitter` 进行文档切分。
- 使用 LangChain `OpenAIEmbeddings` 调用 OpenAI-compatible Embedding API。
- 使用 LangChain `Chroma` 封装 ChromaDB 本地持久化向量库，默认目录是 `vector_store/`。
- 使用 `ChatPromptTemplate | ChatOpenAI | StrOutputParser` 生成回答。
- 前端展示运行模式、文档切片数、向量索引数、组件就绪状态和参考来源。
- 未配置 API 时自动进入本地兜底模式。

## 项目结构

```text
.
├── app.py
├── rag/
│   ├── config.py
│   ├── context_builder.py
│   ├── documents.py
│   ├── embeddings.py
│   ├── llm.py
│   ├── local_search.py
│   ├── logging_config.py
│   ├── retriever.py
│   └── vector_store.py
├── data/
│   └── documents/
├── tests/
├── .env.example
├── Dockerfile
├── environment.yml
├── RAG_ChatBot_LangChain_项目解析.md
├── README.md
└── requirements.txt
```

## Conda 环境

推荐使用 Conda 管理虚拟环境。项目要求 Python 3.12。

首次创建环境：

```powershell
cd D:\DeskTop\chatbot
conda env create -f environment.yml
conda activate chatbot
```

如果 `chatbot` 环境已经存在，可以更新环境：

```powershell
conda activate chatbot
conda env update -f environment.yml --prune
```

也可以手动创建 Conda 环境并安装依赖：

```powershell
conda create -n chatbot python=3.12 -y
conda activate chatbot
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

检查依赖是否正常：

```powershell
python --version
python -c "import langchain, langchain_openai, langchain_chroma, chromadb, dotenv; print('dependencies ok')"
```

## 配置

复制配置模板：

```powershell
Copy-Item .env.example .env
```

填写 `.env`：

```env
LLM_API_KEY="your_llm_api_key"
LLM_BASE_URL="https://api.example.com/v1"
LLM_MODEL="your-chat-model"

EMBEDDING_API_KEY="your_embedding_api_key"
EMBEDDING_BASE_URL="https://api.example.com/v1"
EMBEDDING_MODEL="your-embedding-model"
```

上面的 `your_*`、`your-*` 和 `api.example.com` 都只是模板占位符。必须替换成真实服务商提供的 Key、Base URL 和模型名，否则页面会继续显示“本地兜底模式”。

`LLM_BASE_URL` 和 `EMBEDDING_BASE_URL` 支持 OpenAI-compatible base URL。可以填写 `https://api.example.com` 或 `https://api.example.com/v1`，程序会自动归一化到 `/v1`。

LLM 模型和 Embedding 模型不能混用。以阿里云百炼 DashScope 为例，`LLM_MODEL` 可以使用聊天模型，`EMBEDDING_MODEL` 应使用向量模型，例如 `text-embedding-v4`。

DashScope 示例：

```env
LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL="qwen3.6-plus"

EMBEDDING_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL="text-embedding-v4"
```

不要把 `EMBEDDING_MODEL` 写成 `qwen3.6-plus` 这类聊天模型，否则启动或重建索引时会调用 `/embeddings` 失败，并出现类似错误：

```text
Unsupported model `qwen3.6-plus` for OpenAI compatibility mode.
```

ChromaDB 向量库重建注意事项：同一时间只运行一个 FastAPI 服务进程（通常是 `python app.py` 启动的进程）。如果开了多个服务进程，它们会共享同一个 `vector_store/` 目录；一个进程重建索引时，另一个进程可能仍持有旧的 collection 引用，导致查询时报错。

典型错误：

```text
Collection [xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx] does not exist.
```

遇到这个错误时，先停止所有正在运行的服务进程，再重新执行 `python app.py` 启动一个新进程。当前代码已经改为清空 collection 内文档，而不是删除整个 collection，以减少这个问题再次发生。

常用运行参数也可在 `.env` 中调整：

```env
DOCUMENTS_DIR="data/documents"
VECTOR_STORE_DIR="vector_store"
VECTOR_COLLECTION_NAME="docs"
TOP_K=3
CHUNK_SIZE=900
CHUNK_OVERLAP=120
MAX_CONTEXT_CHARS=6000
```

## 启动

确保已经激活 Conda 环境：

```powershell
conda activate chatbot
python app.py
```

打开：

```text
http://127.0.0.1:8000
```

FastAPI 自动接口文档：

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
```

停止刚刚启动的服务：

- 如果服务运行在当前 PowerShell 窗口，按 `Ctrl+C`。
- 如果服务已经在后台运行，先查找 `app.py` 进程，再按 PID 终止：

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.Name -match '^(python|pythonw)\.exe$' -and $_.CommandLine -match 'app\.py' } |
  Select-Object ProcessId,CommandLine

Stop-Process -Id <ProcessId> -Force
```

完整配置后，应用启动时会读取 `data/documents/`，通过 LangChain splitter 切分文档，生成 embeddings 并写入 ChromaDB。添加或修改知识库文档后，在页面侧边栏点击“重建知识库索引”。

## API

`GET /api/status`

返回运行模式、文档切片数、向量数量、组件状态和启动警告。

`POST /api/chat`

请求：

```json
{"message": "这个项目的 RAG 流程是什么？"}
```

响应：

```json
{
  "answer": "模型生成的回答",
  "sources": [
    {"source": "rag_chatbot_guide.md", "score": 0.91, "text": "相关片段"}
  ],
  "mode": "rag"
}
```

`POST /api/rebuild`

重新读取知识库文档，并在完整 RAG 模式下重建 ChromaDB 向量索引。

## 扩展知识库

把 `.md` 或 `.txt` 文件放入：

```text
data/documents/
```

然后重建索引。不要把生成的 `vector_store/` 提交到 GitHub。

## 测试

```powershell
conda activate chatbot
python -m compileall app.py rag tests
python -m unittest discover -s tests
```

## Docker

镜像仍通过 `python app.py` 启动，容器内由 Uvicorn 运行 FastAPI 应用。

```powershell
docker build -t rag-chatbot .
docker run --rm -p 8000:8000 --env-file .env rag-chatbot
```

如果没有 `.env`，容器仍可启动，但只会使用本地关键词检索兜底模式。

## 常见问题

- `conda activate chatbot` 失败：先确认已经执行过 `conda env create -f environment.yml`。
- 依赖导入失败：在已激活环境中执行 `conda env update -f environment.yml --prune`。
- 页面显示本地兜底模式：说明 `.env` 没有完整配置 `LLM_*` 和 `EMBEDDING_*`，或者仍在使用 `your_*` / `api.example.com` 这类模板占位值。修改 `.env` 后需要停止当前服务并重新执行 `python app.py`。
- 修改 `.env` 后页面仍然不变：浏览器连接的可能是旧 FastAPI 服务进程。先停止正在运行的服务进程，再重新执行 `python app.py`；必要时检查 8000 端口是否仍被旧进程占用。
- 完整 RAG 初始化失败并提示 `Unsupported model ... /embeddings`：通常是把聊天模型填到了 `EMBEDDING_MODEL`。DashScope 请使用 `text-embedding-v4` 这类向量模型。
- 完整 RAG 初始化失败并提示 `input.contents` 参数不合法：确认依赖已经更新到当前代码版本；项目已在 `OpenAIEmbeddings` 中关闭 token-array 输入，兼容只接受字符串的 OpenAI-compatible embedding 服务。
- 完整 RAG 调用失败并提示 `Collection [...] does not exist`：通常是多个 `app.py` 进程同时使用同一个 ChromaDB 持久化目录，或旧进程持有了已失效的 collection 引用。停止所有旧进程，只保留一个新启动的服务。
- `vector_count` 为 0：完整 RAG 模式下点击“重建知识库索引”，或确认 `data/documents/` 下有 `.md` / `.txt` 文件。
