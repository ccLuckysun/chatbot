# RAG ChatBot 示例知识库

这个项目是一个简单但完整的中文 RAG 问答助手。当前主链路已经改造成 LangChain 标准 RAG，同时保留原来的 Web 页面、HTTP API、`.env` 配置方式和本地关键词兜底模式。

完整 RAG 模式使用的核心组件包括：LangChain `Document`、`RecursiveCharacterTextSplitter`、`OpenAIEmbeddings`、`langchain_chroma.Chroma`、`ChatPromptTemplate`、`ChatOpenAI` 和 `StrOutputParser`。

## RAG 工作流程

RAG 是 Retrieval-Augmented Generation 的缩写，意思是检索增强生成。当前项目的完整流程是：先读取 `data/documents` 下的知识库文件，生成 LangChain `Document`；再用 `RecursiveCharacterTextSplitter` 切分成 chunk；随后由 `OpenAIEmbeddings` 生成向量并写入 Chroma；用户提问时，系统用 Chroma 做语义相似度检索，把命中的片段整理成上下文，再通过 `ChatPromptTemplate | ChatOpenAI | StrOutputParser` 生成中文答案。

这种方式可以让回答更贴近项目自己的资料，并且能在界面中展示参考来源。

## 项目组件

- `rag/documents.py`：读取 `.md` 和 `.txt` 文件，返回 LangChain `Document`，并用 `RecursiveCharacterTextSplitter` 按 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 切分。
- `rag/embeddings.py`：创建 `OpenAIEmbeddings`，继续兼容 OpenAI-compatible Embedding API。
- `rag/vector_store.py`：封装 LangChain `Chroma`，负责向量持久化、写入、清空和相似度检索。
- `rag/context_builder.py`：把检索到的 `Document + score` 整理成 LLM 可读的上下文。
- `rag/llm.py`：创建 `ChatOpenAI`，继续读取 `LLM_BASE_URL` 和 `LLM_MODEL`。
- `rag/retriever.py`：项目的编排中心，负责初始化 LangChain RAG 链路、重建索引、回答问题，并在失败时切换到本地兜底。
- `rag/local_search.py`：无 API 时的本地关键词检索，只使用 `Document.page_content` 做关键词打分，不调用 Embedding、Chroma 或 LLM。

## 运行模式

如果 `.env` 中完整配置了 `LLM_*` 和 `EMBEDDING_*`，应用会进入完整 RAG 模式。启动或点击“重建知识库索引”时，系统会读取文档、生成向量并写入 ChromaDB。

如果没有完整配置这些变量，应用会进入本地关键词检索兜底模式。这个模式不调用外部 API，只适合演示文档读取、简单检索和来源展示，不代表完整 RAG 能力。

完整 RAG 与本地兜底的区别是：完整 RAG 会调用 Embedding API、使用 Chroma 向量检索，并调用 LLM 生成回答；本地兜底只做关键词匹配，然后把命中的片段直接拼成参考回答。

## 如何扩展知识库

把新的 `.txt` 或 `.md` 文件放入 `data/documents` 目录，然后在网页侧边栏点击“重建知识库索引”。完整 RAG 模式下会重新生成 ChromaDB 向量索引；本地兜底模式下会重新读取文档并更新关键词检索内容。

## GitHub 上传建议

不要提交真实 `.env`、虚拟环境、日志文件和本地向量索引目录。项目已经通过 `.gitignore` 忽略这些运行时文件，可以提交 `.env.example` 作为配置模板。
