# RAG ChatBot 示例知识库

这个项目是一个简单、完整、可以上传到 GitHub 的 RAG ChatBot。它使用 Python 标准库构建网页聊天界面，先用本地检索找到相关知识库片段，再把问题和上下文发送给外部大模型 API 生成回答。

## RAG 工作流程

RAG 是 Retrieval-Augmented Generation 的缩写，意思是检索增强生成。典型流程是：用户提问，系统先把问题转换成向量，再从知识库中检索最相关的文档片段，然后把问题和文档上下文一起交给大语言模型生成答案。这样可以让回答更贴近项目自己的资料，并显示参考来源。

## 运行模式

如果 `.env` 文件里配置了 `API_KEY`、`API_URL`、`MODEL_NAME`，应用会进入外部 API RAG 模式。这个模式会用本地检索找到知识库上下文，再使用配置的外部模型生成中文回答。

如果没有完整配置这 3 项，应用会进入本地兜底模式。本地兜底模式不会调用外部 API，而是用关键词检索知识库，并基于检索片段返回模板化回答。这个模式适合 GitHub 克隆后快速演示。

## 如何扩展知识库

把新的 `.txt` 或 `.md` 文件放入 `data/documents` 目录，然后在网页侧边栏点击“重建知识库索引”。如果处于外部 API RAG 模式，系统会重新读取文档并在后续回答中使用新的检索内容；如果处于本地兜底模式，系统同样会重新读取文档内容。

## GitHub 上传建议

不要把真实 `.env` 文件、虚拟环境、日志文件、向量索引目录上传到 GitHub。项目已经提供 `.gitignore`，会忽略这些本地文件。可以把 `.env.example` 提交到仓库，方便其他人知道需要配置哪些环境变量。

## 常用启动命令

Windows PowerShell 可以使用以下命令创建和启用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
