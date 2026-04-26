# RAG ChatBot

一个可上传 GitHub 的中文 RAG ChatBot 小项目。项目支持接入外部大模型 API，不限定为 ChatGPT；用户只需要在 `.env` 中填写 3 个字段：`API_KEY`、`API_URL`、`MODEL_NAME`。

当前实现采用“本地检索 + 外部模型生成”的 RAG 流程：系统先从 `data/documents/` 检索相关知识库片段，再把上下文和问题发送给你配置的外部 API 生成回答。没有配置 API 时，会自动进入本地兜底模式，仍然可以演示检索和来源展示。

## 功能

- 可交互网页聊天界面
- 支持 `.txt` 和 `.md` 知识库文档
- 外部 API RAG 模式：本地检索上下文 + 外部模型生成回答
- 本地兜底模式：无 API 配置也能演示检索和来源展示
- 参考来源展示
- 一键重建知识库索引
- 完整依赖说明、`.env.example`、`.gitignore` 和 Dockerfile

## API 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后只填写 3 项：

```env
API_KEY=你的外部 API Key
API_URL=https://api.example.com/v1/chat/completions
MODEL_NAME=你的模型名称
```

`API_URL` 支持 OpenAI-compatible 的聊天接口：

- 完整接口地址：`https://api.example.com/v1/chat/completions`
- base URL：`https://api.example.com` 或 `https://api.example.com/v1`

如果填写 base URL，程序会自动补成 `/v1/chat/completions`。常见兼容服务包括 OpenRouter、DeepSeek、部分国内大模型平台的 OpenAI-compatible endpoint，以及自建兼容接口。

## 依赖说明

[requirements.txt](requirements.txt) 已按用途分组并写明每个包的作用，可以直接一键安装：

本项目要求 Python 版本：

```text
Python 3.12.x
```

注意：`requirements.txt` 只能安装 Python 包，不能安装 Python 解释器本身。使用 Conda 时，先创建带 Python 3.12 的环境：

```powershell
conda create -n chatbot python=3.12 -y
conda activate chatbot
```

然后安装包依赖：

```powershell
python -m pip install -r requirements.txt
```

也可以使用 [environment.yml](environment.yml) 一键创建环境并安装依赖：

```powershell
conda env create -f environment.yml
conda activate chatbot
```

依赖组包括：

- Web UI：`streamlit`
- 外部大模型 API / OpenAI-compatible 能力：`openai`、`tiktoken`
- Agent/RAG 编排：`langgraph`、`langchain`、`langchain-openai`
- 向量数据库扩展：`chromadb`
- 数据处理与可视化：`pandas`、`matplotlib`、`plotly`
- 交互终端兼容：`wcwidth`
- 环境变量管理：`python-dotenv`

如果国内网络下载较慢，可以使用镜像：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装完成后建议执行一次依赖检查：

```powershell
python -m pip check
python -c "import streamlit, openai, tiktoken, langgraph, langchain, langchain_openai, chromadb, pandas, matplotlib, plotly, dotenv, wcwidth; print('dependencies ok')"
```

## 项目结构

```text
.
├── app.py
├── rag/
│   ├── config.py
│   ├── documents.py
│   ├── embeddings.py
│   ├── external_api.py
│   ├── local_search.py
│   ├── logging_config.py
│   ├── retriever.py
│   └── vector_store.py
├── data/
│   └── documents/
├── logs/
├── .env.example
├── .gitignore
├── Dockerfile
├── requirements.txt
└── README.md
```

## 本地复现

下面是一套从零复现项目的 PowerShell 流程。

1. 进入项目目录：

```powershell
cd D:\DeskTop\chatbot
```

2. 准备 Conda 环境。

推荐直接使用 `environment.yml` 创建完整环境：

```powershell
conda env create -f environment.yml
conda activate chatbot
```

如果你想手动创建环境：

```powershell
conda create -n chatbot python=3.12 -y
conda activate chatbot
```

如果 `chatbot` 环境已经存在，但检查发现没有自己的 Python：

```powershell
conda install -n chatbot python=3.12 -y
```

检查命令：

```powershell
Test-Path D:\conda\envs\chatbot\python.exe
```

返回 `True` 才说明该 Conda 环境有自己的 Python。

3. 安装依赖。

如果你已经使用 `environment.yml` 创建环境，这一步通常已经完成。手动安装时执行：

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果下载慢，可以使用国内镜像：

```powershell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. 验证依赖完整：

```powershell
python -m pip check
python -c "import streamlit, openai, tiktoken, langgraph, langchain, langchain_openai, chromadb, pandas, matplotlib, plotly, dotenv, wcwidth; print('dependencies ok')"
```

5. 配置外部 API。

复制配置模板：

```powershell
Copy-Item .env.example .env
```

然后只填写 `.env` 里的 3 项：

```env
API_KEY=你的外部 API Key
API_URL=https://api.example.com/v1/chat/completions
MODEL_NAME=你的模型名称
```

6. 启动项目：

```powershell
python app.py
```

启动后打开浏览器：

```text
http://127.0.0.1:8000
```

如果没有配置 `.env`，项目会自动进入本地兜底模式，仍然可以测试知识库检索和参考来源展示。

## 扩展知识库

把 `.txt` 或 `.md` 文件放到：

```text
data/documents/
```

然后在网页侧边栏点击“重建知识库索引”。

## Docker 运行

```powershell
docker build -t rag-chatbot .
docker run --rm -p 8000:8000 --env-file .env rag-chatbot
```

没有 `.env` 时也可以运行：

```powershell
docker run --rm -p 8000:8000 rag-chatbot
```

## 上传 GitHub

```powershell
git init
git add .
git commit -m "Initial RAG chatbot project"
```

`.gitignore` 已经忽略 `.env`、`.venv/`、`vector_store/`、`chroma_db/`、日志文件和缓存目录。
