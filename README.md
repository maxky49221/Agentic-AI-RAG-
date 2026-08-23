# Agentic AI — Multi-Agent RAG System
A multi-agent AI system built with LangChain and LangGraph that uses Retrieval-Augmented Generation (RAG) to answer questions from a local knowledge base.

## Requirements
- Python 3.11+
- Azure OpenAI access (gpt-5-mini endpoint)

## Installation
```bash
uv init
uv add langchain langchain-openai langchain-huggingface langchain-community
uv add langgraph faiss-cpu sentence-transformers python-dotenv
uv add langchain-text-splitters
```

## Configuration
Create a `.env` file in the project root:
```env
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://oaibblinnocandiddate01.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini
AZURE_OPENAI_API_VERSION=2024-02-01
```

## Usage
```bash
uv run main.py
```
Then type your query when prompted:
```
You: ธนาคารกรุงเทพมีนโยบายด้านความยั่งยืนอย่างไร
```

## Sample Queries
- `ธนาคารกรุงเทพก่อตั้งเมื่อปีไหน`
- `ธนาคารกรุงเทพมีสาขาต่างประเทศที่ไหนบ้าง`
- `นโยบาย Net Zero ของธนาคารกรุงเทพคืออะไร`
- `ธนาคารกรุงเทพให้บริการอะไรบ้าง`

## Agent Details
### Data Retriever Agent
- Performs semantic search on `knowledge_base.txt` using HuggingFace embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- Returns top 5 relevant text chunks without generating any response
- Does not call the LLM — raw retrieval only

### Report Generator Agent
- Receives raw snippets from the Data Retriever
- Uses Azure OpenAI (gpt-5-mini) to synthesize a cohesive, non-redundant, well-formatted answer
- Aware of the original user query for context

## Framework
- **LangGraph** — Sequential multi-agent orchestration
- **LangChain** — LLM integration, prompt management
- **FAISS** — Local vector store for semantic search
- **HuggingFace Sentence Transformers** — Thai-friendly multilingual embeddings
