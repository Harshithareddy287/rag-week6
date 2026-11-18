# rag-week6

Simple RAG lab over company policy docs.

## Setup

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env   # then fill in your Azure OpenAI values
```

## Run tests

```bash
pytest -v
```

The LLM test is skipped unless `AZURE_OPENAI_API_KEY` is set.
