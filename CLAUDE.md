# Project rules
- NEVER run git commit, git push, git merge, or switch branches. I do all git myself.
- Never write real API keys. Never create or edit .env.
- Keep code simple and minimal. Build only what the current phase asks for.
- The .venv is already set up (Python 3.14). Don't create environments or install packages.

# Project
Simple RAG lab over company policy docs.
Stack: Python 3.14, sentence-transformers (all-MiniLM-L6-v2), chromadb,
cross-encoder reranker, Azure OpenAI (deployment "gpt-6-sol"), pytest, GitHub Actions.
