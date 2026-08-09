# NewsRAG

A Streamlit app with three independent, chat-with-your-sources RAG workspaces — pick one from the landing screen and start asking questions.

| Workspace | What it does |
|---|---|
| 📰 **Newsroom** | Load a full day's coverage from 14 major newspapers (Indian + international). Pick a publication and a date, then chat with what was actually printed. |
| 🧭 **Topic Radar** | Search any keyword or story across every publisher NewsAPI indexes, over a date range. Good for tracking a developing story across outlets. |
| 📄 **Document Vault** | Upload your own PDFs and chat with them directly, page by page. Fully isolated from any newspaper content. |

Each workspace keeps its **own chat history, its own retrieval scope, and its own loaded sources** — nothing bleeds between them. Under the hood it's LangChain + FAISS for retrieval, Groq (Llama 3.1 8B Instant) for generation, and NewsAPI for article data.

## Features

- Card-based landing screen to pick a workspace (with a per-workspace scroll-position memory, so switching back doesn't dump you at the top of the page)
- Conversational RAG with query rewriting (follow-up questions are contextualized against chat history before retrieval)
- "Summarize this" style queries pull the *full* indexed context for that scope instead of relying on similarity search
- Multiple sources can be loaded per workspace, with an option to scope chat to "all loaded" or one specific source
- Session-only, in-memory FAISS index — no external vector DB required
- Sidebar with live counts of articles/pages loaded and chunks indexed

## Requirements

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys) (free tier available)
- A [NewsAPI.org API key](https://newsapi.org/register) (free tier available; note free tier only covers roughly the last month of articles)

## Setup

1. **Clone / download the project**, then install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your API keys.** Create `.streamlit/secrets.toml` in the project root:

   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   NEWSAPI_KEY = "your-newsapi-key"
   ```

   (If deploying to Streamlit Community Cloud, add these under your app's **Settings → Secrets** instead of committing this file.)

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   It opens at `http://localhost:8501`.


## Project structure

```
.
├── app.py              # main Streamlit app
├── requirements.txt
├── README.md
└── .streamlit/
    └── secrets.toml     # you create this — not committed
```

## Tech stack

Streamlit · LangChain (core, text-splitters, community, groq, huggingface) · FAISS · sentence-transformers (`all-MiniLM-L6-v2`) · Groq (Llama 3.1 8B Instant) · pypdf · NewsAPI

## Known limitations

- The FAISS index and chat histories live in Streamlit session state — refreshing the browser tab or restarting the server clears everything loaded so far.
- NewsAPI's free tier has request-volume and historical-range limits; heavy use may hit rate limits.
- Summary-style answers are capped at ~14,000 characters of context (`MAX_SUMMARY_CHARS`) to stay within the LLM's context window; very large loaded sets get truncated.
