import streamlit as st
import streamlit.components.v1 as components
import requests
import datetime
import hashlib
import io

from pypdf import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ---------- CONFIG ----------
st.set_page_config(page_title="NewsRAG", page_icon="📰", layout="wide")
MAX_DAYS_BACK = 30  # user can only pick a date within the last 10 days
SUMMARY_KEYWORDS = ("summar", "overview", "tl;dr", "tldr", "gist", "main points")
RESTART_KEYWORDS = {"restart", "reset", "new chat", "clear chat", "clear"}
MAX_SUMMARY_CHARS = 14000  # cap on raw context sent for a full-document summary


NEWSPAPER_NAME_MATCHES = {
    "The Hindu": ["hindu"],
    "Times of India": ["times of india"],
    "Hindustan Times": ["hindustan times"],
    "The Indian Express": ["indian express"],
    "NDTV": ["ndtv"],
    "India Today": ["india today"],
    "Deccan Herald": ["deccan herald"],
    "The Telegraph India": ["telegraph india"],
    "Livemint": ["livemint", "mint"],
    "Business Standard": ["business standard", "business-standard"],
    "The Economic Times": ["economic times"],
    "Firstpost": ["firstpost"],
    "The New York Times": ["new york times"],   # 🇺🇸 US
    "The Guardian": ["the guardian", "guardian"],  # 🇬🇧 UK
}

NEWSPAPER_ICONS = {
    "The Hindu": "🟠", "Times of India": "🔴", "Hindustan Times": "🟢",
    "The Indian Express": "🔵", "NDTV": "⚫", "India Today": "🟣",
    "Deccan Herald": "🟡", "The Telegraph India": "🟤", "Livemint": "🟩",
    "Business Standard": "🟦", "The Economic Times": "🟧", "Firstpost": "⬛",
    "The New York Times": "🇺🇸", "The Guardian": "🇬🇧",
}

# ---------- MODE SELECTOR CONFIG ----------
# Drives the landing "choose a mode" cards, one per independent RAG workspace.
MODES = {
    "news": {
        "emoji": "📰",
        "title": "Newsroom",
        "desc": "Pull a full day's coverage from 14 major newspapers, Indian and international. Pick a publication and a date, then chat with what was actually printed.",
        "badge": "14 publishers · Date-scoped",
        "color": "#F7931E",
        "bg": "linear-gradient(160deg, rgba(247,147,30,0.14), rgba(193,18,31,0.05))",
        "border": "rgba(247,147,30,0.45)",
        "glow": "rgba(247,147,30,0.55)",
        "button_label": "📰 Open Newsroom",
    },
    "topic": {
        "emoji": "🧭",
        "title": "Topic Radar",
        "desc": "Track any keyword or story across every publisher NewsAPI indexes, not just one paper. Great for following a developing situation over a date range.",
        "badge": "All publishers · Keyword search",
        "color": "#B084F5",
        "bg": "linear-gradient(160deg, rgba(176,132,245,0.14), rgba(109,40,217,0.05))",
        "border": "rgba(176,132,245,0.45)",
        "glow": "rgba(176,132,245,0.55)",
        "button_label": "🧭 Open Topic Radar",
    },
    "pdf": {
        "emoji": "📄",
        "title": "Document Vault",
        "desc": "Upload your own PDFs, reports, papers, clippings, and chat with them directly, page by page. Fully isolated from any newspaper content.",
        "badge": "Multi-file · Page-level chunks",
        "color": "#3FD9C7",
        "bg": "linear-gradient(160deg, rgba(63,217,199,0.14), rgba(6,148,162,0.05))",
        "border": "rgba(63,217,199,0.45)",
        "glow": "rgba(63,217,199,0.55)",
        "button_label": "📄 Open Document Vault",
    },
}

# ---------- STYLING (dark-theme safe: no forced light backgrounds) ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header-wrap {
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.2rem;
    }
    .main-header {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 4rem;
        font-weight: 700;
        letter-spacing: -1.5px;
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 45%, #C1121F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .subtitle {
        text-align: center;
        opacity: 0.65;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.6rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        margin: 0.15rem;
        border-radius: 999px;
        background: rgba(255, 107, 53, 0.12);
        border: 1px solid rgba(255, 107, 53, 0.35);
        font-size: 0.85rem;
    }
    .scope-pill {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        background: rgba(120, 130, 255, 0.15);
        border: 1px solid rgba(120, 130, 255, 0.4);
        font-size: 0.8rem;
        margin-bottom: 0.6rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-left: 3px solid #F7931E;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 700;
        font-size: 1.05rem;
    }

    /* ---- Mode-selector landing cards ---- */
    .mode-select-heading {
        text-align: center;
        font-size: 1.05rem;
        opacity: 0.7;
        margin-bottom: 1.8rem;
        font-family: 'Space Grotesk', sans-serif;
    }
    .mode-card {
        border: 1.5px solid var(--mc-border);
        border-radius: 20px;
        padding: 2.1rem 1.5rem 1.4rem;
        text-align: center;
        min-height: 300px;
        display: flex;
        flex-direction: column;
        align-items: center;
        background: var(--mc-bg);
        box-shadow: 0 0 0 1px rgba(255,255,255,0.02);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
        margin-bottom: 0.7rem;
    }
    .mode-card.active {
        box-shadow: 0 0 26px -6px var(--mc-glow);
        transform: translateY(-2px);
    }
    .mode-icon {
        font-size: 2.6rem;
        margin-bottom: 0.7rem;
        filter: drop-shadow(0 0 10px var(--mc-glow));
    }
    .mode-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--mc-color);
        margin-bottom: 0.6rem;
        letter-spacing: -0.3px;
    }
    .mode-desc {
        font-size: 0.89rem;
        opacity: 0.75;
        line-height: 1.5;
        margin-bottom: 1.1rem;
        min-height: 78px;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.32rem 0.9rem;
        border-radius: 999px;
        border: 1px solid var(--mc-border);
        color: var(--mc-color);
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
        white-space: nowrap;
    }
    .mode-hint {
        font-size: 0.76rem;
        opacity: 0.45;
        margin-top: auto;
    }
    .active-mode-banner {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- CACHED RESOURCES ----------
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=st.secrets["GROQ_API_KEY"],
        model_name="openai/gpt-oss-20b"
        temperature=0.2,
    )

embedder = load_embedder()
llm = load_llm()

# ---------- SESSION STATE ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "seen_ids" not in st.session_state:
    st.session_state.seen_ids = set()
# Two FULLY INDEPENDENT chat threads — the newspaper tab and the PDF tab
# never see each other's conversation or retrieval context.
if "chat_history_news" not in st.session_state:
    st.session_state.chat_history_news = []
if "chat_history_pdf" not in st.session_state:
    st.session_state.chat_history_pdf = []
if "chat_history_topic" not in st.session_state:
    st.session_state.chat_history_topic = []
if "loaded_groups" not in st.session_state:
    # group_key -> {"label": str, "kind": "news" | "pdf" | "topic", "count": int}
    st.session_state.loaded_groups = {}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "active_mode" not in st.session_state:
    st.session_state.active_mode = None  # "news" | "topic" | "pdf" | None

# ---------- HELPERS: NEWS ----------
def article_id(link):
    return hashlib.md5(link.encode()).hexdigest()

def fetch_articles(newspaper_name, target_date):
    """
    Fetch broadly (q=India, date-scoped, no source/domain filter — this is
    the combination confirmed to work), then filter to the chosen
    publication client-side using the article's own source.name field.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "India",
        "from": target_date.isoformat(),
        "to": target_date.isoformat(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": st.secrets["NEWSAPI_KEY"],
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        st.error(f"NewsAPI error: {data.get('code', '')} — {data.get('message', 'unknown error')}")
        return []

    total = data.get("totalResults", 0)
    all_articles = data.get("articles", [])

    match_terms = NEWSPAPER_NAME_MATCHES.get(newspaper_name, [])
    filtered = [
        a for a in all_articles
        if any(term in (a.get("source", {}).get("name", "") or "").lower() for term in match_terms)
    ]

    st.caption(
        f"🔍 Debug: {total} total results for that date, {len(all_articles)} fetched, "
        f"{len(filtered)} matched '{newspaper_name}' by publication name."
    )

    actual_sources = sorted(set(
        a.get("source", {}).get("name", "Unknown") for a in all_articles
    ))
    with st.expander(f"🔍 Debug: actual source names found ({len(actual_sources)} unique)"):
        st.write(actual_sources)

    return filtered

def articles_to_documents(raw_articles, source_name, target_date, group_key):
    """Convert raw NewsAPI articles into LangChain Documents, deduplicated."""
    docs = []
    for entry in raw_articles:
        link = entry.get("url", "")
        aid = article_id(link)
        if not link or aid in st.session_state.seen_ids:
            continue

        title = entry.get("title", "") or ""
        description = entry.get("description", "") or ""
        content = entry.get("content", "") or ""
        full_text = f"{title}. {description} {content}".strip()

        if not full_text:
            continue

        docs.append(Document(
            page_content=full_text,
            metadata={
                "title": title,
                "source": source_name,
                "date": target_date.isoformat(),
                "link": link,
                "kind": "news",
                "group_key": group_key,
            }
        ))
        st.session_state.seen_ids.add(aid)
    return docs

# ---------- HELPERS: TOPIC SEARCH ----------
def fetch_topic_articles(topic, date_from, date_to):
    """
    Search NewsAPI's /everything endpoint directly by keyword/topic, across
    ALL indexed publishers (no source-name filtering) — unlike the newspaper
    tab, which fetches broadly then filters down to one publication, here the
    breadth across publishers is the point.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "from": date_from.isoformat(),
        "to": date_to.isoformat(),
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "apiKey": st.secrets["NEWSAPI_KEY"],
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "ok":
        st.error(f"NewsAPI error: {data.get('code', '')} — {data.get('message', 'unknown error')}")
        return []

    articles = data.get("articles", [])
    st.caption(f"🔍 Found {data.get('totalResults', 0)} total results for \"{topic}\", showing top {len(articles)}.")
    return articles

def topic_articles_to_documents(raw_articles, topic, group_key):
    """Convert topic-search articles into Documents, keeping each article's
    OWN publisher name (unlike the newspaper tab, results here span many
    different outlets) so citations stay accurate."""
    docs = []
    for entry in raw_articles:
        link = entry.get("url", "")
        aid = article_id(link)
        if not link or aid in st.session_state.seen_ids:
            continue

        title = entry.get("title", "") or ""
        description = entry.get("description", "") or ""
        content = entry.get("content", "") or ""
        full_text = f"{title}. {description} {content}".strip()
        if not full_text:
            continue

        publisher = entry.get("source", {}).get("name", "Unknown source")
        published_date = (entry.get("publishedAt") or "")[:10] or datetime.date.today().isoformat()

        docs.append(Document(
            page_content=full_text,
            metadata={
                "title": title,
                "source": publisher,
                "date": published_date,
                "link": link,
                "kind": "topic",
                "group_key": group_key,
            }
        ))
        st.session_state.seen_ids.add(aid)
    return docs

# ---------- HELPERS: PDF ----------
def pdf_id(filename, filesize):
    return hashlib.md5(f"{filename}:{filesize}".encode()).hexdigest()

def pdf_to_documents(uploaded_file):
    """Extract text from an uploaded PDF (page by page) into LangChain Documents."""
    file_bytes = uploaded_file.getvalue()
    fid = pdf_id(uploaded_file.name, len(file_bytes))
    group_key = f"pdf::{uploaded_file.name}::{fid}"
    if fid in st.session_state.seen_ids:
        return [], group_key

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Couldn't read '{uploaded_file.name}': {e}")
        return [], group_key

    docs = []
    today = datetime.date.today().isoformat()
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "title": f"{uploaded_file.name} — page {page_num}",
                "source": f"📄 {uploaded_file.name}",
                "date": today,
                "link": "",
                "kind": "pdf",
                "group_key": group_key,
                "page": page_num,
            }
        ))

    st.session_state.seen_ids.add(fid)
    return docs, group_key

# ---------- SHARED: VECTORSTORE ----------
def build_or_update_vectorstore(documents, group_key, group_label, group_kind):
    """Chunk documents, add to FAISS, and register the source group."""
    if not documents:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(documents)
    st.session_state.total_chunks += len(chunks)

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = FAISS.from_documents(chunks, embedder)
    else:
        st.session_state.vectorstore.add_documents(chunks)

    existing = st.session_state.loaded_groups.get(group_key)
    st.session_state.loaded_groups[group_key] = {
        "label": group_label,
        "kind": group_kind,
        "count": (existing["count"] if existing else 0) + len(documents),
    }

def groups_of_kind(kind):
    return {k: v for k, v in st.session_state.loaded_groups.items() if v["kind"] == kind}

def sync_scroll_position(mode_key):
    """
    Streamlit reruns the whole script on every widget click and normally
    snaps the page back to the top. This keeps a per-workspace scroll offset
    in the browser's sessionStorage (survives reruns, since Streamlit never
    does a real page navigation) and restores it right after switching, so
    coming back to a workspace lands you where you left it.
    """
    key = mode_key or "landing"
    components.html(f"""
        <script>
        (function() {{
            const win = window.parent;
            const storeKey = "streamlit_scroll_{key}";
            const saved = win.sessionStorage.getItem(storeKey);
            if (saved !== null) {{
                setTimeout(() => win.scrollTo(0, parseInt(saved, 10)), 60);
            }}
            if (!win.__scrollSyncAttached) {{
                win.__scrollSyncAttached = true;
                let ticking = false;
                win.addEventListener('scroll', function() {{
                    if (!ticking) {{
                        win.requestAnimationFrame(function() {{
                            const activeKey = win.__activeScrollKey || "landing";
                            win.sessionStorage.setItem("streamlit_scroll_" + activeKey, win.scrollY);
                            ticking = false;
                        }});
                        ticking = true;
                    }}
                }});
            }}
            win.__activeScrollKey = "{key}";
        }})();
        </script>
    """, height=0)

# Prompts built once at module load — reused across calls.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Given the chat history and the latest user question, "
               "rewrite it as a standalone question that can be understood "
               "without the chat history. Do NOT answer it, just rewrite it. "
               "If the question is already standalone, return it unchanged."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document Q&A assistant. Answer the question using ONLY the "
     "context below. If the answer isn't in the context, say you don't "
     "have enough information. Cite the source and date for every claim. "
     "If sources disagree, mention both.\n\nContext:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

def is_summary_query(query):
    q = query.lower()
    return any(kw in q for kw in SUMMARY_KEYWORDS)

def get_full_context_text(kind, scope_key=None):
    """Pull every chunk belonging to a kind (optionally one specific group)
    directly from the FAISS docstore — used for whole-document summaries so
    we don't miss content just because it scored low on embedding similarity."""
    store = st.session_state.vectorstore.docstore._dict
    matched = [
        d for d in store.values()
        if d.metadata.get("kind") == kind
        and (scope_key is None or d.metadata.get("group_key") == scope_key)
    ]
    matched.sort(key=lambda d: (d.metadata.get("group_key", ""), d.metadata.get("page", 0)))

    parts, seen_group = [], None
    for d in matched:
        if d.metadata.get("group_key") != seen_group:
            seen_group = d.metadata.get("group_key")
            parts.append(f"\n[{d.metadata['source']}, {d.metadata['date']}]")
        parts.append(d.page_content)
    text = "\n".join(parts)
    truncated = len(text) > MAX_SUMMARY_CHARS
    return text[:MAX_SUMMARY_CHARS], matched, truncated

def run_conversational_rag(query, chat_history, kind, scope_key):
    """
    Two-step conversational RAG, scoped to ONE kind ('news' or 'pdf') so the
    two chat tabs never bleed into each other. scope_key narrows further to
    a single loaded source within that kind, or None to search all of them.
    """
    if is_summary_query(query):
        context, matched_docs, truncated = get_full_context_text(kind, scope_key)
        if truncated:
            context += "\n\n[...content truncated for length...]"
        qa_messages = QA_PROMPT.format_messages(
            context=context, chat_history=chat_history, input=query
        )
        answer = llm.invoke(qa_messages).content
        return answer, matched_docs

    search_kwargs = {"k": 8 if scope_key else 6}
    search_kwargs["filter"] = {"group_key": scope_key} if scope_key else {"kind": kind}
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs=search_kwargs)

    if chat_history:
        rewrite_messages = CONTEXTUALIZE_PROMPT.format_messages(
            chat_history=chat_history, input=query
        )
        standalone_query = llm.invoke(rewrite_messages).content.strip()
    else:
        standalone_query = query

    retrieved_docs = retriever.invoke(standalone_query)

    context = "\n\n".join(
        f"[{d.metadata['source']}, {d.metadata['date']}] {d.metadata['title']}\n{d.page_content}"
        for d in retrieved_docs
    ) if retrieved_docs else "No relevant content found."

    qa_messages = QA_PROMPT.format_messages(
        context=context, chat_history=chat_history, input=query
    )
    answer = llm.invoke(qa_messages).content

    return answer, retrieved_docs

KIND_AVATAR = {"news": "📰", "pdf": "📄", "topic": "🧭"}
KIND_PLACEHOLDER = {
    "news": "Ask about the loaded newspaper(s)...",
    "pdf": "Ask about the uploaded PDF(s)...",
    "topic": "Ask about the topic articles you've pulled...",
}

def render_chat_panel(kind, chat_history_key, scope_key, empty_message):
    """Renders one fully self-contained chat panel (history, input, sources)
    for a given kind ('news', 'pdf', or 'topic'). Completely independent of
    the other tabs — separate history, separate retrieval scope."""
    if not groups_of_kind(kind):
        st.info(empty_message)
        return

    avatar = KIND_AVATAR[kind]
    st.subheader("💬 Chat")
    hist = st.session_state[chat_history_key]

    top_col, btn_col = st.columns([4, 1])
    with top_col:
        if scope_key:
            label = st.session_state.loaded_groups[scope_key]["label"]
            st.markdown(f'<span class="scope-pill">🎯 Talking to: {label}</span>', unsafe_allow_html=True)
    with btn_col:
        if hist and st.button("🆕 New chat", key=f"new_chat_{kind}", use_container_width=True):
            st.session_state[chat_history_key] = []
            st.rerun()

    for msg in hist:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        msg_avatar = "🧑" if role == "user" else avatar
        with st.chat_message(role, avatar=msg_avatar):
            st.write(msg.content)

    query = st.chat_input(KIND_PLACEHOLDER[kind], key=f"chat_input_{kind}")
    if query:
        # Typing "restart" / "reset" / "new chat" clears this thread instead
        # of being sent to the model — same effect as the button above.
        if query.strip().lower() in RESTART_KEYWORDS:
            st.session_state[chat_history_key] = []
            st.rerun()

        with st.chat_message("user", avatar="🧑"):
            st.write(query)
        with st.chat_message("assistant", avatar=avatar):
            with st.spinner("Thinking..."):
                answer, retrieved_docs = run_conversational_rag(query, hist, kind, scope_key)
            st.write(answer)
            if retrieved_docs:
                with st.expander(f"📚 Sources used ({len(retrieved_docs)})"):
                    for d in retrieved_docs:
                        m = d.metadata
                        if m.get("link"):
                            st.markdown(f"**[{m['title']}]({m['link']})**")
                        else:
                            st.markdown(f"**{m['title']}**")
                        st.caption(f"{m['source']} — {m['date']}")

        st.session_state[chat_history_key].append(HumanMessage(content=query))
        st.session_state[chat_history_key].append(AIMessage(content=answer))

        st.rerun()

# ---------- UI: HEADER (big, centered) ----------
st.markdown(
    '<div class="main-header-wrap"><span class="main-header">📰 NewsRAG</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Three independent RAG workspaces — Frontpage Digest, Topic Radar, '
    'and Document Vault — powered by LangChain.</p>',
    unsafe_allow_html=True,
)

# ---------- SIDEBAR: STATUS ONLY ----------
with st.sidebar:
    st.header("📊 Session status")
    news_count = sum(v["count"] for v in groups_of_kind("news").values())
    pdf_count = sum(v["count"] for v in groups_of_kind("pdf").values())
    topic_count = sum(v["count"] for v in groups_of_kind("topic").values())
    c1, c2, c3 = st.columns(3)
    c1.metric("News articles", news_count)
    c2.metric("Topic articles", topic_count)
    c3.metric("PDF pages", pdf_count)
    st.metric("Chunks indexed", st.session_state.total_chunks)

    if groups_of_kind("news"):
        st.divider()
        st.subheader("📰 Newspapers loaded")
        badges = "".join(f'<span class="source-badge">{v["label"]}</span>' for v in groups_of_kind("news").values())
        st.markdown(badges, unsafe_allow_html=True)

    if groups_of_kind("topic"):
        st.divider()
        st.subheader("🧭 Topics loaded")
        badges = "".join(f'<span class="source-badge">{v["label"]}</span>' for v in groups_of_kind("topic").values())
        st.markdown(badges, unsafe_allow_html=True)

    if groups_of_kind("pdf"):
        st.divider()
        st.subheader("📄 PDFs loaded")
        badges = "".join(f'<span class="source-badge">{v["label"]}</span>' for v in groups_of_kind("pdf").values())
        st.markdown(badges, unsafe_allow_html=True)

# ---------- MODE SELECTOR: fully independent Frontpage / Topic / PDF workspaces ----------
st.markdown('<p class="mode-select-heading">Choose a workspace to get started</p>', unsafe_allow_html=True)

mode_cols = st.columns(3)
for col, mode_key in zip(mode_cols, MODES):
    m = MODES[mode_key]
    is_active = st.session_state.active_mode == mode_key
    with col:
        st.markdown(f"""
        <div class="mode-card {'active' if is_active else ''}"
             style="--mc-bg:{m['bg']}; --mc-border:{m['border']}; --mc-glow:{m['glow']}; --mc-color:{m['color']};">
            <div class="mode-icon">{m['emoji']}</div>
            <div class="mode-title">{m['title']}</div>
            <div class="mode-desc">{m['desc']}</div>
            <div class="mode-badge">{m['badge']}</div>
            <div class="mode-hint">↓ Click below to select</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button(
            m['button_label'], key=f"select_{mode_key}", use_container_width=True,
            type="primary" if is_active else "secondary",
        ):
            st.session_state.active_mode = mode_key
            st.rerun()

st.write("")
st.divider()

active_mode = st.session_state.active_mode
sync_scroll_position(active_mode)
if active_mode is None:
    st.info("👆 Pick a workspace above — each keeps its own sources and its own chat.")
else:
    m = MODES[active_mode]
    st.markdown(
        f'<div class="active-mode-banner" style="color:{m["color"]};">{m["emoji"]} {m["title"]}</div>',
        unsafe_allow_html=True,
    )

if active_mode == "news":
    st.markdown("#### Load newspaper coverage")
    earliest_allowed = datetime.date.today() - datetime.timedelta(days=MAX_DAYS_BACK)
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        newspaper_name = st.selectbox("Newspaper", list(NEWSPAPER_NAME_MATCHES.keys()), key="newspaper_select")
    with col2:
        target_date = st.date_input(
            "Date", value=datetime.date.today(),
            min_value=earliest_allowed, max_value=datetime.date.today(),
            key="newspaper_date",
        )
    with col3:
        st.write("")
        st.write("")
        load_clicked = st.button("Load articles", use_container_width=True, type="primary", key="load_news_btn")

    if load_clicked:
        group_key = f"news::{newspaper_name}::{target_date.isoformat()}"
        group_label = f"{NEWSPAPER_ICONS.get(newspaper_name, '📰')} {newspaper_name} · {target_date}"
        with st.spinner(f"Fetching {newspaper_name} — {target_date}..."):
            raw = fetch_articles(newspaper_name, target_date)
            docs = articles_to_documents(raw, newspaper_name, target_date, group_key)
            build_or_update_vectorstore(docs, group_key, group_label, "news")
        if docs:
            st.success(f"✅ Loaded {len(docs)} articles from {newspaper_name} ({target_date}).")
        else:
            st.warning("No new articles found for that newspaper/date combination.")

    news_groups = groups_of_kind("news")
    news_scope_key = None
    if len(news_groups) > 1:
        options = ["__all__"] + list(news_groups.keys())
        news_scope_key = st.radio(
            "Chat about", options,
            format_func=lambda k: "🌐 All loaded newspapers" if k == "__all__" else news_groups[k]["label"],
            horizontal=True, key="news_scope",
        )
        news_scope_key = None if news_scope_key == "__all__" else news_scope_key

    st.divider()
    render_chat_panel(
        "news", "chat_history_news", news_scope_key,
        "👆 Load a newspaper above to start chatting.",
    )

elif active_mode == "topic":
    st.markdown("#### Search articles by topic")
    st.caption("Search any keyword or subject across all of NewsAPI's publishers — not limited to one newspaper.")
    earliest_allowed_topic = datetime.date.today() - datetime.timedelta(days=MAX_DAYS_BACK)
    col1, col2, col3, col4 = st.columns([2, 1.3, 1.3, 1])
    with col1:
        topic_query = st.text_input("Topic or keyword", placeholder="e.g. climate change, elections, AI regulation", key="topic_input")
    with col2:
        topic_from = st.date_input(
            "From", value=earliest_allowed_topic,
            min_value=earliest_allowed_topic, max_value=datetime.date.today(),
            key="topic_from",
        )
    with col3:
        topic_to = st.date_input(
            "To", value=datetime.date.today(),
            min_value=earliest_allowed_topic, max_value=datetime.date.today(),
            key="topic_to",
        )
    with col4:
        st.write("")
        st.write("")
        load_topic_clicked = st.button("Search", use_container_width=True, type="primary", key="load_topic_btn")

    if load_topic_clicked:
        if not topic_query.strip():
            st.warning("Enter a topic or keyword first.")
        elif topic_from > topic_to:
            st.warning("'From' date must be before 'To' date.")
        else:
            group_key = f"topic::{topic_query.strip().lower()}::{topic_from.isoformat()}::{topic_to.isoformat()}"
            group_label = f"🔍 \"{topic_query.strip()}\" ({topic_from} → {topic_to})"
            with st.spinner(f"Searching articles about '{topic_query}'..."):
                raw = fetch_topic_articles(topic_query.strip(), topic_from, topic_to)
                docs = topic_articles_to_documents(raw, topic_query.strip(), group_key)
                build_or_update_vectorstore(docs, group_key, group_label, "topic")
            if docs:
                st.success(f"✅ Loaded {len(docs)} articles about '{topic_query}'.")
            else:
                st.warning("No new articles found for that topic/date range.")

    topic_groups = groups_of_kind("topic")
    topic_scope_key = None
    if len(topic_groups) > 1:
        options = ["__all__"] + list(topic_groups.keys())
        topic_scope_key = st.radio(
            "Chat about", options,
            format_func=lambda k: "🌐 All loaded topics" if k == "__all__" else topic_groups[k]["label"],
            horizontal=True, key="topic_scope",
        )
        topic_scope_key = None if topic_scope_key == "__all__" else topic_scope_key

    st.divider()
    render_chat_panel(
        "topic", "chat_history_topic", topic_scope_key,
        "👆 Search a topic above to start chatting.",
    )

elif active_mode == "pdf":
    st.markdown("#### Upload a PDF")
    st.caption("Upload one or more article PDFs — this chat only sees PDFs, never newspaper content.")
    uploaded_pdfs = st.file_uploader(
        "Choose PDF file(s)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader"
    )
    if uploaded_pdfs and st.button("Load PDF(s)", type="primary", key="load_pdf_btn"):
        total_new_pages = 0
        for uploaded_file in uploaded_pdfs:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                docs, group_key = pdf_to_documents(uploaded_file)
                build_or_update_vectorstore(docs, group_key, f"📄 {uploaded_file.name}", "pdf")
            if docs:
                total_new_pages += len(docs)
                st.success(f"✅ Loaded {len(docs)} page(s) from {uploaded_file.name}.")
            else:
                st.info(f"'{uploaded_file.name}' was already loaded or had no extractable text.")
        if total_new_pages == 0:
            st.warning("No new text was extracted from the uploaded file(s).")

    pdf_groups = groups_of_kind("pdf")
    pdf_scope_key = None
    if len(pdf_groups) > 1:
        options = ["__all__"] + list(pdf_groups.keys())
        pdf_scope_key = st.radio(
            "Chat about", options,
            format_func=lambda k: "🌐 All loaded PDFs" if k == "__all__" else pdf_groups[k]["label"],
            horizontal=True, key="pdf_scope",
        )
        pdf_scope_key = None if pdf_scope_key == "__all__" else pdf_scope_key

    st.divider()
    render_chat_panel(
        "pdf", "chat_history_pdf", pdf_scope_key,
        "👆 Upload a PDF above to start chatting.",
    )
