import ast
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def _tokenize(text: str) -> list[str]:
    return str(text).lower().split()


@st.cache_resource(show_spinner=False)
def _build_bm25_index(corpus: list[str]) -> dict[str, Any]:
    tokenized_docs = [_tokenize(doc) for doc in corpus]
    doc_freqs = [Counter(doc) for doc in tokenized_docs]
    doc_lengths = np.array([len(doc) for doc in tokenized_docs], dtype=np.float32)
    avgdl = float(doc_lengths.mean()) if len(doc_lengths) else 0.0

    n_docs = len(tokenized_docs)
    term_doc_count: Counter[str] = Counter()
    for freq in doc_freqs:
        term_doc_count.update(freq.keys())

    idf = {
        term: math.log(1 + (n_docs - count + 0.5) / (count + 0.5))
        for term, count in term_doc_count.items()
    }

    return {
        "doc_freqs": doc_freqs,
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "idf": idf,
        "n_docs": n_docs,
    }


@st.cache_data(show_spinner=False)
def load_models() -> dict[str, Any]:
    base = Path(__file__).resolve().parent

    with open(base / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(base / "products.pkl", "rb") as f:
        products = pickle.load(f)
    if not isinstance(products, pd.DataFrame):
        products = pd.DataFrame(products)

    with open(base / "bm25_corpus.pkl", "rb") as f:
        bm25_corpus = pickle.load(f)
    if not isinstance(bm25_corpus, list):
        bm25_corpus = list(bm25_corpus)

    min_len = min(len(products), len(bm25_corpus))
    products = products.iloc[:min_len].copy().reset_index(drop=True)
    bm25_corpus = bm25_corpus[:min_len]

    products["bm25_text"] = bm25_corpus
    products["product_id"] = products.get("uniq_id", products.index.astype(str))
    products["product_name"] = products.get("product_name", "").fillna("Unknown Product")
    products["image"] = products.get("image", "").fillna("")
    products["retail_price"] = products.get("retail_price", 0.0).fillna(0.0)
    products["discounted_price"] = products.get("discounted_price", 0.0).fillna(0.0)

    bm25_index = _build_bm25_index(bm25_corpus)

    faiss_index = None
    sentence_model = None
    try:
        import faiss  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        faiss_index = faiss.read_index(str(base / "best_model.index"))
        sentence_model = SentenceTransformer(meta.get("model_name", "all-MiniLM-L6-v2"))
    except Exception:
        # Optional semantic scoring: UI still works with BM25-only.
        faiss_index = None
        sentence_model = None

    return {
        "meta": meta,
        "products": products,
        "bm25_index": bm25_index,
        "faiss_index": faiss_index,
        "sentence_model": sentence_model,
    }


def _bm25_scores(query: str, bm25_index: dict[str, Any], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    terms = _tokenize(query)
    n_docs = bm25_index["n_docs"]
    scores = np.zeros(n_docs, dtype=np.float32)
    if not terms:
        return scores

    doc_freqs = bm25_index["doc_freqs"]
    doc_lengths = bm25_index["doc_lengths"]
    avgdl = bm25_index["avgdl"] or 1.0
    idf = bm25_index["idf"]

    for idx in range(n_docs):
        doc_tf = doc_freqs[idx]
        dl = doc_lengths[idx]
        denom_norm = k1 * (1 - b + b * (dl / avgdl))
        total = 0.0
        for term in terms:
            tf = doc_tf.get(term, 0)
            if tf == 0:
                continue
            term_idf = idf.get(term, 0.0)
            total += term_idf * ((tf * (k1 + 1)) / (tf + denom_norm))
        scores[idx] = total
    return scores


def _normalize(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v - min_v < 1e-9:
        return np.zeros_like(arr)
    return (arr - min_v) / (max_v - min_v)


def search_products(query: str, top_k: int = 20) -> pd.DataFrame:
    assets = load_models()
    products = assets["products"]
    bm25_scores = _bm25_scores(query, assets["bm25_index"])

    alpha = float(assets["meta"].get("best_alpha", 0.5))
    final_scores = bm25_scores.copy()

    if assets["faiss_index"] is not None and assets["sentence_model"] is not None:
        query_vec = assets["sentence_model"].encode([query], convert_to_numpy=True).astype("float32")
        faiss_scores, faiss_indices = assets["faiss_index"].search(query_vec, min(top_k * 4, len(products)))
        semantic = np.zeros(len(products), dtype=np.float32)
        sim = np.asarray(faiss_scores[0], dtype=np.float32)
        idxs = np.asarray(faiss_indices[0], dtype=np.int64)
        valid_mask = idxs >= 0
        semantic[idxs[valid_mask]] = sim[valid_mask]

        bm25_n = _normalize(bm25_scores)
        sem_n = _normalize(semantic)
        final_scores = alpha * sem_n + (1.0 - alpha) * bm25_n

    top_idx = np.argsort(final_scores)[::-1][:top_k]
    out = products.iloc[top_idx][["product_id", "product_name", "image", "retail_price", "discounted_price"]].copy()
    out["score"] = final_scores[top_idx]
    out = out[out["score"] > 0.01].reset_index(drop=True)
    return out


def _extract_image_url(image_value: Any) -> str:
    url = ""
    if image_value is None:
        url = ""
    elif isinstance(image_value, list):
        url = str(image_value[0]) if image_value else ""
    else:
        text = str(image_value).strip()
        if text:
            parsed_ok = False
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list) and parsed:
                    url = str(parsed[0])
                    parsed_ok = True
            except Exception:
                pass
            if not parsed_ok:
                try:
                    parsed = ast.literal_eval(text)
                    if isinstance(parsed, list) and parsed:
                        url = str(parsed[0])
                        parsed_ok = True
                except Exception:
                    pass
            if not parsed_ok:
                cleaned = text.replace('[', '').replace(']', '').replace('"', '').replace("'", "")
                url = cleaned.split(',')[0].strip()

    if url:
        url = url.replace("http://", "https://")

    if not url:
        return "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIiB2aWV3Qm94PSIwIDAgMzAwIDMwMCI+PHJlY3Qgd2lkdGg9IjMwMCIgaGVpZ2h0PSIzMDAiIGZpbGw9IiNmOGZhZmMiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZm9udC1mYW1pbHk9InNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMjAiIGZpbGw9IiM2NDc0OGIiIHRleHQtYW5jaG9yPSJtaWRkbGUiIGRvbWluYW50LWJhc2VsaW5lPSJtaWRkbGUiPk5vIEltYWdlPC90ZXh0Pjwvc3ZnPg=="
    return url


def display_results(results: pd.DataFrame) -> None:
    if results.empty:
        st.info("No results found")
        return

    cols = st.columns(4)
    for i, row in results.iterrows():
        with cols[i % 4]:
            image_url = _extract_image_url(row.get("image", ""))
            score_val = float(row.get("score", 0.0))
            retail = float(row.get("retail_price", 0.0))
            discounted = float(row.get("discounted_price", 0.0))
            
            badge_html = ""
            if discounted > 0 and discounted < retail:
                discount_pct = int((1 - discounted/retail) * 100)
                badge_html = f'<div class="sale-badge">{discount_pct}% OFF</div>'
            
            price_html = ""
            if discounted > 0 and discounted < retail:
                price_html = f'<div class="price-wrap"><span class="price-new">₹{discounted:,.2f}</span><span class="price-old">₹{retail:,.2f}</span></div>'
            elif retail > 0:
                price_html = f'<div class="price-wrap"><span class="price-new">₹{retail:,.2f}</span></div>'
            elif discounted > 0:
                price_html = f'<div class="price-wrap"><span class="price-new">₹{discounted:,.2f}</span></div>'
            else:
                price_html = f'<div class="price-wrap"><span class="price-new">Price Unavailable</span></div>'
            
            product_name = str(row['product_name']).replace('"', '&quot;')
            
            card_html = f"""<div class="premium-card fade-in" style="animation-delay: {min(i * 0.05, 0.5)}s">
<div class="card-image-container">
{badge_html}
<img class="card-image" src="{image_url}" alt="{product_name}" />
</div>
<div class="card-content">
<div class="product-brand">PREMIUM</div>
<h3 class="product-title" title="{product_name}">{product_name}</h3>
{price_html}
<div class="card-footer">
<div class="match-score">
<span class="score-dot"></span>
Match: {score_val:.2f}
</div>
</div>
</div>
</div>"""
            st.markdown(card_html, unsafe_allow_html=True)


st.set_page_config(page_title="Product Search", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Stunning Background */
    .stApp {
        background-color: #fafcff;
        background-image: 
            radial-gradient(at 0% 0%, hsla(253,16%,97%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(225,39%,96%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(339,49%,97%,1) 0, transparent 50%);
        background-attachment: fixed;
        color: #0f172a;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: center;
        padding: 3rem 0 2rem 0;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #111827 0%, #4338ca 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 500;
    }

    /* First-class Input box */
    div[data-testid="stTextInput"] {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Override Streamlit's default wrapper styles */
    div[data-testid="stTextInput"] div[data-baseweb="input"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 99px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0,0,0,0.02) !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        overflow: visible !important;
    }
    
    /* When focused, change the wrapper */
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within {
        border-color: #6366f1 !important;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.15), 0 0 0 4px rgba(99, 102, 241, 0.1) !important;
        transform: translateY(-2px);
    }

    /* Style the actual input field */
    div[data-testid="stTextInput"] input {
        color: #0f172a !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        padding: 16px 24px !important;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border: none !important;
        box-shadow: none !important;
        transform: none !important;
        outline: none !important;
    }

    /* Premium Card Design */
    .premium-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 20px;
        margin-bottom: 30px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        overflow: hidden;
        height: 100%;
        display: flex;
        flex-direction: column;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05), inset 0 0 0 1px rgba(255, 255, 255, 0.5);
    }
    /* Image container */
    .card-image-container {
        width: 100%;
        aspect-ratio: 1;
        position: relative;
        overflow: hidden;
        background: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
    }
    .card-image {
        width: 100%;
        height: 100%;
        object-fit: contain;
        mix-blend-mode: multiply;
    }
    
    /* Badges & Overlays */
    .sale-badge {
        position: absolute;
        top: 12px;
        left: 12px;
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        z-index: 2;
        box-shadow: 0 4px 10px rgba(239, 68, 68, 0.3);
    }
    /* Removed overlay & btn */
    
    /* Content */
    .card-content {
        padding: 20px;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        background: white;
    }
    .product-brand {
        font-size: 0.7rem;
        font-weight: 700;
        color: #6366f1;
        letter-spacing: 0.1em;
        margin-bottom: 6px;
    }
    .product-title {
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.3;
        color: #0f172a;
        margin-bottom: 16px;
        margin-top: 0;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Pricing */
    .price-wrap {
        margin-top: auto;
        margin-bottom: 16px;
        display: flex;
        align-items: baseline;
        gap: 8px;
        flex-wrap: wrap;
    }
    .price-new {
        font-size: 1.4rem;
        font-weight: 800;
        color: #0f172a;
    }
    .price-old {
        font-size: 0.9rem;
        color: #94a3b8;
        text-decoration: line-through;
        font-weight: 500;
    }
    
    /* Footer */
    .card-footer {
        padding-top: 16px;
        border-top: 1px solid #f1f5f9;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .match-score {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .score-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 2px rgba(16,185,129,0.2);
    }
    
    /* Animations */
    .fade-in {
        animation: slideUpFade 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(30px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">Discover Premium Products</h1>
    <p class="hero-subtitle">Experience the next generation of intelligent product search.</p>
</div>
""", unsafe_allow_html=True)

query = st.text_input(" ", placeholder="What are you looking for today?...", label_visibility="collapsed")

assets = load_models()

if query.strip():
    with st.spinner("Searching products..."):
        results_df = search_products(query.strip(), top_k=20)
    display_results(results_df)
else:
    st.empty()
