import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import html
import re
import os
import hashlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENALEX_EMAIL = os.getenv("OPENALEX_EMAIL", "")


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at 15% 15%,
                rgba(34, 211, 238, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 20%,
                rgba(99, 102, 241, 0.20),
                transparent 30%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(139, 92, 246, 0.14),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #020617 0%,
                #07152f 45%,
                #0b1f3a 100%
            );

        color: #e5faff;
        min-height: 100vh;
    }

    .stApp::before {
        content: "";
        position: fixed;
        width: 420px;
        height: 420px;
        top: 80px;
        left: -180px;
        background: rgba(34, 211, 238, 0.14);
        filter: blur(100px);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }

    .stApp::after {
        content: "";
        position: fixed;
        width: 450px;
        height: 450px;
        right: -180px;
        bottom: 40px;
        background: rgba(99, 102, 241, 0.15);
        filter: blur(110px);
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
    }

    .main .block-container {
        position: relative;
        z-index: 1;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
        max-width: 1250px;
    }

    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;

        background:
            linear-gradient(
                90deg,
                #67e8f9,
                #22d3ee,
                #818cf8,
                #c084fc
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        text-shadow:
            0 0 35px rgba(34, 211, 238, 0.20);

        margin-bottom: 0.1rem;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.1rem;
    }

    h1, h2, h3, h4 {
        color: #f0fdff !important;
    }

    .stButton > button {
        min-height: 45px;
        border-radius: 12px !important;

        background:
            linear-gradient(
                135deg,
                #0369a1,
                #0891b2
            ) !important;

        border: 1px solid rgba(103, 232, 249, 0.30) !important;
        color: white !important;
        font-weight: 700 !important;

        box-shadow:
            0 8px 20px rgba(8, 145, 178, 0.22) !important;

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease !important;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                #0891b2,
                #4f46e5
            ) !important;

        transform: translateY(-2px) !important;

        box-shadow:
            0 14px 30px rgba(34, 211, 238, 0.25) !important;
    }

    .quick-question button {
        min-height: 52px !important;
        border-radius: 14px !important;

        background:
            linear-gradient(
                135deg,
                #0369a1,
                #0891b2
            ) !important;

        border: 1px solid rgba(103, 232, 249, 0.35) !important;

        color: white !important;
        font-size: 12px !important;
        font-weight: 700 !important;

        padding: 7px 5px !important;

        box-shadow:
            0 8px 20px rgba(8, 145, 178, 0.25) !important;

        white-space: normal !important;
    }

    .quick-question button:hover {
        background:
            linear-gradient(
                135deg,
                #0891b2,
                #4f46e5
            ) !important;

        transform: translateY(-3px) !important;

        box-shadow:
            0 14px 30px rgba(34, 211, 238, 0.30) !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.72);
        border: 1px dashed rgba(103, 232, 249, 0.35);
        border-radius: 16px;
        padding: 0.8rem;
    }

    [data-testid="stFileUploader"] section {
        background: transparent !important;
    }

    .stTextInput input {
        background: rgba(2, 6, 23, 0.78) !important;
        color: #e2faff !important;
        border: 1px solid rgba(103, 232, 249, 0.22) !important;
        border-radius: 12px !important;
    }

    .stTextInput input:focus {
        border-color: #22d3ee !important;

        box-shadow:
            0 0 0 2px rgba(34, 211, 238, 0.12) !important;
    }

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                135deg,
                rgba(15, 23, 42, 0.92),
                rgba(8, 47, 73, 0.72)
            );

        border: 1px solid rgba(103, 232, 249, 0.18);
        border-radius: 16px;
        padding: 1rem;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.22);
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }

    [data-testid="stMetricValue"] {
        color: #67e8f9 !important;
    }

    .answer-card {
        background:
            linear-gradient(
                135deg,
                rgba(8, 47, 73, 0.90),
                rgba(30, 27, 75, 0.80)
            );

        border-left: 4px solid #22d3ee;
        border-radius: 14px;

        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;

        box-shadow:
            0 12px 28px rgba(0, 0, 0, 0.25);
    }

    .answer-card p {
        color: #e2faff;
        line-height: 1.65;
        margin: 0;
    }

    .quick-info {
        color: #94a3b8;
        font-size: 0.9rem;
        margin-bottom: 0.8rem;
    }

    .section-divider {
        height: 1px;

        background:
            linear-gradient(
                90deg,
                transparent,
                rgba(103, 232, 249, 0.25),
                transparent
            );

        margin: 1.2rem 0;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 1.5rem;
        padding-top: 1rem;

        border-top:
            1px solid rgba(103, 232, 249, 0.08);
    }

    a {
        color: #67e8f9 !important;
        text-decoration: none !important;
    }

    a:hover {
        color: #c084fc !important;
        text-decoration: underline !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "processed" not in st.session_state:
    st.session_state.processed = False

if "paper_name" not in st.session_state:
    st.session_state.paper_name = ""

if "paper_text" not in st.session_state:
    st.session_state.paper_text = ""

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "local_results" not in st.session_state:
    st.session_state.local_results = []

if "internet_results" not in st.session_state:
    st.session_state.internet_results = []

if "internet_checked" not in st.session_state:
    st.session_state.internet_checked = False


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    if not text:
        return chunks

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start = end - overlap

        if start >= len(text):
            break

    return chunks


# ============================================================
# CLEAN ANSWER
# ============================================================

def clean_answer(answer):

    if not answer:
        return "No answer was generated."

    answer = str(answer)

    answer = re.sub(
        r"<[^>]*>",
        "",
        answer
    )

    answer = answer.replace(
        "```html",
        ""
    )

    answer = answer.replace(
        "```",
        ""
    )

    prefixes = [
        "Answer:",
        "AI Answer:",
        "Response:",
        "Final Answer:"
    ]

    answer = answer.strip()

    for prefix in prefixes:

        if answer.lower().startswith(
            prefix.lower()
        ):

            answer = answer[
                len(prefix):
            ].strip()

    return answer.strip()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# CHROMA RAG COLLECTION
# ============================================================

@st.cache_resource
def get_chroma_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    return client.get_or_create_collection(
        name="research_papers"
    )


collection = get_chroma_collection()


# ============================================================
# GEMINI SIMILARITY COLLECTION
# ============================================================

@st.cache_resource
def get_gemini_collection():

    client = chromadb.PersistentClient(
        path="chroma_db"
    )

    try:

        return client.get_or_create_collection(
            name="research_paper_gemini",
            metadata={
                "hnsw:space": "cosine"
            }
        )

    except Exception:

        return None


gemini_collection = get_gemini_collection()


# ============================================================
# GEMINI CLIENT
# ============================================================

@st.cache_resource
def get_gemini_client():

    if not GEMINI_API_KEY:
        return None

    try:

        return genai.Client(
            api_key=GEMINI_API_KEY
        )

    except Exception:

        return None


gemini_client = get_gemini_client()


# ============================================================
# SEARCH PAPER
# ============================================================

def search_paper(
    question,
    source,
    n_results=5
):

    try:

        question_embedding = (
            embedding_model.encode(
                [question]
            )
        )

        results = collection.query(
            query_embeddings=(
                question_embedding.tolist()
            ),
            n_results=n_results,
            where={
                "source": source
            }
        )

        documents = results.get(
            "documents",
            []
        )

        if len(documents) == 0:
            return ""

        if len(documents[0]) == 0:
            return ""

        return "\n\n".join(
            documents[0]
        )

    except Exception:

        return ""


# ============================================================
# SECTION CONTEXT
# ============================================================

def get_section_context(
    text,
    question_type
):

    if not text:
        return ""

    section_keywords = {

        "objective": [
            "objective",
            "objectives",
            "aim",
            "purpose",
            "research goal",
            "research question"
        ],

        "methodology": [
            "methodology",
            "method",
            "methods",
            "proposed method",
            "proposed approach",
            "approach",
            "architecture",
            "framework",
            "experimental setup",
            "research methodology"
        ],

        "dataset": [
            "dataset",
            "datasets",
            "data collection",
            "data source",
            "data used",
            "experimental data"
        ],

        "conclusion": [
            "conclusion",
            "conclusions",
            "concluding remarks",
            "final remarks",
            "discussion and conclusion"
        ],

        "limitation": [
            "limitation",
            "limitations",
            "future work",
            "future research",
            "further work",
            "challenges",
            "constraints",
            "weaknesses",
            "shortcomings",
            "areas for improvement",
            "recommendations"
        ]
    }

    keywords = section_keywords.get(
        question_type,
        []
    )

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    relevant = []

    for paragraph in paragraphs:

        cleaned = re.sub(
            r"\s+",
            " ",
            paragraph
        ).strip()

        if not cleaned:
            continue

        lower = cleaned.lower()

        score = 0

        for keyword in keywords:

            if keyword in lower:
                score += 1

        if score > 0:

            relevant.append(
                (
                    score,
                    cleaned
                )
            )

    relevant.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = [
        item[1]
        for item in relevant[:20]
    ]

    if selected:
        return "\n\n".join(selected)

    return ""


# ============================================================
# GENERATE AI ANSWER
# ============================================================

def generate_answer(
    question,
    source,
    question_type="general",
    n_results=5
):

    semantic_context = search_paper(
        question,
        source,
        12 if question_type != "general"
        else n_results
    )

    full_text = (
        st.session_state.paper_text
    )

    section_context = ""

    if question_type != "general":

        section_context = get_section_context(
            full_text,
            question_type
        )

    context_parts = []

    if section_context:

        context_parts.append(
            "RELEVANT SECTION TEXT:\n"
            + section_context
        )

    if semantic_context:

        context_parts.append(
            "SEMANTICALLY RELEVANT TEXT:\n"
            + semantic_context
        )

    if question_type in [
        "limitation",
        "conclusion",
        "summary"
    ]:

        if full_text:

            context_parts.append(
                "FULL RESEARCH PAPER:\n"
                + full_text
            )

    context = "\n\n".join(
        context_parts
    )

    if not context:

        return (
            "I could not find this information "
            "in the uploaded research paper."
        )

    instructions = {

        "objective": """
Answer ONLY the main objective or purpose.

Look for objective, aim, purpose,
research goal, or problem being addressed.

Do not explain methodology, dataset,
results, limitations, or conclusion.
""",

        "methodology": """
Answer ONLY the methodology.

Look for methodology, methods,
proposed approach, algorithms, models,
architecture, framework, and procedures.

Do not provide unrelated results or conclusion.
""",

        "dataset": """
Answer ONLY about the dataset or data.

Look for dataset name, data source,
data collection, sample count,
training data, testing data,
validation data, and data characteristics.

Do not provide methodology or conclusion.
""",

        "conclusion": """
Answer ONLY the conclusion.

Look for conclusion, conclusions,
concluding remarks, final remarks,
or discussion and conclusion.

Do not give a general summary.
""",

        "limitation": """
Answer ONLY the limitations and future work.

Search the entire paper.

Look for:
- Limitations
- Study limitations
- Research limitations
- Challenges
- Constraints
- Weaknesses
- Shortcomings
- Future Work
- Future Research
- Further Work
- Areas for Improvement
- Recommendations

The information may appear in Discussion
or Conclusion rather than a section named
"Limitations".

Do not invent limitations.

If there is genuinely no information about
limitations or future work anywhere in the paper,
say exactly:

I could not find limitations or future work
in the uploaded research paper.
""",

        "summary": """
Give a concise overall summary.

Include:
1. Problem
2. Objective
3. Methodology
4. Important findings
5. Conclusion.

Keep it concise.
""",

        "general": """
Answer ONLY the user's specific question.
Do not provide unrelated information.
"""
    }

    instruction = instructions.get(
        question_type,
        instructions["general"]
    )

    prompt = f"""
You are an AI Research Paper Assistant.

Use ONLY the provided research paper.

Rules:
1. Never invent information.
2. Never use outside knowledge.
3. Follow the requested question type.
4. Search the entire context carefully.
5. Information may appear under different headings.
6. Do not output HTML.
7. Do not output code.
8. Do not provide unrelated information.

QUESTION TYPE:
{question_type}

INSTRUCTIONS:
{instruction}

RESEARCH PAPER:
============================================================

{context}

============================================================

USER QUESTION:
{question}

============================================================

Give a direct answer.
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500
                }
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        answer = data.get(
            "response",
            ""
        )

        answer = clean_answer(
            answer
        )

        if not answer:

            return (
                "I could not find this information "
                "in the uploaded research paper."
            )

        return answer

    except requests.exceptions.ConnectionError:

        return (
            "Ollama is not running. "
            "Please start Ollama and make sure "
            "llama3.2 is installed."
        )

    except requests.exceptions.Timeout:

        return (
            "Ollama took too long to respond. "
            "Please try again."
        )

    except Exception as e:

        return f"Ollama error: {e}"


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def generate_gemini_embeddings(texts):

    if gemini_client is None:
        return None

    if texts is None:
        return None

    if len(texts) == 0:
        return None

    try:

        result = gemini_client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=768
            )
        )

        vectors = []

        for embedding in result.embeddings:

            vector = np.asarray(
                embedding.values,
                dtype=np.float32
            )

            norm = np.linalg.norm(
                vector
            )

            if norm > 0:
                vector = vector / norm

            vectors.append(vector)

        return np.asarray(
            vectors,
            dtype=np.float32
        )

    except Exception as e:

        st.error(
            f"Gemini embedding error: {e}"
        )

        return None


# ============================================================
# CREATE PAPER GEMINI EMBEDDING
# ============================================================

def create_paper_gemini_embedding(text):

    if not text:
        return None

    chunks = create_chunks(
        text,
        chunk_size=1200,
        overlap=200
    )

    if len(chunks) == 0:
        return None

    if len(chunks) > 8:

        selected = (
            chunks[:5]
            + chunks[-3:]
        )

    else:

        selected = chunks

    embeddings = (
        generate_gemini_embeddings(
            selected
        )
    )

    if embeddings is None:
        return None

    paper_embedding = np.mean(
        embeddings,
        axis=0
    )

    paper_embedding = np.asarray(
        paper_embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        paper_embedding
    )

    if norm > 0:
        paper_embedding = (
            paper_embedding / norm
        )

    return paper_embedding


# ============================================================
# PAPER HASH
# ============================================================

def create_paper_hash(text):

    normalized = re.sub(
        r"\s+",
        " ",
        text.lower()
    ).strip()

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# ADD PAPER TO LOCAL DATABASE
# ============================================================

def add_paper_to_local_database():

    if gemini_client is None:

        st.error(
            "Gemini API key is not configured."
        )

        return

    if gemini_collection is None:

        st.error(
            "Similarity database is unavailable."
        )

        return

    text = (
        st.session_state.paper_text
    )

    if not text:

        st.error(
            "No processed research paper found."
        )

        return

    with st.spinner(
        "Creating Gemini embedding..."
    ):

        embedding = (
            create_paper_gemini_embedding(
                text
            )
        )

    if embedding is None:

        st.error(
            "Could not create paper embedding."
        )

        return

    paper_hash = create_paper_hash(
        text
    )

    try:

        existing = gemini_collection.get(
            where={
                "paper_hash": paper_hash
            }
        )

        existing_ids = existing.get(
            "ids"
        )

        if (
            existing_ids is not None
            and len(existing_ids) > 0
        ):

            st.info(
                "📄 This paper is already in the local database."
            )

            return

        gemini_collection.add(
            ids=[
                paper_hash
            ],
            embeddings=[
                embedding.tolist()
            ],
            documents=[
                text[:5000]
            ],
            metadatas=[
                {
                    "paper_hash":
                        paper_hash,
                    "paper_name":
                        st.session_state.paper_name
                }
            ]
        )

        st.success(
            "✅ Paper added to local similarity database."
        )

    except Exception as e:

        st.error(
            f"Unable to save paper: {e}"
        )


# ============================================================
# LOCAL SIMILARITY
# ============================================================

def check_local_similarity():

    if gemini_client is None:

        st.error(
            "Gemini API key is not configured."
        )

        return []

    if gemini_collection is None:

        st.error(
            "Similarity database is unavailable."
        )

        return []

    text = (
        st.session_state.paper_text
    )

    if not text:

        st.warning(
            "No processed research paper found."
        )

        return []

    with st.spinner(
        "Checking similarity with saved papers..."
    ):

        current_embedding = (
            create_paper_gemini_embedding(
                text
            )
        )

    if current_embedding is None:

        st.error(
            "Could not create the Gemini embedding."
        )

        return []

    try:

        data = gemini_collection.get(
            include=[
                "embeddings",
                "metadatas"
            ]
        )

        embeddings = data.get(
            "embeddings"
        )

        metadatas = data.get(
            "metadatas"
        )

        # No saved papers
        if embeddings is None:

            st.info(
                "📄 No other papers available for comparison."
            )

            return []

        if len(embeddings) == 0:

            st.info(
                "📄 No other papers available for comparison."
            )

            return []

        if metadatas is None:

            st.info(
                "📄 No other papers available for comparison."
            )

            return []

        if len(metadatas) == 0:

            st.info(
                "📄 No other papers available for comparison."
            )

            return []

        current_hash = create_paper_hash(
            text
        )

        results = []

        for vector, metadata in zip(
            embeddings,
            metadatas
        ):

            if metadata is None:
                continue

            saved_hash = metadata.get(
                "paper_hash",
                ""
            )

            # Skip currently uploaded paper
            if saved_hash == current_hash:
                continue

            saved_vector = np.asarray(
                vector,
                dtype=np.float32
            )

            current_vector = np.asarray(
                current_embedding,
                dtype=np.float32
            )

            saved_vector = saved_vector.reshape(
                1,
                -1
            )

            current_vector = current_vector.reshape(
                1,
                -1
            )

            if (
                saved_vector.shape[1]
                != current_vector.shape[1]
            ):
                continue

            similarity = cosine_similarity(
                current_vector,
                saved_vector
            )[0][0]

            similarity_percentage = (
                float(similarity) * 100
            )

            similarity_percentage = max(
                0.0,
                min(
                    100.0,
                    similarity_percentage
                )
            )

            results.append(
                {
                    "paper_name":
                        metadata.get(
                            "paper_name",
                            "Unknown Paper"
                        ),
                    "similarity":
                        similarity_percentage
                }
            )

        # Only the current paper exists
        if len(results) == 0:

            st.info(
                "📄 No other papers available for comparison."
            )

            return []

        results.sort(
            key=lambda x:
                x["similarity"],
            reverse=True
        )

        return results[:10]

    except Exception as e:

        st.error(
            f"Local similarity error: {e}"
        )

        return []


# ============================================================
# RECONSTRUCT OPENALEX ABSTRACT
# ============================================================

def reconstruct_abstract(
    inverted_index
):

    if not inverted_index:
        return ""

    words = []

    for word, positions in (
        inverted_index.items()
    ):

        for position in positions:

            words.append(
                (
                    position,
                    word
                )
            )

    words.sort(
        key=lambda x: x[0]
    )

    return " ".join(
        word
        for _, word in words
    )


# ============================================================
# OPENALEX SEARCH
# ============================================================

def search_openalex(
    query,
    max_results=15
):

    url = (
        "https://api.openalex.org/works"
    )

    params = {
        "search": query,
        "per-page": max_results,
        "select": (
            "id,"
            "display_name,"
            "publication_year,"
            "doi,"
            "authorships,"
            "abstract_inverted_index,"
            "type"
        )
    }

    headers = {
        "User-Agent":
            "AI-Research-Paper-Assistant/1.0"
    }

    if OPENALEX_EMAIL:

        headers["User-Agent"] += (
            f" mailto:{OPENALEX_EMAIL}"
        )

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=60
        )

        response.raise_for_status()

        return response.json().get(
            "results",
            []
        )

    except Exception as e:

        st.error(
            f"OpenAlex search error: {e}"
        )

        return []


# ============================================================
# INTERNET SIMILARITY
# ============================================================

def check_internet_similarity():

    if gemini_client is None:

        st.error(
            "Gemini API key is not configured."
        )

        return []

    text = (
        st.session_state.paper_text
    )

    if not text:
        return []

    beginning = text[:5000]

    lines = [
        line.strip()
        for line in beginning.splitlines()
        if line.strip()
    ]

    title = ""

    for line in lines[:20]:

        cleaned = re.sub(
            r"\s+",
            " ",
            line
        ).strip()

        if (
            15 <= len(cleaned) <= 250
            and not re.search(
                r"abstract|introduction|keywords|university|department|email",
                cleaned,
                re.IGNORECASE
            )
        ):

            title = cleaned
            break

    if not title:

        words = re.findall(
            r"\b[a-zA-Z]{4,}\b",
            beginning.lower()
        )

        title = " ".join(
            words[:8]
        )

    if not title:

        title = "research paper"

    st.info(
        f"OpenAlex search: {title}"
    )

    papers = search_openalex(
        title,
        15
    )

    if not papers:
        return []

    current_embedding = (
        create_paper_gemini_embedding(
            text
        )
    )

    if current_embedding is None:
        return []

    candidate_texts = []
    valid_papers = []

    for paper in papers:

        abstract = reconstruct_abstract(
            paper.get(
                "abstract_inverted_index"
            )
        )

        paper_title = paper.get(
            "display_name",
            ""
        )

        combined = (
            paper_title
            + "\n"
            + abstract
        )

        if len(combined.strip()) < 30:
            continue

        candidate_texts.append(
            combined
        )

        valid_papers.append(
            paper
        )

    if len(candidate_texts) == 0:

        return []

    with st.spinner(
        "Comparing with online research papers..."
    ):

        candidate_embeddings = (
            generate_gemini_embeddings(
                candidate_texts
            )
        )

    if candidate_embeddings is None:
        return []

    results = []

    for paper, vector in zip(
        valid_papers,
        candidate_embeddings
    ):

        current_vector = np.asarray(
            current_embedding,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )

        candidate_vector = np.asarray(
            vector,
            dtype=np.float32
        ).reshape(
            1,
            -1
        )

        if (
            current_vector.shape[1]
            != candidate_vector.shape[1]
        ):
            continue

        similarity = cosine_similarity(
            current_vector,
            candidate_vector
        )[0][0]

        percentage = max(
            0.0,
            min(
                100.0,
                float(similarity) * 100
            )
        )

        authors = []

        for author in paper.get(
            "authorships",
            []
        ):

            author_name = (
                author
                .get("author", {})
                .get("display_name")
            )

            if author_name:

                authors.append(
                    author_name
                )

        results.append(
            {
                "title":
                    paper.get(
                        "display_name",
                        "Untitled Research Paper"
                    ),

                "similarity":
                    percentage,

                "year":
                    paper.get(
                        "publication_year",
                        "Unknown"
                    ),

                "authors":
                    ", ".join(
                        authors[:5]
                    ),

                "doi":
                    paper.get("doi"),

                "openalex":
                    paper.get("id")
            }
        )

    results.sort(
        key=lambda x:
            x["similarity"],
        reverse=True
    )

    return results[:5]


# ============================================================
# TITLE
# ============================================================

st.markdown(
    """
    <div class="main-title">
        📚 AI Research Paper Assistant
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Read • Understand • Question • Compare • Research
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "### 📄 Upload Research Paper"
)

uploaded_file = st.file_uploader(
    "Upload a PDF research paper",
    type=["pdf"],
    help="Upload your research paper in PDF format."
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    st.success(
        "✅ PDF uploaded successfully!"
    )

    if st.button(
        "⚙️ Process Research Paper",
        use_container_width=True
    ):

        try:

            with st.spinner(
                "Reading and processing the research paper..."
            ):

                reader = PdfReader(
                    uploaded_file
                )

                # Actual PDF page count
                pages = len(
                    reader.pages
                )

                extracted_text = ""

                for page in reader.pages:

                    page_text = (
                        page.extract_text()
                    )

                    if page_text:

                        extracted_text += (
                            page_text
                            + "\n"
                        )

                extracted_text = (
                    extracted_text.strip()
                )

                if not extracted_text:

                    st.error(
                        "❌ No readable text was found in this PDF."
                    )

                    st.stop()

                chunks = create_chunks(
                    extracted_text,
                    1000,
                    200
                )

                if len(chunks) == 0:

                    st.error(
                        "❌ Could not create text chunks."
                    )

                    st.stop()

                embeddings = (
                    embedding_model.encode(
                        chunks,
                        show_progress_bar=False
                    )
                )

                ids = [
                    f"{uploaded_file.name}_{i}"
                    for i in range(
                        len(chunks)
                    )
                ]

                metadatas = [
                    {
                        "source":
                            uploaded_file.name,
                        "chunk":
                            i + 1
                    }
                    for i in range(
                        len(chunks)
                    )
                ]

                collection.upsert(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas
                )

                st.session_state.processed = True

                st.session_state.paper_name = (
                    uploaded_file.name
                )

                st.session_state.paper_text = (
                    extracted_text
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.page_count = (
                    pages
                )

                st.session_state.local_results = []

                st.session_state.internet_results = []

                st.session_state.internet_checked = False

            st.success(
                "🎉 Research paper processed successfully!"
            )

        except Exception as e:

            st.error(
                f"Error processing PDF: {e}"
            )


# ============================================================
# PROCESSED PAPER
# ============================================================

if st.session_state.processed:

    # ========================================================
    # PAPER INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 📊 Paper Processing Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Pages",
            st.session_state.page_count
        )

    with col2:

        st.metric(
            "Characters",
            f"{len(st.session_state.paper_text):,}"
        )

    with col3:

        st.metric(
            "Text Chunks",
            len(st.session_state.chunks)
        )


    # ========================================================
    # QUICK QUESTIONS
    # ========================================================

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### ⚡ Quick Questions"
    )

    st.markdown(
        '<div class="quick-info">Ask common questions about your research paper.</div>',
        unsafe_allow_html=True
    )

    quick_questions = [

        (
            "🎯 Objective",
            "What is the main objective or purpose of this research?",
            "objective"
        ),

        (
            "🧪 Methodology",
            "What methodology was used in this research?",
            "methodology"
        ),

        (
            "📊 Dataset",
            "What dataset or data was used in this research?",
            "dataset"
        ),

        (
            "📝 Conclusion",
            "What is the conclusion of this research?",
            "conclusion"
        ),

        (
            "📚 Summary",
            "Give me a concise overall summary of this research paper.",
            "summary"
        ),

        (
            "⚠️ Limitations",
            "What are the limitations and future work mentioned in this research?",
            "limitation"
        )
    ]

    quick_cols = st.columns(
        6,
        gap="small"
    )

    for i, (
        label,
        question,
        question_type
    ) in enumerate(
        quick_questions
    ):

        with quick_cols[i]:

            st.markdown(
                '<div class="quick-question">',
                unsafe_allow_html=True
            )

            clicked = st.button(
                label,
                key=f"quick_{i}",
                use_container_width=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

            if clicked:

                with st.spinner(
                    "Finding answer..."
                ):

                    answer = generate_answer(
                        question,
                        st.session_state.paper_name,
                        question_type,
                        5
                    )

                safe_answer = (
                    html.escape(
                        answer
                    )
                    .replace(
                        "\n",
                        "<br>"
                    )
                )

                st.markdown(
                    f"#### {label}"
                )

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <p>{safe_answer}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ========================================================
    # ASK OWN QUESTION
    # ========================================================

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 💬 Ask Your Own Question"
    )

    user_question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What problem does this research solve?"
        )
    )

    if st.button(
        "🤖 Ask AI",
        use_container_width=True
    ):

        if not user_question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Analyzing the research paper..."
            ):

                answer = generate_answer(
                    user_question,
                    st.session_state.paper_name,
                    "general",
                    5
                )

            safe_answer = (
                html.escape(
                    answer
                )
                .replace(
                    "\n",
                    "<br>"
                )
            )

            st.markdown(
                "#### 🤖 AI Answer"
            )

            st.markdown(
                f"""
                <div class="answer-card">
                    <p>{safe_answer}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


    # ========================================================
    # SIMILARITY CHECKER
    # ========================================================

    st.markdown(
        '<div class="section-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 🔍 Research Paper Similarity Checker"
    )

    st.markdown(
        """
        <div class="quick-info">
            Check semantic similarity with saved papers
            and papers available online.
        </div>
        """,
        unsafe_allow_html=True
    )

    similarity_col1, similarity_col2 = st.columns(2)

    with similarity_col1:

        if st.button(
            "💾 Add This Paper to Local Database",
            use_container_width=True
        ):

            add_paper_to_local_database()

    with similarity_col2:

        if st.button(
            "🔎 Check Local Similarity",
            use_container_width=True
        ):

            st.session_state.local_results = (
                check_local_similarity()
            )


    # ========================================================
    # LOCAL RESULTS
    # ========================================================

    if len(
        st.session_state.local_results
    ) > 0:

        st.markdown(
            "#### 🗂️ Local Similarity Results"
        )

        for i, result in enumerate(
            st.session_state.local_results,
            start=1
        ):

            similarity = result[
                "similarity"
            ]

            if similarity >= 85:

                status = "🔴 High"

            elif similarity >= 70:

                status = "🟠 Moderate"

            else:

                status = "🟢 Low"

            st.markdown(
                f"**{i}. {result['paper_name']}**"
            )

            st.write(
                f"**Similarity:** {similarity:.2f}%"
            )

            st.write(
                f"**Status:** {status}"
            )

            st.divider()


    # ========================================================
    # INTERNET SIMILARITY
    # ========================================================

    if st.button(
        "🌐 Check Internet Similarity",
        use_container_width=True
    ):

        st.session_state.internet_checked = True

        st.session_state.internet_results = (
            check_internet_similarity()
        )

    if st.session_state.internet_checked:

        results = (
            st.session_state.internet_results
        )

        if len(results) > 0:

            st.markdown(
                "### 🌐 Similar Research Papers Found"
            )

            for i, paper in enumerate(
                results,
                start=1
            ):

                title = paper.get(
                    "title",
                    "Untitled Research Paper"
                )

                similarity = paper.get(
                    "similarity",
                    0
                )

                year = paper.get(
                    "year",
                    "Unknown"
                )

                authors = paper.get(
                    "authors",
                    ""
                )

                doi = paper.get(
                    "doi"
                )

                openalex = paper.get(
                    "openalex"
                )

                if similarity >= 85:

                    status = "🔴 High"

                elif similarity >= 70:

                    status = "🟠 Moderate"

                else:

                    status = "🟢 Low"

                st.markdown(
                    f"**{i}. {title}**"
                )

                st.write(
                    f"**Similarity:** {similarity:.2f}%"
                )

                st.write(
                    f"**Status:** {status}"
                )

                st.write(
                    f"**Year:** {year}"
                )

                if authors:

                    st.write(
                        f"**Authors:** {authors}"
                    )

                if doi:

                    st.markdown(
                        f"🔗 [View Research Paper]({doi})"
                    )

                elif openalex:

                    st.markdown(
                        f"🔗 [View OpenAlex Record]({openalex})"
                    )

                st.divider()

        else:

            st.info(
                "No sufficiently similar research papers were found online."
            )

    st.markdown(
        """
        <div class="quick-info">
            ⚠️ Similarity percentages represent semantic similarity.
            They are not a definitive plagiarism or copyright verdict.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        📚 AI Research Paper Assistant
        <br>
        Powered by Streamlit • Sentence Transformers •
        ChromaDB • Ollama • Gemini • OpenAlex
    </div>
    """,
    unsafe_allow_html=True
)