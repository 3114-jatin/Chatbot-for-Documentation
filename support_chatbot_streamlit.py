"""
Support Documentation Chatbot (single-file Streamlit app)

Features
- Upload or point to a folder containing Word (.docx), PDF, Excel (.xlsx/.xls/.csv), CSV, TXT, Bash/.sh/.ps1 scripts
- Extracts text, splits into chunks, embeds using sentence-transformers, builds a FAISS vector index
- Interactive chat UI (Streamlit) with conversational memory in session_state
- Uses OpenAI's Chat Completions API to generate answers using retrieved context (you can switch to another LLM)
- Ability to persist and load the index (pickle)

Requirements (example)
- python >= 3.8
- pip install streamlit openai sentence-transformers faiss-cpu python-docx pdfplumber pandas openpyxl tiktoken nltk

Save this file as `support_chatbot_streamlit.py` and run:

    pip install -r requirements.txt
    export OPENAI_API_KEY="sk-..."   # or set in your environment on Windows
    streamlit run support_chatbot_streamlit.py

Notes
- This is a single-file starter. For production, split into modules and secure your API keys.
- The app defaults to using OpenAI. If you prefer to run local LLMs, adapt the `call_llm` function.

"""

import os
import io
import pickle
import tempfile
from typing import List, Tuple

import streamlit as st
import pandas as pd
import pdfplumber
import docx
import openai
from sentence_transformers import SentenceTransformer
import faiss
import nltk
from nltk.tokenize import sent_tokenize

# Ensure punkt is downloaded
nltk.download('punkt')
nltk.download("punkt_tab")

# -----------------------------
# Configuration / Helpers
# -----------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # sentence-transformers model
EMBED_DIM = 384  # embedding dim for all-MiniLM-L6-v2
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

# -----------------------------
# File text extraction helpers
# -----------------------------


def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    paras = [p.text for p in doc.paragraphs if p.text]
    return "\n".join(paras)


def extract_text_from_excel(file_bytes: bytes) -> str:
    # read all sheets and concatenate
    with io.BytesIO(file_bytes) as b:
        try:
            xls = pd.read_excel(b, sheet_name=None)
        except Exception:
            # maybe CSV
            b.seek(0)
            try:
                df = pd.read_csv(b)
                return df.to_csv(index=False)
            except Exception:
                return ""
    pieces = []
    for sheet_name, df in xls.items():
        pieces.append(f"Sheet: {sheet_name}\n")
        pieces.append(df.to_csv(index=False))
    return "\n".join(pieces)


def extract_text_from_csv(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as b:
        try:
            df = pd.read_csv(b)
            return df.to_csv(index=False)
        except Exception:
            b.seek(0)
            txt = b.read().decode('utf-8', errors='ignore')
            return txt


def extract_text_from_text(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode('utf-8')
    except Exception:
        try:
            return file_bytes.decode('latin-1')
        except Exception:
            return ''


def extract_text_from_file(filename: str, file_bytes: bytes) -> str:
    fname = filename.lower()
    if fname.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    if fname.endswith('.docx') or fname.endswith('.doc'):
        return extract_text_from_docx(file_bytes)
    if fname.endswith('.xlsx') or fname.endswith('.xls'):
        return extract_text_from_excel(file_bytes)
    if fname.endswith('.csv'):
        return extract_text_from_csv(file_bytes)
    if fname.endswith('.txt') or fname.endswith('.log') or fname.endswith('.ps1') or fname.endswith('.sh') or fname.endswith('.bash'):
        return extract_text_from_text(file_bytes)
    # fallback try
    return extract_text_from_text(file_bytes)

# -----------------------------
# Text chunking / splitting
# -----------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # split into sentences, then group until reach chunk_size
    sents = sent_tokenize(text)
    chunks = []
    cur = []
    cur_len = 0
    for s in sents:
        sl = len(s.split())
        if cur_len + sl > chunk_size and cur:
            chunks.append(' '.join(cur))
            # start new chunk with overlap
            if overlap > 0:
                # take last few sentences for overlap
                overlap_words = overlap
                # simplistic: keep last sentence(s)
                cur = cur[-3:] if len(cur) >= 3 else cur
                cur_len = sum(len(x.split()) for x in cur)
            else:
                cur = []
                cur_len = 0
        cur.append(s)
        cur_len += sl
    if cur:
        chunks.append(' '.join(cur))
    return chunks

# -----------------------------
# Embeddings and FAISS index
# -----------------------------
class DocIndex:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.metadatas = []  # parallel list storing (source_filename, chunk_text)

    def build(self, texts: List[Tuple[str, str]]):
        # texts: list of (source, chunk)
        corpus = [t[1] for t in texts]
        if not corpus:
            raise ValueError("No text to index")
        embeddings = self.embedder.encode(corpus, convert_to_numpy=True)
        dim = embeddings.shape[1]
        # create faiss index
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadatas = [{'source': t[0], 'text': t[1], 'id': i} for i, t in enumerate(texts)]

    def save(self, path: str):
        data = {
            'metadatas': self.metadatas,
        }
        # save index
        faiss.write_index(self.index, path + '.index')
        with open(path + '.meta.pkl', 'wb') as f:
            pickle.dump(data, f)

    def load(self, path: str):
        self.index = faiss.read_index(path + '.index')
        with open(path + '.meta.pkl', 'rb') as f:
            data = pickle.load(f)
        self.metadatas = data['metadatas']
        # init embedder
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def query(self, query_text: str, top_k: int = TOP_K):
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            m = self.metadatas[idx]
            results.append({'score': float(score), 'source': m['source'], 'text': m['text']})
        return results

# -----------------------------
# LLM call (OpenAI ChatCompletion)
# -----------------------------

def call_llm(system_prompt: str, user_prompt: str, api_key: str, conversation_history: List[Tuple[str, str]] = None):
    """
    Simple wrapper to call OpenAI Chat API. conversation_history is a list of (role, content)
    """
    openai.api_key = api_key
    messages = []
    messages.append({"role": "system", "content": system_prompt})
    if conversation_history:
        for role, content in conversation_history:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})
    try:
        resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=messages, temperature=0.2)
    except Exception:
        # fallback to gpt-4 if not available
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature=0.2)
    return resp['choices'][0]['message']['content']

# -----------------------------
# Streamlit UI
# -----------------------------

def init_session_state():
    if 'docs' not in st.session_state:
        st.session_state.docs = []  # list of dicts: {filename, text}
    if 'index_obj' not in st.session_state:
        st.session_state.index_obj = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []  # list of (role, text)
    if 'api_key' not in st.session_state:
        st.session_state.api_key = os.environ.get('OPENAI_API_KEY', '')


def sidebar_controls():
    st.sidebar.title('Controls')
    st.sidebar.markdown('Upload files or load existing index')
    uploaded = st.sidebar.file_uploader('Upload files (multiple)', accept_multiple_files=True)
    folder = st.sidebar.text_input('Or provide a local folder path (server-side)')
    if st.sidebar.button('Ingest uploaded files'):
        if uploaded:
            ingest_files(uploaded)
        else:
            st.sidebar.warning('Please select files to upload')
    if st.sidebar.button('Ingest folder (server)'):
        if folder and os.path.isdir(folder):
            files = []
            for root, _, filenames in os.walk(folder):
                for fn in filenames:
                    path = os.path.join(root, fn)
                    with open(path, 'rb') as f:
                        files.append(type('F', (), {'name': fn, 'read': lambda f=f: f.read()}))
            if files:
                ingest_files(files)
            else:
                st.sidebar.warning('No files found in folder')
        else:
            st.sidebar.error('Folder path invalid or not accessible')

    st.sidebar.markdown('---')
    st.sidebar.text_input('OpenAI API Key (optional)', key='api_key_input')
    if st.sidebar.button('Save API Key'):
        if st.session_state.get('api_key_input'):
            st.session_state.api_key = st.session_state.api_key_input
            st.sidebar.success('API key saved to session (not persistent)')
        else:
            st.sidebar.error('Enter an API key first')

    st.sidebar.markdown('Index actions')
    if st.sidebar.button('Save Index to disk'):
        if st.session_state.index_obj:
            out = st.text_input('Filename prefix (no extension)', value='support_index')
            try:
                st.session_state.index_obj.save(out)
                st.sidebar.success('Index saved as: {}.index + {}.meta.pkl'.format(out, out))
            except Exception as e:
                st.sidebar.error(f'Failed to save index: {e}')
        else:
            st.sidebar.error('No index to save')

    uploaded_index = st.sidebar.file_uploader('Load saved index (.zip not supported in this single-file app)')
    # Note: loading FAISS index from uploaded file isn't trivial in single-file Streamlit environment


def ingest_files(files):
    st.info('Starting ingestion...')
    docs = []
    for f in files:
        try:
            content = f.read()
            text = extract_text_from_file(f.name, content)
            if not text or text.strip() == '':
                st.warning(f'No text extracted from {f.name}')
                continue
            docs.append({'filename': f.name, 'text': text})
            st.write(f'Processed: {f.name} (chars: {len(text)})')
        except Exception as e:
            st.error(f'Failed to process {f.name}: {e}')
    if not docs:
        st.warning('No documents were ingested')
        return
    # Split and prepare for embedding
    pairs = []
    for d in docs:
        chunks = chunk_text(d['text'])
        for c in chunks:
            pairs.append((d['filename'], c))
    # build index
    idx = DocIndex()
    with st.spinner('Embedding and building index (this may take a while)...'):
        idx.build(pairs)
    st.session_state.index_obj = idx
    st.session_state.docs = docs
    st.success('Index built and stored in session')


def main():
    st.set_page_config(page_title='Support Docs Chatbot', layout='wide')
    st.title('Application Support Documentation — Chatbot')
    init_session_state()

    # sidebar
    sidebar_controls()

    left, right = st.columns([3, 2])

    with left:
        st.header('Chat')
        query = st.text_input('Ask a question about your documents', key='query_input')
        if st.button('Send') or (query and st.session_state.get('auto_ask')):
            if not st.session_state.index_obj:
                st.error('Please ingest documents first (use the sidebar)')
            else:
                api_key = st.session_state.api_key or os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    st.error('OpenAI API key required — either set OPENAI_API_KEY env var or paste in sidebar')
                else:
                    # Retrieve
                    results = st.session_state.index_obj.query(query, top_k=TOP_K)
                    context = '\n\n---\n\n'.join([f"Source: {r['source']}\n{r['text']}" for r in results])
                    system_prompt = (
                        "You are an expert support assistant. Use the provided context (from official docs and scripts) to answer the user's question. "
                        "If the context is insufficient, say so and provide next steps. Keep answers concise and provide steps, commands, or code snippets when helpful."
                    )
                    # Build conversation history for LLM
                    conv = st.session_state.conversation[-6:] if st.session_state.conversation else []
                    # append system context as a system message in the user prompt for clarity
                    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
                    try:
                        answer = call_llm(system_prompt, user_prompt, api_key, conversation_history=conv)
                    except Exception as e:
                        st.error(f'LLM call failed: {e}')
                        answer = ""
                    if answer:
                        st.session_state.conversation.append(("user", query))
                        st.session_state.conversation.append(("assistant", answer))
                        st.success('Answer generated — see below')
                        st.markdown('**Answer:**')
                        st.write(answer)
                        with st.expander('Sources used'):
                            for r in results:
                                st.write(f"**{r['source']}** — score: {r['score']:.4f}")
                                st.write(r['text'][:1000] + ('...' if len(r['text']) > 1000 else ''))

    with right:
        st.header('Conversation')
        if st.session_state.conversation:
            for role, text in st.session_state.conversation[::-1]:
                if role == 'user':
                    st.markdown(f"**You:** {text}")
                else:
                    st.markdown(f"**Assistant:** {text}")
        else:
            st.write('No conversation yet. Ingest documents and ask a question.')

        st.markdown('---')
        st.write('Session documents')
        if st.session_state.docs:
            for d in st.session_state.docs:
                st.write(f"{d['filename']} — {len(d['text'])} chars")
        else:
            st.write('No documents ingested yet')

    st.markdown('---')
    st.write('Tips:')
    st.write('- Upload multiple files (pdf, docx, xlsx, csv, txt, .sh, .ps1) then click "Ingest uploaded files" in the sidebar.')
    st.write('- Use the "Save Index" option in the sidebar to persist the FAISS index for later reuse (single-file saving only).')


if __name__ == '__main__':
    main()



