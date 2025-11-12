# vectordb.py
import os
import re
import chromadb
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document
import fitz  # PyMuPDF

DATA_DIR = "./data"
CHROMA_DIR = "./chroma_db"
INIT_FLAG = os.path.join(CHROMA_DIR, ".initialized")

class VectorDB:
    """Chroma-backed vector store wrapper. Auto-ingests files from ./data on first run."""

    def __init__(self, collection_name: str = "rag_documents"):
        os.makedirs(CHROMA_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        # embedding model (SentenceTransformer)
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # use simple, valid collection name
        self.collection = self.client.get_or_create_collection(collection_name)
        print(f"✅ VectorDB initialized: {collection_name}")

        # On first run, load all files from DATA_DIR into the collection
        if not os.path.exists(INIT_FLAG):
            self._ingest_data_dir()
            # create init flag to avoid re-ingesting duplicates
            with open(INIT_FLAG, "w") as f:
                f.write("initialized")

    def _read_file(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        text = ""
        try:
            if ext == ".pdf":
                with fitz.open(file_path) as doc:
                    for p in doc:
                        text += p.get_text()
            elif ext in [".docx", ".doc"]:
                doc = Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + "\n"
            elif ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                # unsupported extension -> ignore
                text = ""
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            text = ""
        return text.strip()

    def chunk_text(self, text: str, chunk_size=600, overlap=100) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        return splitter.split_text(text)

    def _ingest_data_dir(self):
        """Read all files from DATA_DIR and add to collection (chunk & embed)."""
        if not os.path.isdir(DATA_DIR):
            print(f"⚠️ Data dir {DATA_DIR} not found; skipping auto-ingest.")
            return
        file_paths = []
        for fname in sorted(os.listdir(DATA_DIR)):
            full = os.path.join(DATA_DIR, fname)
            if os.path.isfile(full) and fname.lower().split(".")[-1] in ("txt", "md", "docx", "doc", "pdf"):
                file_paths.append(full)
        if not file_paths:
            print("⚠️ No supported files found in data/ to ingest.")
            return
        docs = [{"content": p} for p in file_paths]
        print(f"ℹ️ Auto-ingesting {len(docs)} files from {DATA_DIR} into Chroma ...")
        self.add_documents(docs)

    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        documents: list of dicts like {"content": "/path/to/file.txt"} OR {"content": "raw text ..."}
        This method reads files or accepts raw text strings.
        """
        all_texts, all_ids = [], []
        for i, doc in enumerate(documents):
            content = doc.get("content")
            text = ""
            if isinstance(content, str) and os.path.exists(content):
                text = self._read_file(content)
            elif isinstance(content, str):
                # treat as raw text
                text = content
            if not text or len(text.strip()) < 20:
                continue
            chunks = self.chunk_text(text)
            for j, chunk in enumerate(chunks):
                # create a sensible id to avoid collisions
                doc_id = f"doc{i}_{j}"
                all_texts.append(chunk)
                all_ids.append(doc_id)
        if not all_texts:
            print("⚠️ No text chunks to add.")
            return
        embeddings = self.embedding_model.encode(all_texts).tolist()
        # Chroma's collection.add expects ids, embeddings, documents (documents can be the text)
        self.collection.add(ids=all_ids, embeddings=embeddings, documents=all_texts)
        print(f"✅ Added {len(all_texts)} text chunks to VectorDB")

    def search(self, query: str, n_results: int = 3) -> Dict[str, Any]:
        """
        Return a dict {'documents': [chunk_str,...], 'distances': [..]}.
        If no results, returns {'documents': []}
        """
        q_emb = self.embedding_model.encode([query]).tolist()
        try:
            res = self.collection.query(query_embeddings=q_emb, n_results=n_results, include=["documents", "distances"])
        except TypeError:
            # fallback if include param differs by version
            res = self.collection.query(query_embeddings=q_emb, n_results=n_results)
        docs = []
        dists = []
        if res and res.get("documents"):
            docs = res["documents"][0]
            dists = res.get("distances", [[]])[0] if res.get("distances") else []
        return {"documents": docs, "distances": dists}

    # Helper: strict post-filtering to extract a single reaction line containing both reactants
    @staticmethod
    def extract_reaction_line(query: str, chunks: List[str]) -> str:
        """
        Given a query like "Zn + HCl" or "Zn HCl", attempt to find a short line
        from chunks that contains both reactants (case-insensitive).
        Returns the matched line (trimmed) or empty string.
        """
        if not query or not chunks:
            return ""
        # normalize reactant tokens (split on +, comma, whitespace)
        tokens = re.split(r"[\+\-,/]|and|\s+", query)
        tokens = [t.strip().replace("₂", "2").replace("₁", "1") for t in tokens if t.strip()]
        # keep only meaningful tokens (letters/numbers)
        tokens = [re.sub(r"[^A-Za-z0-9μ°₀₂₂\-_]+", "", t) for t in tokens]
        tokens = [t.lower() for t in tokens if t]
        if not tokens:
            return ""

        # look for a single short line in chunks that contains all tokens
        for chunk in chunks:
            # split chunk into lines and small sentences
            for line in re.split(r"\n+", chunk):
                low = line.lower()
                if all(tok in low for tok in tokens):
                    # return the line trimmed, but prefer short chemical-equation-like lines
                    line = line.strip()
                    # If the line is long, try to extract a substring with the equation arrow or plus
                    m = re.search(r"([^\n]{0,200}(?:→|->|=|=>|\+)[^\n]{0,200})", line)
                    if m:
                        return m.group(1).strip()
                    return line
        # fallback: search whole chunk text for inline equation-like fragments that contain tokens
        for chunk in chunks:
            # search for equation patterns
            candidates = re.findall(r"([A-Za-z0-9+\s\(\)\u2082\u2083\u2192\-\=]+)", chunk)
            for cand in candidates:
                low = cand.lower()
                if all(tok in low for tok in tokens):
                    return cand.strip()
        return ""
