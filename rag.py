"""
PS36 - Research Paper Q&A
Core RAG pipeline:  PDF -> page-aware chunks -> Gemini embeddings -> FAISS -> grounded answer with [p. N] citations.

Deliberately free of Streamlit so it can be tested on its own.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Protocol, Sequence

import numpy as np
from pypdf import PdfReader

try:
    import faiss  # faiss-cpu
except ImportError:  # exact numpy search gives identical results at single-paper scale
    faiss = None

# --------------------------------------------------------------------------- config
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768           # truncated output -> vectors are re-normalised before indexing
EMBED_BATCH = 50
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 150
HISTORY_TURNS = 4
HISTORY_CHARS = 600

ProgressFn = Optional[Callable[[int, int], None]]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str], task_type: str, on_progress: ProgressFn = None) -> np.ndarray: ...


# --------------------------------------------------------------------------- prompts
SYSTEM_PROMPT = """You are a research assistant helping a student understand ONE research paper.

Rules:
1. Answer ONLY from the excerpts inside <context>. Do not use outside knowledge about the paper, its authors, or later work.
2. After every factual claim, cite the page it came from, e.g. [p. 4]. Use only page numbers that appear in the excerpt headers.
3. If the excerpts do not contain the answer, reply: "The paper excerpts I retrieved don't cover this." and suggest a more specific question. Never guess.
4. Copy numbers, metrics, dataset names and equations exactly as written in the excerpts.
5. Be concise: one short paragraph or at most 6 bullet points. Explain jargon in plain words the first time it appears.
6. Use <conversation_so_far> only to resolve follow-ups like "explain that more simply"; facts must still come from <context>."""

# Baseline kept selectable in the UI so the prompt improvement can be demonstrated side by side.
SYSTEM_PROMPT_V1 = "You are a helpful assistant. Answer the question about the research paper using the excerpts below."

PROMPTS = {
    "v2 - grounded + page citations": SYSTEM_PROMPT,
    "v1 - basic (baseline)": SYSTEM_PROMPT_V1,
}

# label -> (question, force-include abstract/intro)
PRESETS: dict[str, tuple[str, bool]] = {
    "📝 Summary": ("Give a structured summary of this paper: the problem, the proposed approach, "
                  "the key results, and the main takeaway.", True),
    "🎯 Contributions": ("What are the main contributions of this paper?", True),
    "⚙️ Method": ("Explain the proposed method or architecture step by step.", False),
    "📊 Results": ("Which datasets, baselines and evaluation metrics are used, and what are the headline results?", False),
    "⚠️ Limitations": ("What limitations does the paper acknowledge, and what future work do the authors suggest?", False),
}


# --------------------------------------------------------------------------- data
@dataclass(frozen=True)
class Chunk:
    text: str
    page: int   # 1-based
    idx: int    # position in reading order


@dataclass
class Paper:
    name: str
    title: str
    n_pages: int
    sha256: str
    chunks: list[Chunk]


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: Optional[float]  # cosine similarity; None for force-included intro chunks


# --------------------------------------------------------------------------- PDF -> chunks
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n\n")
_TITLE_NOISE = re.compile(r"^(arxiv|preprint|proceedings|published|journal|vol\.|doi|http|copyright|under review)", re.I)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join words hyphenated across lines
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # single line breaks -> spaces
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _units(text: str) -> Iterator[str]:
    """Sentence-ish units; hard-wraps anything longer than a chunk (tables, equations)."""
    for part in _SENT_SPLIT.split(text):
        part = (part or "").strip()
        while len(part) > CHUNK_CHARS:
            cut = part.rfind(" ", 0, CHUNK_CHARS)
            cut = cut if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
            yield part[:cut].strip()
            part = part[cut:].strip()
        if part:
            yield part


def split_page(text: str) -> list[str]:
    """Greedy sentence packing with sentence-level overlap. Chunks never cross pages, so citations stay exact."""
    chunks: list[str] = []
    cur: list[str] = []
    size = 0
    carried = 0  # how many leading units of `cur` are overlap from the previous chunk
    for unit in _units(text):
        if cur and size + len(unit) + 1 > CHUNK_CHARS:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            tail_size = 0
            for u in reversed(cur):
                if tail_size + len(u) + 1 > CHUNK_OVERLAP:
                    break
                tail.insert(0, u)
                tail_size += len(u) + 1
            cur, size, carried = tail, tail_size, len(tail)
        cur.append(unit)
        size += len(unit) + 1
    if cur:
        fresh = " ".join(cur[carried:])
        if chunks and len(fresh) < MIN_CHUNK_CHARS:
            chunks[-1] += " " + fresh
        else:
            chunks.append(" ".join(cur))
    return chunks


def _guess_title(reader: PdfReader, first_page_raw: str) -> str:
    try:
        meta = ((reader.metadata.title if reader.metadata else None) or "").strip()
    except Exception:
        meta = ""
    if 10 <= len(meta) <= 250 and not re.search(r"\.(docx?|tex|pdf)$|^untitled|^microsoft word", meta, re.I):
        return meta
    for line in first_page_raw.splitlines():
        line = line.strip()
        if len(line) >= 15 and not _TITLE_NOISE.match(line):
            return line[:200]
    return "Untitled paper"


def load_pdf(data: bytes, name: str) -> Paper:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = list(reader.pages)
    except Exception as e:
        raise ValueError(f"Could not read this PDF ({type(e).__name__}). Is it password-protected or corrupted?") from e

    chunks: list[Chunk] = []
    first_raw = ""
    for pno, page in enumerate(pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        if pno == 1:
            first_raw = raw
        for piece in split_page(_clean(raw)):
            chunks.append(Chunk(piece, pno, len(chunks)))

    if not chunks:
        raise ValueError("No extractable text found. This looks like a scanned/image-only PDF, which needs OCR first.")
    return Paper(name, _guess_title(reader, first_raw), len(pages), hashlib.sha256(data).hexdigest(), chunks)


# --------------------------------------------------------------------------- Gemini
_TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED", "timed out")


def _with_retry(call: Callable, attempts: int = 6):
    delay = 5.0
    for attempt in range(attempts):
        try:
            return call()
        except Exception as e:  # SDK raises google.genai.errors.APIError subclasses
            if attempt == attempts - 1 or not any(t in str(e) for t in _TRANSIENT):
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


class Gemini:
    """One API key for both embeddings and generation (google-genai SDK)."""

    def __init__(self, api_key: str):
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)

    def embed(self, texts: Sequence[str], task_type: str, on_progress: ProgressFn = None) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = list(texts[start:start + EMBED_BATCH])
            res = _with_retry(lambda b=batch: self._client.models.embed_content(
                model=EMBED_MODEL,
                contents=b,
                config=self._types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBED_DIM),
            ))
            vectors.extend(e.values for e in res.embeddings)
            if on_progress:
                on_progress(len(vectors), len(texts))
        return np.asarray(vectors, dtype=np.float32)

    def stream(self, system: str, prompt: str, model: str, temperature: float) -> Iterator[str]:
        config = self._types.GenerateContentConfig(system_instruction=system, temperature=temperature)
        for chunk in self._client.models.generate_content_stream(model=model, contents=prompt, config=config):
            if chunk.text:
                yield chunk.text


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "API_KEY_INVALID" in msg or "API key not valid" in msg:
        return "Invalid Gemini API key. Create one at https://aistudio.google.com/apikey and paste it in the sidebar or .env."
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Free-tier rate limit hit. Wait ~1 minute and retry, or lower top-k / switch to gemini-2.5-flash-lite."
    if "404" in msg or "NOT_FOUND" in msg:
        return "Model not found for this API key. Pick another model in the sidebar."
    if "PERMISSION_DENIED" in msg or "403" in msg:
        return "Permission denied by the Gemini API. Check the key's project has the Generative Language API enabled."
    return f"{type(e).__name__}: {msg[:300]}"


# --------------------------------------------------------------------------- vector index
def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norms, 1e-12, None)


class VectorIndex:
    """Cosine similarity = inner product on L2-normalised vectors."""

    def __init__(self, vectors: np.ndarray):
        if vectors.ndim != 2 or len(vectors) == 0:
            raise ValueError("Embeddings came back empty.")
        self._vecs = _normalize(vectors)
        self.size, self.dim = self._vecs.shape
        self._faiss = None
        if faiss is not None:
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._faiss.add(self._vecs)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        q = _normalize(query.reshape(1, -1))
        k = min(k, self.size)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        sims = self._vecs @ q[0]
        top = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in top]


# --------------------------------------------------------------------------- Q&A
class PaperQA:
    def __init__(self, paper: Paper, vectors: np.ndarray):
        self.paper = paper
        self.index = VectorIndex(vectors)

    @classmethod
    def build(cls, paper: Paper, embedder: Embedder, cache_dir: Optional[Path] = None,
              on_progress: ProgressFn = None) -> "PaperQA":
        """Embeds every chunk once; caches vectors on disk so restarts/re-uploads cost no API quota."""
        cache_file = None
        if cache_dir is not None:
            cache_file = Path(cache_dir) / f"{paper.sha256[:16]}_{EMBED_MODEL}_{EMBED_DIM}_{CHUNK_CHARS}_{CHUNK_OVERLAP}.npy"
            if cache_file.exists():
                vectors = np.load(cache_file)
                if len(vectors) == len(paper.chunks):
                    if on_progress:
                        on_progress(len(vectors), len(vectors))
                    return cls(paper, vectors)
        vectors = embedder.embed([c.text for c in paper.chunks], "RETRIEVAL_DOCUMENT", on_progress)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, vectors)
        return cls(paper, vectors)

    def retrieve(self, query: str, embedder: Embedder, k: int = 5, include_intro: bool = False) -> list[Hit]:
        qv = embedder.embed([query], "RETRIEVAL_QUERY")[0]
        hits = [Hit(self.paper.chunks[i], s) for i, s in self.index.search(qv, k)]
        if include_intro:  # abstract + intro rarely win similarity search for "summarise this"
            seen = {h.chunk.idx for h in hits}
            intro = [Hit(c, None) for c in self.paper.chunks[:2] if c.idx not in seen]
            hits = intro + hits
        return hits


def wants_overview(question: str) -> bool:
    return bool(re.search(r"summar|overview|abstract|main idea|tl;?dr|contribution|what is this paper", question, re.I))


def retrieval_query(question: str, history: Sequence[dict]) -> str:
    """Short follow-ups ("why?", "explain that simply") are searched together with the previous question."""
    if len(question.split()) <= 8:
        prev = next((m["content"] for m in reversed(history) if m["role"] == "user" and not m.get("error")), None)
        if prev:
            return f"{prev}\n{question}"
    return question


def build_prompt(title: str, hits: Sequence[Hit], question: str, history: Sequence[dict] = ()) -> str:
    excerpts = sorted(hits, key=lambda h: h.chunk.idx)  # reading order reads better than score order
    context = "\n\n".join(f"[Excerpt {n} | p. {h.chunk.page}]\n{h.chunk.text}" for n, h in enumerate(excerpts, 1))

    convo = ""
    turns = [m for m in history if not m.get("error")][-HISTORY_TURNS:]
    if turns:
        lines = []
        for m in turns:
            who = "Student" if m["role"] == "user" else "Assistant"
            text = m["content"] if len(m["content"]) <= HISTORY_CHARS else m["content"][:HISTORY_CHARS] + "…"
            lines.append(f"{who}: {text}")
        convo = "<conversation_so_far>\n" + "\n".join(lines) + "\n</conversation_so_far>\n\n"

    return f"Paper title: {title}\n\n<context>\n{context}\n</context>\n\n{convo}Question: {question}"
