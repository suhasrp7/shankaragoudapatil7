from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False

import numpy as np
from sentence_transformers import SentenceTransformer

from .pdf_utils import extract_text_chunks_from_pdf, TextChunk


@dataclass
class SearchResult:
    score: float
    text: str
    file_path: str
    page_number: int
    kind: str  # "book" | "paper"
    subject: Optional[str] = None


class SemanticSearchIndex:
    def __init__(self, storage_dir: str | Path, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "faiss.index"
        self.meta_path = self.storage_dir / "corpus.json"
        self.emb_path = self.storage_dir / "embeddings.npy"
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index = None  # faiss.IndexFlatIP or None
        self.corpus: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None  # used when FAISS is unavailable
        self.use_faiss = FAISS_AVAILABLE

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def _compute_embeddings(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self._normalize(embeddings)

    def _scan_directories(self, books_dir: Path, papers_dir: Path) -> List[TextChunk]:
        all_chunks: List[TextChunk] = []
        if books_dir.exists():
            for root, _, files in os.walk(books_dir):
                for f in files:
                    if f.lower().endswith(".pdf"):
                        pdf_path = Path(root) / f
                        all_chunks.extend(extract_text_chunks_from_pdf(pdf_path, kind="book"))
        if papers_dir.exists():
            for subject in sorted([d for d in papers_dir.iterdir() if d.is_dir()]):
                for root, _, files in os.walk(subject):
                    for f in files:
                        if f.lower().endswith(".pdf"):
                            pdf_path = Path(root) / f
                            all_chunks.extend(
                                extract_text_chunks_from_pdf(pdf_path, kind="paper", subject=subject.name)
                            )
        return all_chunks

    def _is_index_stale(self, books_dir: Path, papers_dir: Path) -> bool:
        if self.use_faiss:
            if not self.index_path.exists() or not self.meta_path.exists():
                return True
            index_mtime = max(self.index_path.stat().st_mtime, self.meta_path.stat().st_mtime)
        else:
            if not self.emb_path.exists() or not self.meta_path.exists():
                return True
            index_mtime = max(self.emb_path.stat().st_mtime, self.meta_path.stat().st_mtime)

        newest_pdf_mtime = 0.0
        for base in [books_dir, papers_dir]:
            if base.exists():
                for root, _, files in os.walk(base):
                    for f in files:
                        if f.lower().endswith(".pdf"):
                            p = Path(root) / f
                            newest_pdf_mtime = max(newest_pdf_mtime, p.stat().st_mtime)
        return newest_pdf_mtime > index_mtime

    def load_or_build(self, books_dir: str | Path, papers_dir: str | Path, force_rebuild: bool = False):
        books_dir = Path(books_dir)
        papers_dir = Path(papers_dir)
        if not force_rebuild and not self._is_index_stale(books_dir, papers_dir):
            self._load_from_disk()
            return
        self.rebuild(books_dir, papers_dir)

    def rebuild(self, books_dir: Path, papers_dir: Path):
        chunks = self._scan_directories(books_dir, papers_dir)
        self.corpus = [
            {
                "text": c.text,
                "file_path": str(c.file_path),
                "page": c.page_number,
                "kind": c.kind,
                "subject": c.subject,
            }
            for c in chunks
        ]
        texts = [c["text"] for c in self.corpus]

        if not texts:
            if self.use_faiss:
                self.index = None
            else:
                self.embeddings = None
            self._save_to_disk()
            return

        embeddings = self._compute_embeddings(texts)
        if self.use_faiss:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
            self.index.add(embeddings)
        else:
            self.embeddings = embeddings
        self._save_to_disk()

    def _save_to_disk(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f)
        if self.use_faiss:
            if self.index is not None and FAISS_AVAILABLE:
                faiss.write_index(self.index, str(self.index_path))  # type: ignore[attr-defined]
        else:
            if self.embeddings is not None:
                np.save(self.emb_path, self.embeddings)

    def _load_from_disk(self):
        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)
        if self.use_faiss and self.index_path.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(self.index_path))  # type: ignore[attr-defined]
            self.embeddings = None
        elif self.emb_path.exists():
            self.embeddings = np.load(self.emb_path)
            self.index = None

    def query(
        self,
        text: str,
        k: int = 10,
        kind_filter: Optional[str] = None,
        subject_filter: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        if self.use_faiss:
            if self.index is None:
                return []
            q = self._compute_embeddings([text])
            scores, indices = self.index.search(q, max(k * 5, 10))  # type: ignore[union-attr]
            candidates = list(zip(indices[0].tolist(), scores[0].tolist()))
        else:
            if self.embeddings is None:
                return []
            q = self._compute_embeddings([text])  # (1, d)
            sims = (q @ self.embeddings.T)[0]
            top_idx = np.argsort(-sims)[: max(k * 5, 10)]
            candidates = [(int(i), float(sims[int(i)])) for i in top_idx]

        results: List[SearchResult] = []
        for idx, score in candidates:
            if idx < 0 or idx >= len(self.corpus):
                continue
            item = self.corpus[idx]
            if kind_filter and item.get("kind") != kind_filter:
                continue
            if subject_filter and item.get("kind") == "paper" and item.get("subject") not in subject_filter:
                continue
            results.append(
                SearchResult(
                    score=float(score),
                    text=item["text"],
                    file_path=item["file_path"],
                    page_number=int(item["page"]),
                    kind=item.get("kind", "book"),
                    subject=item.get("subject"),
                )
            )
            if len(results) >= k:
                break
        return results