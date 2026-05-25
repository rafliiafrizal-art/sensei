import os
import json
import hashlib
import re
import pdfplumber
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
import chromadb
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple

from ui import Spinner, success, info, warn, error, enable_windows_ansi, CYAN, DIM, RESET, BOLD

load_dotenv()
enable_windows_ansi()

# ─── Konfigurasi ────────────────────────────────────────────────
DOCUMENT_FOLDER   = "document"
PROCESSED_LOG     = ".processed_files.json"
CHROMA_PATH       = "./chroma_db"
COLLECTION_NAME   = "documents"
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

CHUNK_SIZE        = 600
CHUNK_OVERLAP     = 150
MIN_CHUNK_LEN     = 80
CHROMA_BATCH_SIZE = 200
BGE_PREFIX        = "passage: "

# GPU otomatis jika tersedia
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# Batch lebih besar di GPU karena bisa fp16 + paralel; di CPU tetap kecil
EMBED_BATCH_SIZE = 128 if DEVICE == "cuda" else 32
MAX_WORKERS      = 4   # thread paralel untuk ekstraksi file
# ────────────────────────────────────────────────────────────────


def _file_hash(path: str) -> str:
    """MD5 hash dari isi file — deteksi perubahan konten."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _clean_text(text: str) -> str:
    """Normalisasi whitespace dan karakter aneh."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'﻿|​|\xa0', ' ', text)
    return text.strip()


def extract_pdf(file_path: str) -> List[Tuple[str, int]]:
    """Extract teks PDF per halaman. Return: [(teks, page_num), ...]"""
    results = []
    try:
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                raw  = page.extract_text(layout=False) or ""
                text = _clean_text(raw)
                if text:
                    results.append((text, page_num))
            print(f"   {DIM}PDF: {len(results)}/{total} halaman berisi teks{RESET}")
    except Exception as e:
        error(f"Gagal baca PDF '{Path(file_path).name}': {e}",
              hint="Pastikan file tidak corrupt / terenkripsi.")
    return results


def extract_docx(file_path: str) -> List[Tuple[str, int]]:
    """Extract teks DOCX dengan deteksi page-break dari XML."""
    results = []
    try:
        doc      = Document(file_path)
        page_num = 1
        buffer: List[str] = []

        for para in doc.paragraphs:
            has_break = para._element.find(
                f'.//{qn("w:lastRenderedPageBreak")}'
            ) is not None or para._element.find(
                f'.//{qn("w:pageBreak")}'
            ) is not None

            if has_break and buffer:
                text = _clean_text("\n".join(buffer))
                if text:
                    results.append((text, page_num))
                buffer = []
                page_num += 1

            if para.text.strip():
                buffer.append(para.text)

        if buffer:
            text = _clean_text("\n".join(buffer))
            if text:
                results.append((text, page_num))

        print(f"   {DIM}DOCX: {page_num} halaman terdeteksi{RESET}")
    except Exception as e:
        error(f"Gagal baca DOCX '{Path(file_path).name}': {e}",
              hint="Pastikan file tidak corrupt / terproteksi password.")
    return results


def create_chunks(text: str, page_num: int) -> List[Dict]:
    """Chunking berbasis paragraf — tidak potong di tengah kalimat."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict] = []
    current = ""

    for para in paragraphs:
        if len(para) > CHUNK_SIZE:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= CHUNK_SIZE:
                    current = (current + " " + sent).strip()
                else:
                    if current and len(current) >= MIN_CHUNK_LEN:
                        chunks.append({"text": current, "page": page_num})
                    current = sent
        else:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = (current + "\n\n" + para).strip()
            else:
                if current and len(current) >= MIN_CHUNK_LEN:
                    chunks.append({"text": current, "page": page_num})
                overlap_text = current[-CHUNK_OVERLAP:] if current else ""
                current = (overlap_text + "\n\n" + para).strip()

    if current and len(current) >= MIN_CHUNK_LEN:
        chunks.append({"text": current, "page": page_num})

    return chunks


def _extract_worker(args: Tuple) -> Tuple[str, str, List[Dict]]:
    """Worker paralel: extract teks + chunking untuk satu file."""
    fp, name, fhash = args
    size_kb = os.path.getsize(str(fp)) / 1024
    print(f"\n📄 {BOLD}{name}{RESET}  {DIM}({size_kb:.1f} KB){RESET}")

    if name.lower().endswith(".pdf"):
        pages = extract_pdf(str(fp))
    elif name.lower().endswith(".docx"):
        pages = extract_docx(str(fp))
    else:
        warn(f"Format tidak didukung: {name}, skip.")
        return name, fhash, []

    chunks: List[Dict] = []
    for text, page_num in pages:
        chunks.extend(create_chunks(text, page_num))

    return name, fhash, chunks


class DocumentScreener:
    def __init__(self):
        # Connect ChromaDB dulu (cepat) — model di-load lazy nanti
        try:
            self.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
            self.col    = self.chroma.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            error(f"Gagal connect ke ChromaDB: {e}",
                  hint=f"Hapus folder '{CHROMA_PATH}' lalu coba lagi.")
            raise SystemExit(1)

        self.model: SentenceTransformer = None  # lazy load
        self.log   = self._load_log()

    # ── Lazy load embedding model ─────────────────────────────────
    def _ensure_model(self):
        """Load embedding model HANYA jika benar-benar dibutuhkan.
        Ini menghemat 30-60 detik saat 'semua file sudah up-to-date'."""
        if self.model is not None:
            return

        device_label = DEVICE.upper()
        if DEVICE == "cuda":
            try:
                device_label = f"GPU ({torch.cuda.get_device_name(0)})"
            except Exception:
                pass

        with Spinner(f"Memuat embedding model BGE-M3 di {device_label} ..."):
            try:
                self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
                # fp16 di GPU → ~2x lebih cepat, kualitas embedding praktis sama
                if DEVICE == "cuda":
                    self.model = self.model.half()
            except torch.cuda.OutOfMemoryError:
                error("CUDA out of memory saat load model.",
                      hint="Tutup aplikasi lain yang pakai GPU, atau set DEVICE=cpu.")
                raise SystemExit(1)
            except Exception as e:
                error(f"Gagal load embedding model: {e}",
                      hint="Cek koneksi internet — model di-download dari HuggingFace pertama kali.")
                raise SystemExit(1)
        success(f"Model siap di {device_label}")

    # ── Log helpers ──────────────────────────────────────────────
    def _load_log(self) -> Dict:
        if os.path.exists(PROCESSED_LOG):
            try:
                with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                warn(f"{PROCESSED_LOG} corrupt, mulai dari awal.")
        return {}

    def _save_log(self):
        with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=2)

    # ── Embedding ─────────────────────────────────────────────────
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed batch dengan prefix BGE-M3 + fp16 (di GPU)."""
        prefixed = [BGE_PREFIX + t for t in texts]
        # inference_mode mematikan autograd → lebih cepat & hemat memory
        with torch.inference_mode():
            vecs = self.model.encode(
                prefixed,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=EMBED_BATCH_SIZE,
                show_progress_bar=len(prefixed) > 50,
            )
        return vecs.tolist()

    # ── ChromaDB ──────────────────────────────────────────────────
    def _delete_old_chunks(self, filename: str):
        """Hapus chunks lama dari ChromaDB sebelum file di-reprocess.
        Tanpa ini, re-screening file akan menghasilkan duplikat atau ID collision."""
        try:
            self.col.delete(where={"file": filename})
        except Exception as e:
            warn(f"Tidak bisa hapus chunks lama untuk {filename}: {e}")

    def _store(self, chunks: List[Dict], embeddings: List[List[float]], filename: str):
        """Insert ke ChromaDB dalam batch."""
        for i in range(0, len(chunks), CHROMA_BATCH_SIZE):
            batch_chunks = chunks[i : i + CHROMA_BATCH_SIZE]
            batch_vecs   = embeddings[i : i + CHROMA_BATCH_SIZE]

            self.col.add(
                ids        = [f"{filename}_p{c['page']}_{i+j}" for j, c in enumerate(batch_chunks)],
                embeddings = batch_vecs,
                documents  = [c["text"] for c in batch_chunks],
                metadatas  = [{"file": filename, "page": str(c["page"])} for c in batch_chunks],
            )

    # ── Scan folder ───────────────────────────────────────────────
    def scan_and_process(self) -> Dict:
        folder = Path(DOCUMENT_FOLDER)
        folder.mkdir(exist_ok=True)

        files = sorted(
            f for ext in ("*.pdf", "*.docx")
            for f in folder.glob(ext)
        )

        if not files:
            warn(f"Tidak ada file PDF/DOCX di folder '{DOCUMENT_FOLDER}'.")
            return {"new": 0, "chunks": 0}

        info(f"{len(files)} file ditemukan di folder '{DOCUMENT_FOLDER}'")

        # Filter file baru / berubah — hash check pakai spinner agar terlihat aktif
        new_files: List[Tuple] = []
        reprocess_names: List[str] = []
        with Spinner("Mengecek perubahan file ..."):
            for fp in files:
                name  = fp.name
                fhash = _file_hash(str(fp))
                saved = self.log.get(name, {})
                if saved.get("hash") == fhash:
                    continue
                if name in self.log:
                    reprocess_names.append(name)
                new_files.append((fp, name, fhash))
        success(f"Pengecekan selesai: {len(new_files)} file baru/berubah, "
                f"{len(files) - len(new_files)} sudah up-to-date")

        for name in reprocess_names:
            info(f"File berubah, akan diproses ulang: {name}")

        if not new_files:
            success("Semua file sudah up-to-date — tidak ada yang perlu di-screening.")
            return {"new": 0, "chunks": 0}

        # Baru sekarang load model (lazy)
        self._ensure_model()

        # ── STEP 1: Ekstraksi paralel (I/O bound) ────────────────
        print(f"\n{CYAN}{BOLD}📂 Ekstraksi teks paralel ({MAX_WORKERS} thread) ...{RESET}")
        file_data: Dict[str, Tuple[str, List[Dict]]] = {}

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(new_files))) as ex:
            futures = {ex.submit(_extract_worker, f): f for f in new_files}
            for fut in as_completed(futures):
                try:
                    name, fhash, chunks = fut.result()
                    if chunks:
                        file_data[name] = (fhash, chunks)
                        print(f"   {DIM}→ {len(chunks)} chunks dari {name}{RESET}")
                    else:
                        warn(f"{name}: tidak ada chunk dihasilkan, skip.")
                except Exception as e:
                    error(f"Error ekstraksi: {e}")

        if not file_data:
            error("Tidak ada file yang berhasil diekstrak.")
            return {"new": 0, "chunks": 0}

        # ── STEP 2: Embed semua chunks sekaligus di GPU ──────────
        all_texts   : List[str]  = []
        chunk_owners: List[str]  = []
        chunk_objs  : List[Dict] = []

        for name, (_, chunks) in file_data.items():
            for chunk in chunks:
                all_texts.append(chunk["text"])
                chunk_owners.append(name)
                chunk_objs.append(chunk)

        print(f"\n{CYAN}{BOLD}🔄 Embedding {len(all_texts)} chunks di {DEVICE.upper()} ...{RESET}")
        try:
            all_embeddings = self._embed(all_texts)
        except torch.cuda.OutOfMemoryError:
            error("CUDA out of memory saat embedding.",
                  hint=f"Kurangi EMBED_BATCH_SIZE (sekarang {EMBED_BATCH_SIZE}) atau pakai CPU.")
            raise SystemExit(1)
        except Exception as e:
            error(f"Gagal embedding: {e}")
            raise SystemExit(1)

        # ── STEP 3: Store ke ChromaDB per file ───────────────────
        print(f"\n{CYAN}{BOLD}💾 Menyimpan ke ChromaDB ...{RESET}")
        file_chunks_map: Dict[str, Tuple[List[Dict], List]] = {}
        for chunk, emb, owner in zip(chunk_objs, all_embeddings, chunk_owners):
            if owner not in file_chunks_map:
                file_chunks_map[owner] = ([], [])
            file_chunks_map[owner][0].append(chunk)
            file_chunks_map[owner][1].append(emb)

        total_chunks = 0
        for name, (chunks, embeddings) in file_chunks_map.items():
            # BUG FIX: hapus chunks lama dulu jika ini re-process
            if name in self.log:
                self._delete_old_chunks(name)
            self._store(chunks, embeddings, name)
            fhash = file_data[name][0]
            self.log[name] = {"hash": fhash, "path": str(folder / name)}
            print(f"   {DIM}✓ {name}: {len(chunks)} chunks{RESET}")
            total_chunks += len(chunks)

        self._save_log()
        return {"new": len(file_data), "chunks": total_chunks}


def main():
    print("=" * 55)
    print(f"  {BOLD}🎯 DOCUMENT SCREENING & VECTORIZATION{RESET}")
    print("=" * 55)

    try:
        screener = DocumentScreener()
        stats    = screener.scan_and_process()
    except KeyboardInterrupt:
        print()
        warn("Dibatalkan oleh user (Ctrl+C).")
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as e:
        error(f"Crash tak terduga: {type(e).__name__}: {e}",
              hint="Cek log di atas untuk konteks.")
        raise SystemExit(1)

    print("\n" + "=" * 55)
    print(f"  {BOLD}📊 RINGKASAN{RESET}")
    print("=" * 55)
    print(f"  File diproses  : {stats['new']}")
    print(f"  Total chunks   : {stats['chunks']}")
    print(f"  Database       : {CHROMA_PATH}")
    print("=" * 55)
    success("Screening selesai! AI siap menjawab pertanyaan.\n")


if __name__ == "__main__":
    main()
