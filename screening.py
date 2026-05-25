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

load_dotenv()

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

# GPU otomatis jika RTX tersedia, CPU jika tidak
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_BATCH_SIZE = 64 if DEVICE == "cuda" else 32
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
    text = re.sub(r'\n{3,}', '\n\n', text)   # max 2 baris kosong
    text = re.sub(r'[ \t]+', ' ', text)       # spasi berulang
    text = re.sub(r'﻿|​|\xa0', ' ', text)  # hidden chars
    return text.strip()


def extract_pdf(file_path: str) -> List[Tuple[str, int]]:
    """
    Extract teks PDF per halaman.
    Return: [(teks_bersih, nomor_halaman), ...]
    """
    results = []
    try:
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, 1):
                raw = page.extract_text(layout=False) or ""
                text = _clean_text(raw)
                if text:
                    results.append((text, page_num))
            print(f"  ✅ PDF: {len(results)}/{total} halaman berisi teks")
    except Exception as e:
        print(f"  ❌ Gagal baca PDF: {e}")
    return results


def extract_docx(file_path: str) -> List[Tuple[str, int]]:
    """
    Extract teks DOCX dengan deteksi page-break dari XML.
    Return: [(teks_bersih, nomor_halaman), ...]
    """
    results = []
    try:
        doc = Document(file_path)
        page_num = 1
        buffer: List[str] = []

        for para in doc.paragraphs:
            # Cek page break di XML
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

        print(f"  ✅ DOCX: {page_num} halaman terdeteksi")
    except Exception as e:
        print(f"  ❌ Gagal baca DOCX: {e}")
    return results


def create_chunks(text: str, page_num: int) -> List[Dict]:
    """
    Chunking berbasis paragraf — tidak potong di tengah kalimat.
    Pisah dulu per paragraf, lalu gabungkan hingga CHUNK_SIZE.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[Dict] = []
    current = ""

    for para in paragraphs:
        # Jika paragraf sendiri > CHUNK_SIZE, pecah per kalimat
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
                # Overlap: ambil akhir chunk sebelumnya
                overlap_text = current[-CHUNK_OVERLAP:] if current else ""
                current = (overlap_text + "\n\n" + para).strip()

    if current and len(current) >= MIN_CHUNK_LEN:
        chunks.append({"text": current, "page": page_num})

    return chunks


def _extract_worker(args: Tuple) -> Tuple[str, str, List[Dict]]:
    """Worker paralel: extract teks + chunking untuk satu file."""
    fp, name, fhash = args
    size_kb = os.path.getsize(str(fp)) / 1024
    print(f"\n📄 {name}  ({size_kb:.1f} KB)")

    if name.lower().endswith(".pdf"):
        pages = extract_pdf(str(fp))
    elif name.lower().endswith(".docx"):
        pages = extract_docx(str(fp))
    else:
        print("  ⚠️  Format tidak didukung, skip.")
        return name, fhash, []

    chunks: List[Dict] = []
    for text, page_num in pages:
        chunks.extend(create_chunks(text, page_num))

    return name, fhash, chunks


class DocumentScreener:
    def __init__(self):
        self.chroma  = chromadb.PersistentClient(path=CHROMA_PATH)
        self.col     = self.chroma.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"📥 Memuat embedding model BGE-M3 di {DEVICE.upper()} ...")
        self.model   = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        if DEVICE == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("✅ Model siap!\n")
        self.log     = self._load_log()

    # ── Log helpers ──────────────────────────────────────────────
    def _load_log(self) -> Dict:
        if os.path.exists(PROCESSED_LOG):
            with open(PROCESSED_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_log(self):
        with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=2)

    # ── Embedding ─────────────────────────────────────────────────
    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Embed dengan prefix BGE-M3. Batching dihandle SentenceTransformer + GPU."""
        prefixed = [BGE_PREFIX + t for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=len(prefixed) > 50,
        )
        return vecs.tolist()

    # ── ChromaDB insert ───────────────────────────────────────────
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
            print("⚠️  Tidak ada file di folder document.")
            return {"new": 0, "chunks": 0}

        print(f"🔍 {len(files)} file ditemukan di folder document\n")

        # Filter file baru / berubah
        new_files: List[Tuple] = []
        for fp in files:
            name  = fp.name
            fhash = _file_hash(str(fp))
            saved = self.log.get(name, {})
            if saved.get("hash") == fhash:
                print(f"⏭️  Skip (tidak berubah): {name}")
            else:
                if name in self.log:
                    print(f"🔁 File berubah, proses ulang: {name}")
                new_files.append((fp, name, fhash))

        if not new_files:
            print("\n✅ Semua file sudah up-to-date.")
            return {"new": 0, "chunks": 0}

        print(f"\n⚡ {len(new_files)} file akan diproses ...\n")

        # ── STEP 1: Ekstraksi paralel (I/O bound → thread pool) ───
        print("📂 Ekstraksi teks secara paralel ...")
        file_data: Dict[str, Tuple[str, List[Dict]]] = {}  # name → (fhash, chunks)

        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(new_files))) as ex:
            futures = {ex.submit(_extract_worker, f): f for f in new_files}
            for fut in as_completed(futures):
                try:
                    name, fhash, chunks = fut.result()
                    if chunks:
                        file_data[name] = (fhash, chunks)
                        print(f"  📦 {name}: {len(chunks)} chunks")
                    else:
                        print(f"  ⚠️  {name}: tidak ada chunk, skip.")
                except Exception as e:
                    print(f"  ❌ Error ekstraksi: {e}")

        if not file_data:
            return {"new": 0, "chunks": 0}

        # ── STEP 2: Embed semua chunks sekaligus di GPU ────────────
        all_texts   : List[str]        = []
        chunk_owners: List[str]        = []  # nama file pemilik setiap chunk
        chunk_objs  : List[Dict]       = []

        for name, (_, chunks) in file_data.items():
            for chunk in chunks:
                all_texts.append(chunk["text"])
                chunk_owners.append(name)
                chunk_objs.append(chunk)

        print(f"\n🔄 Embedding {len(all_texts)} chunks di {DEVICE.upper()} ...")
        all_embeddings = self._embed(all_texts)

        # ── STEP 3: Store ke ChromaDB per file ────────────────────
        file_chunks_map: Dict[str, Tuple[List[Dict], List]] = {}
        for chunk, emb, owner in zip(chunk_objs, all_embeddings, chunk_owners):
            if owner not in file_chunks_map:
                file_chunks_map[owner] = ([], [])
            file_chunks_map[owner][0].append(chunk)
            file_chunks_map[owner][1].append(emb)

        total_chunks = 0
        for name, (chunks, embeddings) in file_chunks_map.items():
            self._store(chunks, embeddings, name)
            fhash = file_data[name][0]
            self.log[name] = {"hash": fhash, "path": str(DOCUMENT_FOLDER + "/" + name)}
            print(f"  ✅ {name}: {len(chunks)} chunks tersimpan")
            total_chunks += len(chunks)

        self._save_log()
        return {"new": len(file_data), "chunks": total_chunks}


def main():
    print("=" * 55)
    print("  🎯 DOCUMENT SCREENING & VECTORIZATION")
    print("=" * 55)

    screener = DocumentScreener()
    stats    = screener.scan_and_process()

    print("\n" + "=" * 55)
    print("  📊 RINGKASAN")
    print("=" * 55)
    print(f"  File diproses  : {stats['new']}")
    print(f"  Total chunks   : {stats['chunks']}")
    print(f"  Database       : {CHROMA_PATH}")
    print("=" * 55)
    print("  ✅ Screening selesai! AI siap menjawab pertanyaan.\n")


if __name__ == "__main__":
    main()
