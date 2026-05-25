import os
import sys
import threading
import chromadb
import ollama
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from prompt import SYSTEM_PROMPT, build_rag_prompt, build_clarification_prompt, build_followup_prompt
from ui import (
    Spinner, success, info, warn, error, enable_windows_ansi,
    BOLD, DIM, CYAN, GREEN, YELLOW, RESET,
)

load_dotenv()
enable_windows_ansi()

# ─── Konfigurasi ────────────────────────────────────────────────
CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "documents"
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
LLM_MODEL        = os.getenv("LLM_MODEL",        "llama3.2")
OLLAMA_HOST      = os.getenv("OLLAMA_HOST",       "http://localhost:11434")
TOP_K            = 5
BGE_PREFIX       = "query: "
MAX_HISTORY      = 6
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ────────────────────────────────────────────────────────────────


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    print("=" * 60)
    print(f"  {BOLD}🎓 GURU AI — Asisten Belajar Pribadi{RESET}")
    print(f"  {DIM}Model : {LLM_MODEL}  |  RAG : BGE-M3 + ChromaDB{RESET}")
    print("=" * 60)
    print("  Ketik pertanyaan Anda, atau:")
    print(f"  {CYAN}/keluar{RESET} — exit  |  {CYAN}/clear{RESET} — bersihkan layar")
    print(f"  {CYAN}/list{RESET} — lihat file yang sudah diproses")
    print("=" * 60)
    print()


class GuruBot:
    def __init__(self):
        # ── Validasi Ollama (fail-fast jika tidak running) ───────
        self._check_ollama()

        # ── Connect ChromaDB ─────────────────────────────────────
        try:
            self.chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
            self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)
        except Exception as e:
            error(f"Gagal connect ChromaDB: {e}",
                  hint=f"Pastikan folder '{CHROMA_PATH}' ada dan tidak corrupt.")
            raise SystemExit(1)

        total = self.collection.count()
        if total == 0:
            warn("Database kosong! Jalankan screening.py terlebih dahulu.")
        else:
            success(f"{total} chunks siap dari database dokumen.")

        # ── Load embedding model dengan spinner ──────────────────
        device_label = DEVICE.upper()
        if DEVICE == "cuda":
            try:
                device_label = f"GPU ({torch.cuda.get_device_name(0)})"
            except Exception:
                pass

        with Spinner(f"Memuat embedding model BGE-M3 di {device_label} ..."):
            try:
                self.embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
                # fp16 di GPU → query embedding ~2x lebih cepat
                if DEVICE == "cuda":
                    self.embed_model = self.embed_model.half()
            except torch.cuda.OutOfMemoryError:
                error("CUDA out of memory.",
                      hint="Tutup aplikasi lain yang pakai GPU.")
                raise SystemExit(1)
            except Exception as e:
                error(f"Gagal load embedding model: {e}",
                      hint="Cek koneksi internet (model auto-download pertama kali).")
                raise SystemExit(1)
        success(f"Embedding model siap di {device_label}")

        # ── Warmup embedding di background (encode dummy text) ───
        # Encode pertama selalu lambat (kernel compilation, cache, dll).
        # Lakukan di background sambil user baca header — query pertama jadi instant.
        threading.Thread(target=self._warmup_embedding, daemon=True).start()

        # Ollama client persistent
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        self.history: list[dict] = []

    def _warmup_embedding(self):
        """Encode dummy text agar query pertama tidak terasa lambat."""
        try:
            with torch.inference_mode():
                self.embed_model.encode(
                    BGE_PREFIX + "warmup",
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception:
            pass  # silent fail — warmup hanya optimasi

    def _check_ollama(self):
        with Spinner(f"Mengecek Ollama di {OLLAMA_HOST} ..."):
            try:
                client = ollama.Client(host=OLLAMA_HOST)
                models = client.list()
                names  = [m.model for m in models.models]
            except Exception:
                error(f"Tidak bisa terhubung ke Ollama di {OLLAMA_HOST}.",
                      hint="Jalankan: ollama serve  (di terminal terpisah)")
                raise SystemExit(1)

        if not any(LLM_MODEL in n for n in names):
            error(f"Model '{LLM_MODEL}' tidak ditemukan di Ollama.",
                  hint=f"Jalankan: ollama pull {LLM_MODEL}")
            raise SystemExit(1)

        success(f"Ollama OK — model {LLM_MODEL} tersedia.")

    def _embed_query(self, query: str) -> list[float]:
        with torch.inference_mode():
            vec = self.embed_model.encode(
                BGE_PREFIX + query.lower().strip(),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vec.tolist()

    def _retrieve(self, query: str) -> list[dict]:
        """Ambil TOP_K chunk paling relevan dari ChromaDB."""
        total = self.collection.count()
        if total == 0:
            return []

        vec     = self._embed_query(query)
        results = self.collection.query(
            query_embeddings=[vec],
            n_results=min(TOP_K, total),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist < 0.75:
                chunks.append({
                    "text"   : doc,
                    "source" : meta.get("file", "unknown"),
                    "page"   : meta.get("page", "?"),
                    "score"  : round(1 - dist, 3),
                })
        return chunks

    def _ask_llm(self, prompt: str) -> str:
        """Kirim ke LLM dengan spinner sampai token pertama, lalu stream."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-MAX_HISTORY:],
            {"role": "user",   "content": prompt},
        ]

        # Spinner sampai token pertama muncul (LLM cold-start bisa lama)
        sp = Spinner("Guru AI sedang berpikir ...", color=YELLOW).start()
        full_response = ""
        first_token   = True

        try:
            stream = self.ollama_client.chat(
                model=LLM_MODEL, messages=messages, stream=True,
            )
            for chunk in stream:
                token = chunk.message.content
                if first_token and token:
                    sp.stop()
                    print(f"\n{GREEN}{BOLD}🤖 Guru AI:{RESET}\n")
                    first_token = False
                if token:
                    print(token, end="", flush=True)
                    full_response += token
        except Exception as e:
            sp.stop(success_=False)
            error(f"LLM error: {e}",
                  hint="Pastikan Ollama masih berjalan dan model tersedia.")
            return ""

        if first_token:  # tidak ada token sama sekali
            sp.stop(success_=False, final_msg="LLM tidak mengembalikan response.")
            return ""

        print("\n")
        return full_response

    def _list_files(self):
        """Tampilkan file yang ada di database."""
        try:
            all_meta = self.collection.get(include=["metadatas"])["metadatas"]
            files    = sorted({m["file"] for m in all_meta if "file" in m})
            if files:
                print(f"\n{CYAN}{BOLD}📂 File di database:{RESET}")
                for f in files:
                    print(f"   • {f}")
            else:
                warn("Belum ada file di database.")
            print()
        except Exception as e:
            error(f"Gagal ambil daftar file: {e}")

    def chat(self, question: str) -> str:
        """Proses satu pertanyaan: retrieve → build prompt → tanya LLM."""
        try:
            with Spinner("Mencari konteks di dokumen ..."):
                chunks = self._retrieve(question)
        except Exception as e:
            error(f"Gagal retrieve dari ChromaDB: {e}")
            return ""

        # Tampilkan sumber yang ditemukan (transparansi ke user)
        if chunks:
            sources = sorted({f"{c['source']} (hal. {c['page']})" for c in chunks})
            print(f"{DIM}📎 Konteks: {', '.join(sources)}{RESET}")

        if not chunks:
            prompt = build_clarification_prompt(question)
        elif self.history:
            prompt = build_followup_prompt(self.history, question, chunks)
        else:
            prompt = build_rag_prompt(question, chunks)

        response = self._ask_llm(prompt)

        if response:
            self.history.append({"role": "user",      "content": question})
            self.history.append({"role": "assistant", "content": response})

        return response

    def run(self):
        """Loop utama chatbot di terminal."""
        clear_screen()
        print_header()

        while True:
            try:
                user_input = input(f"{BOLD}📝 Anda:{RESET} ").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{CYAN}👋 Sampai jumpa!{RESET}\n")
                break

            if not user_input:
                continue

            cmd = user_input.lower()
            if cmd in ("/keluar", "/exit", "/quit"):
                print(f"\n{CYAN}👋 Sampai jumpa!{RESET}\n")
                break
            elif cmd == "/clear":
                clear_screen()
                print_header()
                self.history = []
                info("Riwayat chat dibersihkan.")
                continue
            elif cmd == "/list":
                self._list_files()
                continue

            self.chat(user_input)


def main():
    try:
        bot = GuruBot()
    except SystemExit:
        raise
    except Exception as e:
        error(f"Gagal inisialisasi bot: {type(e).__name__}: {e}")
        sys.exit(1)

    bot.run()


if __name__ == "__main__":
    main()
