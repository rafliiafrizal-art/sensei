import os
import chromadb
import ollama
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from prompt import SYSTEM_PROMPT, build_rag_prompt, build_clarification_prompt, build_followup_prompt
from qc import check_context_relevance, check_answer_relevance, check_faithfulness, display_qc

load_dotenv()

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
    print("  🎓 GURU AI — Asisten Belajar Pribadi")
    print("  Model : llama3.2  |  RAG : BGE-M3 + ChromaDB")
    print("=" * 60)
    print("  Ketik pertanyaan Anda, atau:")
    print("  /keluar  — exit  |  /clear  — bersihkan layar")
    print("  /list   — lihat file yang sudah diproses")
    print("=" * 60)
    print()


class GuruBot:
    def __init__(self):
        # Pastikan Ollama bisa diakses
        self._check_ollama()

        # Load ChromaDB
        print("🔌 Menghubungkan ke database dokumen ...")
        self.chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma.get_or_create_collection(COLLECTION_NAME)
        total = self.collection.count()

        if total == 0:
            print("⚠️  Database kosong! Jalankan screening.py terlebih dahulu.")
        else:
            print(f"✅ {total} chunks siap dari database dokumen.")

        # Load embedding model dengan GPU jika tersedia
        print(f"📥 Memuat embedding model di {DEVICE.upper()} ...")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        if DEVICE == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print("✅ Siap!\n")

        # Cache ollama client — tidak buat ulang setiap pertanyaan
        self.ollama_client = ollama.Client(host=OLLAMA_HOST)
        self.history: list[dict] = []

    def _check_ollama(self):
        try:
            client = ollama.Client(host=OLLAMA_HOST)
            models = client.list()
            names  = [m.model for m in models.models]
            if not any(LLM_MODEL in n for n in names):
                print(f"⚠️  Model '{LLM_MODEL}' tidak ditemukan di Ollama.")
                print(f"   Jalankan: ollama pull {LLM_MODEL}")
            else:
                print(f"✅ Ollama: model {LLM_MODEL} tersedia.")
        except Exception:
            print("❌ Ollama tidak bisa diakses! Pastikan Ollama sudah berjalan.")
            print("   Jalankan Ollama terlebih dahulu, lalu coba lagi.")
            raise SystemExit(1)

    def _embed_query(self, query: str) -> list[float]:
        # Lowercase agar "flowchart" dan "Flowchart" menghasilkan embedding yang sama
        vec = self.embed_model.encode(
            BGE_PREFIX + query.lower().strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
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
        docs       = results["documents"][0]
        metadatas  = results["metadatas"][0]
        distances  = results["distances"][0]

        for doc, meta, dist in zip(docs, metadatas, distances):
            # Threshold 0.75 agar variasi huruf kecil/besar tetap tertangkap
            if dist < 0.75:
                chunks.append({
                    "text"   : doc,
                    "source" : meta.get("file", "unknown"),
                    "page"   : meta.get("page", "?"),
                    "score"  : round(1 - dist, 3),
                })

        return chunks

    def _ask_llm(self, prompt: str) -> str:
        """Kirim prompt ke llama3.2 via Ollama, streaming output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.history[-MAX_HISTORY:],
            {"role": "user",   "content": prompt},
        ]

        print("\n🤖 Guru AI:\n")
        full_response = ""

        # Streaming agar tidak terasa lambat
        stream = self.ollama_client.chat(
            model    = LLM_MODEL,
            messages = messages,
            stream   = True,
        )
        for chunk in stream:
            token = chunk.message.content
            print(token, end="", flush=True)
            full_response += token

        print("\n")
        return full_response

    def _list_files(self):
        """Tampilkan file yang ada di database."""
        try:
            all_meta = self.collection.get(include=["metadatas"])["metadatas"]
            files    = sorted({m["file"] for m in all_meta if "file" in m})
            if files:
                print("\n📂 File yang sudah diproses di database:")
                for f in files:
                    print(f"   • {f}")
            else:
                print("   (belum ada file)")
            print()
        except Exception as e:
            print(f"❌ Gagal ambil daftar file: {e}\n")

    def _run_qc(self, question: str, response: str, chunks: list[dict]):
        """Hitung dan tampilkan 3 metrik QC setelah jawaban LLM."""
        if not chunks:
            return
        ctx_rel  = check_context_relevance(chunks)
        ans_rel  = check_answer_relevance(question, response, self.embed_model)
        faithful = check_faithfulness(response, chunks, self.embed_model)
        display_qc(ctx_rel, ans_rel, faithful)

    def chat(self, question: str) -> str:
        """Proses satu pertanyaan: retrieve → build prompt → tanya LLM → QC."""
        chunks = self._retrieve(question)

        if not chunks:
            prompt   = build_clarification_prompt(question)
            response = self._ask_llm(prompt)
        elif self.history:
            prompt   = build_followup_prompt(self.history, question, chunks)
            response = self._ask_llm(prompt)
        else:
            prompt   = build_rag_prompt(question, chunks)
            response = self._ask_llm(prompt)

        # Simpan ke history
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": response})

        return response

    def run(self):
        """Loop utama chatbot di terminal."""
        clear_screen()
        print_header()

        while True:
            try:
                user_input = input("📝 Anda: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Sampai jumpa!\n")
                break

            if not user_input:
                continue

            # Perintah khusus
            if user_input.lower() in ("/keluar", "/exit", "/quit"):
                print("\n👋 Sampai jumpa!\n")
                break
            elif user_input.lower() == "/clear":
                clear_screen()
                print_header()
                self.history = []
                print("🗑️  Riwayat chat dibersihkan.\n")
                continue
            elif user_input.lower() == "/list":
                self._list_files()
                continue

            # Jawab pertanyaan
            self.chat(user_input)


def main():
    bot = GuruBot()
    bot.run()


if __name__ == "__main__":
    main()
