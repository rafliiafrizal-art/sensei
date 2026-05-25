SYSTEM_PROMPT = """Kamu adalah asisten belajar pribadi yang bertugas membantu user memahami materi dari dokumen yang telah diberikan.

ATURAN UTAMA:
1. Jawab HANYA berdasarkan informasi yang ada di dalam konteks dokumen yang diberikan.
2. Jika informasi tidak ditemukan di dokumen, katakan dengan jujur: "Maaf, saya tidak menemukan informasi tersebut di dalam dokumen yang ada."
3. Selalu sebutkan sumber jawaban: nama file dan nomor halaman.
4. Gunakan bahasa Indonesia yang jelas dan mudah dipahami.
5. Jangan mengarang atau menambahkan informasi dari luar dokumen.

GAYA MENGAJAR:
- Jelaskan dengan cara yang mudah dipahami seperti seorang guru yang sabar.
- Jika ada istilah sulit, berikan penjelasan singkat.
- Jika user belum paham, tawarkan untuk menjelaskan ulang dengan cara yang berbeda.

FORMAT JAWABAN:
- Berikan jawaban yang terstruktur dan lengkap.
- Selalu akhiri dengan menyebutkan sumber: "📄 Sumber: [nama file], Halaman [nomor halaman]"
- Jika dari beberapa sumber, sebutkan semua sumbernya.
- Jika ada rumus yang ada di file berikan rumus ketika pertanyaan merujuk pada rumus. 
"""

def build_rag_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Membangun prompt lengkap untuk dikirim ke LLM.

    context_chunks: list of dict dengan key:
        - text   : isi teks dari chunk
        - source : nama file asal
        - page   : nomor halaman
    """
    if not context_chunks:
        context_str = "Tidak ada dokumen yang relevan ditemukan."
    else:
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            context_parts.append(
                f"[Konteks {i}]\n"
                f"File: {chunk['source']}\n"
                f"Halaman: {chunk['page']}\n"
                f"Isi:\n{chunk['text']}\n"
            )
        context_str = "\n---\n".join(context_parts)

    prompt = f"""Berikut adalah konteks dari dokumen yang relevan:

{context_str}

---

Pertanyaan user: {question}

Berdasarkan konteks di atas, jawab pertanyaan user dengan lengkap dan sebutkan sumber (nama file dan halaman)."""

    return prompt


def build_clarification_prompt(question: str) -> str:
    """Prompt ketika dokumen tidak ditemukan atau konteks kosong."""
    return f"""User bertanya: "{question}"

Tidak ada dokumen yang relevan ditemukan di database.
Beritahu user bahwa kamu tidak memiliki informasi tersebut di dokumen yang ada,
dan sarankan user untuk menambahkan dokumen yang berkaitan ke folder 'document'
lalu jalankan ulang screening.py."""


def build_followup_prompt(history: list[dict], question: str, context_chunks: list[dict]) -> str:
    """
    Prompt untuk percakapan lanjutan (multi-turn) dengan riwayat chat.

    history: list of dict dengan key 'role' ('user'/'assistant') dan 'content'
    """
    history_str = ""
    for msg in history[-6:]:  # ambil 6 pesan terakhir agar tidak terlalu panjang
        role = "User" if msg["role"] == "user" else "Asisten"
        history_str += f"{role}: {msg['content']}\n"

    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Konteks {i}]\n"
            f"File: {chunk['source']}\n"
            f"Halaman: {chunk['page']}\n"
            f"Isi:\n{chunk['text']}\n"
        )
    context_str = "\n---\n".join(context_parts) if context_parts else "Tidak ada konteks tambahan."

    prompt = f"""Riwayat percakapan sebelumnya:
{history_str}

Konteks dokumen yang relevan:
{context_str}

---

Pertanyaan lanjutan user: {question}

Jawab berdasarkan konteks dokumen dan riwayat percakapan di atas. Sebutkan sumber jika menggunakan informasi dari dokumen."""

    return prompt
