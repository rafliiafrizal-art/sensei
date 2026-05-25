import re
import numpy as np
from sentence_transformers import SentenceTransformer


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def check_context_relevance(chunks: list[dict]) -> float:
    """
    Context Relevance: seberapa relevan chunk yang diambil terhadap pertanyaan.
    Menggunakan skor similarity yang sudah dihitung saat retrieval (1 - cosine distance).
    Skor tinggi = chunk yang diambil memang berkaitan dengan pertanyaan user.
    """
    if not chunks:
        return 0.0
    scores = [c.get("score", 0.0) for c in chunks]
    return round(float(np.mean(scores)), 3)


def check_answer_relevance(question: str, answer: str, model: SentenceTransformer) -> float:
    """
    Answer Relevance: seberapa relevan jawaban terhadap pertanyaan.
    Skor tinggi = jawaban benar-benar menjawab apa yang ditanya, tidak melenceng.
    """
    if not answer.strip() or not question.strip():
        return 0.0
    vecs = model.encode(
        [question, answer],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return round(_cosine(vecs[0], vecs[1]), 3)


def check_faithfulness(answer: str, chunks: list[dict], model: SentenceTransformer) -> float:
    """
    Faithfulness: seberapa banyak kalimat jawaban yang bisa diverifikasi dari konteks.
    Untuk setiap kalimat jawaban, cari similarity tertinggi ke semua konteks.
    Skor tinggi = jawaban berbasis dokumen, bukan karangan AI.
    """
    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?])\s+', answer)
        if len(s.strip()) > 20
    ]
    context_texts = [c["text"] for c in chunks]

    if not sentences or not context_texts:
        return 0.0

    # Embed semua sekaligus dalam satu batch → efisien di GPU
    all_texts = sentences + context_texts
    vecs = model.encode(
        all_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    sent_vecs = vecs[:len(sentences)]
    ctx_vecs  = vecs[len(sentences):]

    # Setiap kalimat jawaban → ambil similarity tertinggi ke konteks manapun
    scores = [max(_cosine(sv, cv) for cv in ctx_vecs) for sv in sent_vecs]
    return round(float(np.mean(scores)), 3)


def _label(score: float) -> str:
    if score >= 0.75:
        return "Tinggi"
    if score >= 0.50:
        return "Sedang"
    return "Rendah"


def display_qc(ctx_rel: float, ans_rel: float, faithful: float):
    """Tampilkan hasil evaluasi QC setelah jawaban AI."""
    print("  " + "-" * 44)
    print("  Evaluasi Kualitas Jawaban")
    print(f"  Context Relevance  : {ctx_rel:.2f}  [{_label(ctx_rel)}]")
    print(f"  Answer Relevance   : {ans_rel:.2f}  [{_label(ans_rel)}]")
    print(f"  Faithfulness       : {faithful:.2f}  [{_label(faithful)}]")
    print("  " + "-" * 44 + "\n")
