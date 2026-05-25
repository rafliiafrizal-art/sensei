#AI Asisten untuk membantu belajar user

##Tujuan project
AI asisten untuk saya belajar dari file yang saya masukan ke dalam document Di proses menggunakan Teknik RAG sehingga AI bisa menjawan dengan spesifik yang ada di file tersebut.

##Tools
-python
-ollama sudah terinstall dengan model llama 3.2 dan deepsek
-ChromaDB untuk vectori storage lokal
-Pdfplumber untuk parsing PDF
-Python-docx untuk parsing DOCX


##Alur utama
1.User masukan file ke folder document sehingga teknik RAG screaning file tersebut dari perintah file screaning.py
2.Sehingga AI bisa menjawab spesifik sesuai materi dengan menunjukan materi ini di file apa serta halaman berapa.

##Konversi penting
-Semua respons dalam bahasa indonesia
-Ollama endpoint http://localhost11434
-mode:llama3.2 atau deepsek yang ada di laptop saya.
-embeding model = BGE-M3
-ketika sudah ada file yang di proses RAG dan ingin proses RAG di folder document ketika ada file baru,file yang sudah di Proses RAG jangan di proses lagi sehingga vector database tidak penuh.
-Ketika edit code hanya sesuai perintah saja jangan menghapus code yang masih berfungsi hanya menghapus code yang sudah tidak terpakai.


##Environtment
OLLAMA_HOST=http://localhost:11434
LLM_MODEL=llama 3.2
EMBEDING_MODEL=BGE-M3

##Cara Run
Python bot.py
