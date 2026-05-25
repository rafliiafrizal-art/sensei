@echo off
title GURU AI - Asisten Belajar
cd /d "%~dp0"
chcp 65001 >nul

cls
echo.
echo  ==========================================
echo    GURU AI - Asisten Belajar Pribadi
echo    llama3.2  .  RAG  .  ChromaDB
echo  ==========================================
echo.

:: Cek Ollama
curl -s http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo  [!] Ollama belum berjalan. Menjalankan ...
    start "" ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo  [OK] Ollama siap
)

:: Cek database dan processed log
if not exist "chroma_db\" (
    echo.
    echo  [!] Database belum ada.
    echo      Jalankan "Screening Dokumen" terlebih dahulu.
    echo.
    echo  Tekan tombol apapun untuk menutup ...
    pause >nul
    exit /b
)

if not exist ".processed_files.json" (
    echo.
    echo  [!] Belum ada dokumen yang diproses.
    echo      Jalankan "Screening Dokumen" terlebih dahulu.
    echo.
    echo  Tekan tombol apapun untuk menutup ...
    pause >nul
    exit /b
)

echo  [OK] Database ditemukan
echo.
echo  Memulai sesi belajar ...
echo  ------------------------------------------
echo.

python bot.py

echo.
echo  ==========================================
echo  Sesi selesai. Tekan tombol apapun ...
echo  ==========================================
pause >nul
