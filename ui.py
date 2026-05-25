"""UI helper kecil: Spinner, warna terminal, dan format pesan error/sukses.
Tidak butuh dependency tambahan — pakai ANSI escape codes biasa.
"""
import sys
import time
import itertools
import threading


# ─── Warna ANSI ───────────────────────────────────────────────────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[31m"
GREEN   = "\033[32m"
YELLOW  = "\033[33m"
BLUE    = "\033[34m"
CYAN    = "\033[36m"
GRAY    = "\033[90m"


def success(msg: str):
    print(f"{GREEN}✅ {msg}{RESET}")


def info(msg: str):
    print(f"{CYAN}ℹ️  {msg}{RESET}")


def warn(msg: str):
    print(f"{YELLOW}⚠️  {msg}{RESET}")


def error(msg: str, hint: str = ""):
    """Pesan error yang jelas, tidak menyembunyikan masalah ke user."""
    print(f"{RED}{BOLD}❌ ERROR:{RESET} {RED}{msg}{RESET}")
    if hint:
        print(f"   {DIM}💡 {hint}{RESET}")


class Spinner:
    """Spinner sederhana dengan threading, aman di Windows Terminal (cp65001).

    Pemakaian:
        with Spinner("Memuat model ..."):
            do_heavy_work()
        success("Model siap")

    Atau manual:
        sp = Spinner("Memproses").start()
        ...
        sp.stop()
    """

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message: str, color: str = CYAN):
        self.message     = message
        self.color       = color
        self._stop_event = threading.Event()
        self._thread     = None
        self._start_time = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Stop dengan status sesuai apakah ada exception atau tidak
        self.stop(success_=(exc_type is None))
        return False  # jangan suppress exception

    def start(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self):
        frames = itertools.cycle(self.FRAMES)
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            frame   = next(frames)
            sys.stdout.write(
                f"\r{self.color}{frame}{RESET} {self.message} "
                f"{DIM}({elapsed:.1f}s){RESET}   "
            )
            sys.stdout.flush()
            time.sleep(0.08)

    def update(self, new_message: str):
        """Ganti pesan spinner di tengah jalan."""
        self.message = new_message

    def stop(self, success_: bool = True, final_msg: str = None):
        """Hentikan spinner. Tampilkan checkmark / X + pesan akhir."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        # Bersihkan baris spinner
        sys.stdout.write("\r" + " " * (len(self.message) + 30) + "\r")
        sys.stdout.flush()
        if final_msg is not None:
            if success_:
                print(f"{GREEN}✅ {final_msg}{RESET}")
            else:
                print(f"{RED}❌ {final_msg}{RESET}")


def enable_windows_ansi():
    """Aktifkan ANSI escape di Windows console lama (cmd.exe).
    Windows Terminal & PowerShell modern sudah aktif by default."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        # STD_OUTPUT_HANDLE = -11, ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x4
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass
