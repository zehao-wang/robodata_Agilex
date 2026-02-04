"""Non-blocking keyboard listener for spacebar toggle."""

import sys
import threading


class KeyboardListener:
    """Listens for spacebar presses in a background thread.

    Uses platform-appropriate raw terminal input (no external dependencies).
    """

    def __init__(self):
        self._callbacks: list = []
        self._running = False
        self._thread = None

    def on_space(self, callback):
        """Register a callback to be called when space is pressed."""
        self._callbacks.append(callback)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listen(self):
        """Read single characters from stdin in raw mode."""
        if sys.platform == "win32":
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_unix(self):
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while self._running:
                ch = sys.stdin.read(1)
                if ch == " ":
                    for cb in self._callbacks:
                        cb()
                elif ch == "q":
                    # Allow 'q' to also trigger quit
                    for cb in self._callbacks:
                        cb()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _listen_windows(self):
        import msvcrt
        while self._running:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode("utf-8", errors="ignore")
                if ch == " ":
                    for cb in self._callbacks:
                        cb()
                elif ch == "q":
                    for cb in self._callbacks:
                        cb()
                    break
