import socket
import threading


def _get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "localhost"
    finally:
        try:
            sock.close()
        except OSError:
            pass


class VideoManager:
    def __init__(self, port=9000, vflip=False, hflip=False, path="/mjpg"):
        self._lock = threading.Lock()
        self._running = False
        self._last_error = None
        self._port = port
        self._path = path
        self._vflip = vflip
        self._hflip = hflip
        self._vilib = None

    def start(self):
        with self._lock:
            if self._running:
                return self.get_url(), False, None
            try:
                from vilib import Vilib
            except Exception as exc:
                self._last_error = str(exc)
                return None, False, self._last_error
            try:
                Vilib.camera_start(vflip=self._vflip, hflip=self._hflip)
                Vilib.display(local=False, web=True)
            except Exception as exc:
                self._last_error = str(exc)
                return None, False, self._last_error
            self._vilib = Vilib
            self._running = True
            return self.get_url(), True, None

    def stop(self):
        with self._lock:
            if not self._running:
                return False
            try:
                if self._vilib is not None:
                    self._vilib.camera_close()
            except Exception as exc:
                self._last_error = str(exc)
                return False
            self._running = False
            return True

    def is_running(self):
        return self._running

    def get_url(self):
        ip = _get_local_ip()
        return f"http://{ip}:{self._port}{self._path}"

    def last_error(self):
        return self._last_error
