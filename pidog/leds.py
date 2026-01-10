import threading
import time

try:
    from .rgb_strip import RGBStrip
except Exception:
    RGBStrip = None


class LedController:
    def __init__(self, light_num=11):
        self._lock = threading.Lock()
        self._state = "off"
        self._countdown_deadline = None
        self._countdown_duration = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._strip = None
        if RGBStrip is not None:
            try:
                self._strip = RGBStrip(0x74, light_num)
            except Exception:
                self._strip = None

    def start(self):
        if self._strip is None:
            return
        if not self._thread.is_alive():
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)
        self._clear()

    def set_state(self, state):
        with self._lock:
            self._state = state

    def set_countdown(self, deadline, duration):
        with self._lock:
            self._countdown_deadline = deadline
            self._countdown_duration = duration

    def clear_countdown(self):
        with self._lock:
            self._countdown_deadline = None
            self._countdown_duration = None

    def _clear(self):
        if self._strip is None:
            return
        data = [[0, 0, 0] for _ in range(self._strip.light_num)]
        self._strip.display(data)

    def _solid(self, color):
        data = [color[:] for _ in range(self._strip.light_num)]
        self._strip.display(data)

    def _sweep_blue(self, step):
        n = self._strip.light_num
        data = [[0, 0, 10] for _ in range(n)]
        pos = step % n
        data[pos] = [0, 0, 255]
        self._strip.display(data)

    def _countdown_green(self, ratio):
        n = self._strip.light_num
        lit = int(round(n * max(0.0, min(1.0, ratio))))
        data = [[0, 0, 0] for _ in range(n)]
        for i in range(lit):
            data[i] = [0, 255, 0]
        self._strip.display(data)

    def _run(self):
        step = 0
        while not self._stop.is_set():
            with self._lock:
                state = self._state
                deadline = self._countdown_deadline
                duration = self._countdown_duration
            if self._strip is None:
                time.sleep(0.1)
                continue

            if state == "wake":
                self._solid([255, 0, 0])
            elif state == "listen":
                if deadline is not None and duration:
                    remaining = deadline - time.time()
                    ratio = remaining / float(duration)
                    self._countdown_green(ratio)
                else:
                    self._solid([0, 255, 0])
            elif state == "speak":
                self._sweep_blue(step)
                step += 1
            else:
                self._clear()

            time.sleep(0.05)
