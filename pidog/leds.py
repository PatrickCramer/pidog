import threading
import time
from collections import deque

try:
    from .rgb_strip import RGBStrip
except Exception:
    RGBStrip = None


class LedController:
    def __init__(self, light_num=11):
        self._lock = threading.Lock()
        self._state = "off"
        self._pulse_queue = deque()
        self._pulse_active = None
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

    def pulse(self, color, count=1, on_time=0.08, off_time=0.06):
        if self._strip is None:
            return
        count = int(count) if count is not None else 1
        if count <= 0:
            return
        with self._lock:
            self._pulse_queue.append(
                {
                    "color": color[:],
                    "count": count,
                    "on_time": max(0.02, float(on_time)),
                    "off_time": max(0.02, float(off_time)),
                }
            )

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

    def _run(self):
        step = 0
        while not self._stop.is_set():
            if self._strip is None:
                time.sleep(0.1)
                continue
            now = time.monotonic()
            with self._lock:
                if self._pulse_active is None and self._pulse_queue:
                    next_pulse = self._pulse_queue.popleft()
                    self._pulse_active = {
                        "color": next_pulse["color"],
                        "remaining": next_pulse["count"],
                        "on_time": next_pulse["on_time"],
                        "off_time": next_pulse["off_time"],
                        "phase": "on",
                        "next_time": now + next_pulse["on_time"],
                    }
                pulse = self._pulse_active
                state = self._state

            if pulse is not None:
                phase = pulse["phase"]
                if phase == "on":
                    self._solid(pulse["color"])
                    if now >= pulse["next_time"]:
                        pulse["remaining"] -= 1
                        pulse["phase"] = "off"
                        pulse["next_time"] = now + pulse["off_time"]
                else:
                    self._clear()
                    if now >= pulse["next_time"]:
                        if pulse["remaining"] > 0:
                            pulse["phase"] = "on"
                            pulse["next_time"] = now + pulse["on_time"]
                        else:
                            with self._lock:
                                self._pulse_active = None
                time.sleep(0.02)
                continue

            if state == "wake":
                self._solid([255, 0, 0])
            elif state == "listen":
                self._solid([0, 255, 0])
            elif state == "speak":
                self._sweep_blue(step)
                step += 1
            else:
                self._clear()

            time.sleep(0.05)
