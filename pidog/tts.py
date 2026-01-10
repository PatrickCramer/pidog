from robot_hat.tts import *

import threading
import queue
import numpy as np

try:
    import pyaudio
except ImportError:
    pyaudio = None

from pathlib import Path
from sunfounder_voice_assistant.tts import Piper, Espeak, Pico2Wave
from sunfounder_voice_assistant.tts import piper as piper_module
from piper import config as piper_config


def create_tts(engine="piper", model="en_US-amy-low", length_scale=None):
    if engine == "piper":
        base = Path.home() / ".cache" / "pidog" / "piper_models"
        base.mkdir(parents=True, exist_ok=True)
        piper_module.PIPER_MODEL_DIR = str(base)
        tts = Piper()
        if model:
            tts.set_model(model)
        tts._length_scale = length_scale
        return tts
    if engine == "espeak":
        return Espeak()
    if engine == "pico2wave":
        return Pico2Wave()
    raise ValueError(f"Unknown TTS engine: {engine}")


def _resolve_output_device(pa, device):
    if device is None:
        return None
    if isinstance(device, int):
        return device
    needle = str(device).lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxOutputChannels", 0) <= 0:
            continue
        name = str(info.get("name", "")).lower()
        if needle in name:
            return i
    return None


def _piper_stream_to_device(tts, text, output_device):
    if pyaudio is None:
        tts.say(text, stream=True)
        return
    if tts.piper is None:
        raise ValueError("Piper model not initialized. Call set_model first.")

    length_scale = getattr(tts, "_length_scale", None)
    syn_config = None
    if length_scale is not None:
        syn_config = piper_config.SynthesisConfig(length_scale=length_scale)

    pa = pyaudio.PyAudio()
    device_index = _resolve_output_device(pa, output_device)
    if device_index is None:
        device_info = pa.get_default_output_device_info()
    else:
        device_info = pa.get_device_info_by_index(device_index)
    target_rate = int(device_info.get("defaultSampleRate", tts.piper.config.sample_rate))

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=target_rate,
        output=True,
        output_device_index=device_index,
    )
    try:
        for chunk in tts.piper.synthesize(text, syn_config=syn_config):
            data = chunk.audio_int16_bytes
            if target_rate != tts.piper.config.sample_rate:
                samples = np.frombuffer(data, dtype=np.int16)
                if samples.size > 1:
                    src_rate = tts.piper.config.sample_rate
                    dst_len = int(samples.size * target_rate / src_rate)
                    x_old = np.linspace(0, 1, num=samples.size, endpoint=False)
                    x_new = np.linspace(0, 1, num=dst_len, endpoint=False)
                    samples = np.interp(x_new, x_old, samples).astype(np.int16)
                    data = samples.tobytes()
            stream.write(data)
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()


def speak_text(tts, text, output_device=None):
    if not text:
        return
    if isinstance(tts, Piper):
        try:
            _piper_stream_to_device(tts, text, output_device)
            return
        except OSError:
            pass
    tts.say(text)


class SpeechQueue:
    def __init__(self, tts, output_device=None):
        self._tts = tts
        self._output_device = output_device
        self._queue = queue.Queue()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()

    def say(self, text):
        if text:
            self._idle.clear()
            self._queue.put(text)

    def stop(self):
        self._stop.set()
        self._queue.put(None)
        self._thread.join(timeout=2)

    def wait_idle(self, timeout=None):
        self._idle.wait(timeout=timeout)

    def _worker(self):
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                return
            try:
                speak_text(self._tts, item, self._output_device)
            except Exception:
                pass
            if self._queue.empty():
                self._idle.set()
