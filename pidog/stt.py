from robot_hat.stt import *

import time
import threading
import queue
import os
from pathlib import Path
import numpy as np
import sounddevice as sd
import wave
from sunfounder_voice_assistant.stt import STT
from sunfounder_voice_assistant.stt import vosk as vosk_module


def _ensure_vosk_model_dir():
    base = Path.home() / ".cache" / "pidog" / "vosk_models"
    base.mkdir(parents=True, exist_ok=True)
    vosk_module.MODEL_BASE_PATH = str(base)


def create_stt(language="en-us", device=None, samplerate=None, wake_words=None):
    _ensure_vosk_model_dir()
    stt = STT(language=None, samplerate=samplerate, device=device)
    try:
        stt.update_model_list()
        large_name = None
        for name, lang in zip(stt.available_model_names, stt.available_languages):
            if lang == language and "small" not in name.lower():
                large_name = name
                break
        if large_name is None:
            base = Path(vosk_module.MODEL_BASE_PATH)
            prefix = f"vosk-model-{language}-"
            candidates = [
                p.name for p in base.iterdir()
                if p.is_dir() and p.name.startswith(prefix) and "small" not in p.name.lower()
            ]
            candidates.sort(reverse=True)
            if candidates:
                large_name = candidates[0]
        if large_name:
            idx = stt.available_languages.index(language)
            stt.available_model_names[idx] = large_name
        stt.set_language(language)
    except Exception:
        if language is not None:
            stt.set_language(language)
    if wake_words:
        if isinstance(wake_words, str):
            wake_words = [wake_words]
        stt.set_wake_words(wake_words)
    return stt


class PorcupineListener:
    def __init__(
        self,
        keyword="jarvis",
        model_path=None,
        sensitivity=0.5,
        device=None,
    ):
        try:
            import pvporcupine
        except Exception as exc:
            raise RuntimeError("pvporcupine is not installed") from exc
        access_key = os.environ.get("PICOVOICE_ACCESS_KEY", "").strip()
        if not access_key:
            raise RuntimeError("PICOVOICE_ACCESS_KEY is not set.")
        self.device = device if device is not None else sd.default.device
        self.keyword = keyword
        self.model_path = model_path
        self.sensitivity = sensitivity
        if model_path:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keyword_paths=[model_path],
                sensitivities=[sensitivity],
            )
        else:
            self.porcupine = pvporcupine.create(
                access_key=access_key,
                keywords=[keyword],
                sensitivities=[sensitivity],
            )
        self.sample_rate = self.porcupine.sample_rate
        self.frame_length = self.porcupine.frame_length
        device_info = sd.query_devices(self.device, "input")
        self.device_rate = int(device_info["default_samplerate"])
        self.stop_listening_event = threading.Event()


def _resample_audio_int16(audio, src_rate, dst_rate, dst_len):
    if src_rate == dst_rate:
        if audio.size == dst_len:
            return audio
        if audio.size > dst_len:
            return audio[:dst_len]
        pad = np.zeros(dst_len - audio.size, dtype=np.int16)
        return np.concatenate([audio, pad])
    audio_f = audio.astype(np.float32) / 32768.0
    resampled = _resample_audio(audio_f, src_rate, dst_rate)
    if resampled.size > dst_len:
        resampled = resampled[:dst_len]
    elif resampled.size < dst_len:
        pad = np.zeros(dst_len - resampled.size, dtype=np.float32)
        resampled = np.concatenate([resampled, pad])
    resampled = np.clip(resampled, -1.0, 1.0)
    return (resampled * 32767.0).astype(np.int16)


def create_porcupine_wakeword(
    keyword="jarvis",
    model_path=None,
    sensitivity=0.5,
    device=None,
):
    return PorcupineListener(
        keyword=keyword,
        model_path=model_path,
        sensitivity=sensitivity,
        device=device,
    )


class WhisperSTT:
    def __init__(self, model="base", language="en", samplerate=16000):
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(
                "Whisper is not installed. Install with: "
                "/home/pat/pidog/.venv/bin/python -m pip install openai-whisper"
            ) from exc
        self._whisper = whisper
        self.model_name = model
        self.language = language
        self.samplerate = samplerate
        self.model = whisper.load_model(model)

    def transcribe(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        result = self.model.transcribe(audio, language=self.language, fp16=False)
        return (result.get("text") or "").strip()


def create_whisper_stt(model="base", language="en", samplerate=16000):
    return WhisperSTT(model=model, language=language, samplerate=samplerate)


def _resample_audio(audio, src_rate, dst_rate):
    if src_rate == dst_rate:
        return audio
    duration = len(audio) / float(src_rate)
    dst_len = max(1, int(duration * dst_rate))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def listen_for_wake_word(stt, wake_words, device=None, break_event=None):
    if isinstance(wake_words, str):
        wake_words = [wake_words]
    wake_words = [w.lower() for w in wake_words]

    stt.stop_listening_event.clear()
    watcher = None
    if break_event is not None:
        def _watch_break():
            break_event.wait()
            stt.stop_listening_event.set()
        watcher = threading.Thread(target=_watch_break, daemon=True)
        watcher.start()
    for result in stt.listen(stream=True, device=device):
        if not result:
            continue
        partial = result.get("partial", "").lower()
        final = result.get("final", "").lower()
        for word in wake_words:
            if word and (word in partial or word in final):
                stt.stop_listening_event.set()
                return word
    return None


def listen_for_wake_word_porcupine(listener, break_event=None):
    listener.stop_listening_event.clear()
    watcher = None
    if break_event is not None:
        def _watch_break():
            break_event.wait()
            listener.stop_listening_event.set()
        watcher = threading.Thread(target=_watch_break, daemon=True)
        watcher.start()

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            return
        q.put(bytes(indata))

    blocksize = int(round(
        listener.frame_length * listener.device_rate / float(listener.sample_rate)
    ))
    with sd.RawInputStream(
        samplerate=listener.device_rate,
        blocksize=blocksize,
        device=listener.device,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            if listener.stop_listening_event.is_set():
                return None
            try:
                data = q.get(timeout=0.5)
            except queue.Empty:
                continue
            audio = np.frombuffer(data, dtype=np.int16)
            audio = _resample_audio_int16(
                audio,
                listener.device_rate,
                listener.sample_rate,
                listener.frame_length,
            )
            keyword_index = listener.porcupine.process(audio.tolist())
            if keyword_index >= 0:
                listener.stop_listening_event.set()
                return listener.keyword or "custom"


def record_utterance(stt, device=None, max_seconds=8, stream=True):
    if not stream:
        text = stt.listen(stream=False, device=device) or ""
        return {"text": text.strip(), "confidence": 1.0 if text else 0.0}

    last_partial = ""
    stop_timer = None

    if max_seconds is not None:
        def _stop_after_timeout():
            time.sleep(max_seconds)
            stt.stop_listening_event.set()

        stop_timer = threading.Thread(target=_stop_after_timeout, daemon=True)
        stop_timer.start()

    for result in stt.listen(stream=True, device=device):
        if not result:
            continue
        partial = result.get("partial", "").strip()
        if partial:
            last_partial = partial
        if result.get("done"):
            text = result.get("final", "").strip()
            return {"text": text, "confidence": 1.0 if text else 0.0}

    text = last_partial.strip()
    return {"text": text, "confidence": 1.0 if text else 0.0}


def record_utterance_whisper(
    stt,
    device=None,
    max_seconds=8,
    silence_threshold=0.01,
    silence_seconds=0.8,
    input_rate=None,
    input_dtype="int16",
):
    frames = []
    target_rate = getattr(stt, "samplerate", 16000)
    heard = False
    last_voice_time = None
    start = time.time()

    def _open_stream(rate, dtype):
        return sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype=dtype,
            device=device,
            blocksize=max(1, int(rate * 0.2)),
        )

    rates = [target_rate]
    if input_rate:
        rates.insert(0, int(input_rate))
    try:
        dev_info = sd.query_devices(device, "input")
        rates.append(int(dev_info.get("default_samplerate", target_rate)))
    except Exception:
        pass
    rates.extend([48000, 44100, 32000, 16000])
    seen = set()
    rates = [r for r in rates if not (r in seen or seen.add(r))]

    stream_ctx = None
    stream_dtype = None
    samplerate = None
    for rate in rates:
        dtypes = (input_dtype, "int16", "float32") if input_dtype else ("int16", "float32")
        for dtype in dtypes:
            try:
                sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype=dtype)
                stream_ctx = _open_stream(rate, dtype)
                stream_dtype = dtype
                samplerate = rate
                break
            except Exception:
                continue
        if stream_ctx is not None:
            break
    if stream_ctx is None:
        raise RuntimeError("Unable to open microphone input stream for Whisper.")

    blocksize = max(1, int(samplerate * 0.2))
    with stream_ctx as stream:
        while True:
            data, _ = stream.read(blocksize)
            if data.size:
                if stream_dtype == "int16":
                    data = data.astype(np.float32) / 32768.0
                frames.append(data.copy())
                rms = float(np.sqrt(np.mean(data**2)))
                now = time.time()
                if rms > silence_threshold:
                    heard = True
                    last_voice_time = now
                if heard and last_voice_time is not None and (now - last_voice_time) >= silence_seconds:
                    break
            if time.time() - start >= max_seconds:
                break

    if not frames:
        return {"text": "", "confidence": 0.0}

    audio = np.concatenate(frames, axis=0).flatten()
    if samplerate and samplerate != target_rate:
        audio = _resample_audio(audio, samplerate, target_rate)
    text = stt.transcribe(audio)
    return {"text": text, "confidence": 1.0 if text else 0.0}


def record_audio_wav(
    output_path,
    device=None,
    max_seconds=8,
    silence_threshold=0.01,
    silence_seconds=0.8,
    input_rate=44100,
    input_dtype="int16",
):
    frames = []
    blocksize = max(1, int(input_rate * 0.2))
    heard = False
    last_voice_time = None
    start = time.time()

    def _to_float(data):
        if input_dtype == "int16":
            return data.astype(np.float32) / 32768.0
        return data.astype(np.float32)

    with sd.InputStream(
        samplerate=input_rate,
        channels=1,
        dtype=input_dtype,
        device=device,
        blocksize=blocksize,
    ) as stream:
        while True:
            data, _ = stream.read(blocksize)
            if data.size:
                frames.append(data.copy())
                rms = float(np.sqrt(np.mean(_to_float(data) ** 2)))
                now = time.time()
                if rms > silence_threshold:
                    heard = True
                    last_voice_time = now
                if heard and last_voice_time is not None and (now - last_voice_time) >= silence_seconds:
                    break
            if time.time() - start >= max_seconds:
                break

    if not frames:
        return {"ok": False, "duration": 0.0, "rms": 0.0}

    audio = np.concatenate(frames, axis=0).flatten()
    if input_dtype != "int16":
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    else:
        audio = audio.astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(input_rate)
        wf.writeframes(audio.tobytes())
    float_audio = audio.astype(np.float32) / 32768.0
    rms = float(np.sqrt(np.mean(float_audio ** 2))) if float_audio.size else 0.0
    duration = float_audio.size / float(input_rate)
    return {"ok": True, "duration": duration, "rms": rms}
