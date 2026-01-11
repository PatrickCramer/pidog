import argparse
import logging
from collections import deque
import atexit
import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
import re
import random
import subprocess
import sys
import threading
import time
import tempfile
import wave
import sounddevice as sd
import signal
import shutil
from pathlib import Path

from .stt import (
    create_stt,
    create_whisper_stt,
    listen_for_wake_word,
    listen_for_wake_word_picovoice,
    record_utterance,
    record_utterance_whisper,
    record_audio_wav,
)
from .leds import LedController
from .tts import create_tts, SpeechQueue
from .video import VideoManager
from .llm import stream_reply, openai_transcribe, openai_chat_reply, openai_tts_to_file
from .dual_touch import DualTouch, TouchStyle
from robot_hat import utils as rh_utils


USE_PICOVOICE_WAKE_WORD = True
PICOVOICE_KEYWORD = "jarvis"
PICOVOICE_SENSITIVITY = 0.5
BARGE_IN_ENABLED = True
BARGE_IN_KEYWORD = "jarvis"
BARGE_IN_SENSITIVITY = 0.4
RADIO_WAKE_ENABLED = True
RADIO_WAKE_KEYWORD = "jarvis"
RADIO_WAKE_SENSITIVITY = 0.5
SOFTMASTER_ENABLE = False
SOFTMASTER_CARD = "Device_1"
SOFTMASTER_STARTUP_PERCENT = 10
WAKE_WORDS = ["jarvis"]
WAKE_WORDS_STRICT = True
SOUND_DIR = Path(__file__).resolve().parents[1] / "sounds"
CONFUSED_SOUNDS = [
    "confused_1.mp3",
    "confused_2.mp3",
    "confused_3.mp3",
]
# Fallback index when named device is not found.
MIC_DEVICE = 2
MIC_DEVICE_NAME = "USB PnP Sound Device"
SPEAKER_DEVICE = None
STT_ENGINE = "vosk"  # "vosk" or "whisper"
STT_LANGUAGE = "en-us"
WHISPER_MODEL = "tiny"  # tiny/base/small/medium/large
WHISPER_LANGUAGE = "en"
TTS_ENGINE = "piper"
TTS_MODEL = "en_US-amy-low"
TTS_LENGTH_SCALE = 0.7
# LLM_MODEL = "smollm:360m"
LLM_MODEL = "llama3.2:1b"
CHATGPT_MODEL = "gpt-4o-mini"
CHATGPT_STT_MODEL = "gpt-4o-transcribe"
CHATGPT_STT_LANGUAGE = "en"
CHATGPT_TTS_MODEL = "gpt-4o-mini-tts"
CHATGPT_TTS_VOICE = "alloy"
CHATGPT_PRO_DEBUG_RECORDING = False

STT_USD_PER_MIN = 0.006   # gpt-4o-transcribe
TTS_USD_PER_MIN = 0.015   # gpt-4o-mini-tts
LLM_IN_USD_PER_1M = 0.15  # gpt-4o-mini input
LLM_OUT_USD_PER_1M = 0.60 # gpt-4o-mini output

MIC_SAMPLE_RATE = None  # use device default for Vosk
MIC_INPUT_RATE = 44100
MIC_INPUT_DTYPE = "int16"
CHATGPT_PRO_SILENCE_THRESHOLD = 0.012
CHATGPT_PRO_SILENCE_SECONDS = 0.65
CHATGPT_PRO_MIN_RMS = 0.008
CHATGPT_PRO_WAKE_MIN_RMS = 0.006

BATTERY_TABLE_2S = [
    (8.40, 100),
    (8.20, 90),
    (8.00, 80),
    (7.90, 70),
    (7.80, 60),
    (7.70, 50),
    (7.60, 40),
    (7.50, 30),
    (7.30, 20),
    (7.10, 10),
    (6.80, 5),
    (6.00, 0),
]

CONFIDENCE_THRESHOLD = 0.5
MAX_UTTERANCE_SECONDS = 6
SYSTEM_PROMPT = (
    "You are a helpful robot dog. Reply in English only, no more than four short phrases. "
    "Keep it friendly, funny and practical."
)

MEMORY_SENTENCES = 2
BBC_WORLD_SERVICE_URL = "http://stream.live.vc.bbcmedia.co.uk/bbc_world_service"
M3U_PATH = "/home/pat/radio_urls.m3u"
RADIO_URLS = {
    "1": "https://icecast.omroep.nl/radio1-bb-mp3",
    "2": "https://icecast.omroep.nl/radio2-bb-mp3",
    "3": "https://icecast.omroep.nl/3fm-bb-mp3",
}
RADIO_ALIASES = {
    "one": "1",
    "two": "2",
    "too": "2",
    "to": "2",
    "three": "3",
}
VOLUME_STEP = 5
TAP_WINDOW_SECONDS = 0.6
RESUME_TAP_WINDOW_SECONDS = 0.6
CONVO_WINDOW_SECONDS = 5
PIDFILE = "/tmp/pidog.pid"
STT_INIT_ASYNC = True
SKIP_WAKE_UNTIL_STT_READY = False
TAP_PULSE_COLOR = [0, 140, 255]
CAMERA_PORT = 9000
CAMERA_VFLIP = False
CAMERA_HFLIP = False
CAMERA_URL_PATH = "/mjpg"


LOGGER = logging.getLogger("pidog")


def _setup_logging():
    log_env = os.environ.get("PIDOG_LOG_FILE")
    if log_env:
        log_path = Path(log_env).expanduser()
    else:
        log_path = Path(__file__).resolve().parents[1] / "logs" / "pidog.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_path = Path("/tmp/pidog.log")
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    level_name = os.environ.get("PIDOG_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    LOGGER.setLevel(level)
    LOGGER.propagate = False
    if not LOGGER.handlers:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handlers_added = False
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            LOGGER.addHandler(file_handler)
            handlers_added = True
        except Exception:
            pass
        if os.environ.get("PIDOG_LOG_STDOUT", "").strip().lower() in ("1", "true", "yes", "on") or not handlers_added:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            LOGGER.addHandler(stream_handler)
    LOGGER.info("logging ready path=%s level=%s", log_path, logging.getLevelName(level))
    return log_path


def _touch_label(val):
    if val == TouchStyle.REAR:
        return "rear"
    if val == TouchStyle.FRONT:
        return "front"
    if val == TouchStyle.REAR_TO_FRONT:
        return "rear_to_front"
    if val == TouchStyle.FRONT_TO_REAR:
        return "front_to_rear"
    if val == TouchStyle.NONE:
        return "none"
    return str(val)


def log_action(action, **fields):
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    parts = [f"action={action}"]
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value!r}")
    LOGGER.info(" ".join(parts))


def _pulse_feedback(leds, count=1, context=None):
    if leds is None:
        return
    leds.pulse(TAP_PULSE_COLOR, count=count)
    log_action("led_pulse", count=count, context=context)


def _wait_for_speaker_or_interrupt(speaker, interrupt_event, poll=0.05):
    if speaker is None:
        return True
    while not speaker.is_idle():
        if interrupt_event.is_set():
            speaker.cancel()
            return False
        speaker.wait_idle(timeout=poll)
    return True


def _sync_radio_mode(radio_mode_event, radio_playing, radio_paused):
    if radio_playing or radio_paused:
        radio_mode_event.set()
    else:
        radio_mode_event.clear()


def _begin_answering(answering_event, interrupt_event, context=None):
    interrupt_event.clear()
    answering_event.set()
    log_action("answering", active=True, context=context)


def _end_answering(answering_event, context=None):
    answering_event.clear()
    log_action("answering", active=False, context=context)


def _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
    if not answer_interrupt_event.is_set():
        return False
    reason = getattr(answer_interrupt_event, "reason", "interrupt")
    answer_interrupt_event.clear()
    if hasattr(answer_interrupt_event, "reason"):
        answer_interrupt_event.reason = None
    if speaker is not None:
        speaker.cancel()
    if reason in ("voice", "tap"):
        _play_confused_sound()
    if leds is not None:
        leds.set_state("wake")
    log_action("answer_interrupted", reason=reason)
    if answering_event.is_set():
        _end_answering(answering_event, context="interrupt")
    return True


SENTENCE_END_RE = re.compile(r"[.!?]")

MISHEAR_MAP = {
    "by dog": "pidog",
    "pie dog": "pidog",
    "hi dog": "pidog",
    "my dog": "pidog",
    "radio one": "radio 1",
    "radio two": "radio 2",
    "radio too": "radio 2",
    "radio to": "radio 2",
    "radio three": "radio 3",
    "play ratio": "play radio",
    "play rado": "play radio",
    "play rodeo": "play radio",
}

SHORT_UTTERANCE_WORDS = {
    "the", "a", "an", "and", "uh", "um", "er", "hmm", "hey",
}

COMMON_ENGLISH_WORDS = {
    "a", "about", "after", "again", "all", "also", "and", "any", "are", "as", "at", "be",
    "because", "been", "before", "but", "by", "can", "could", "do", "does", "for", "from",
    "get", "go", "good", "have", "hello", "help", "here", "hi", "how", "i", "if", "in",
    "is", "it", "just", "like", "me", "my", "no", "not", "of", "ok", "okay", "on", "or",
    "please", "say", "see", "so", "some", "tell", "thanks", "that", "the", "this", "to",
    "up", "want", "we", "what", "when", "where", "who", "why", "yes", "you", "your",
}


def split_sentences(buffer):
    sentences = []
    while True:
        match = SENTENCE_END_RE.search(buffer)
        if not match:
            break
        end = match.end()
        sentence = buffer[:end].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[end:].lstrip()
    return sentences, buffer


def repair_transcript(text):
    cleaned = " ".join(text.strip().split())
    lowered = cleaned.lower()
    for wrong, right in MISHEAR_MAP.items():
        lowered = lowered.replace(wrong, right)
    return lowered


def normalize_radio_command(text):
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return None
    if not any(t in ("radio", "ratio", "rado", "rodeo") for t in tokens):
        return None
    for t in tokens:
        if t in ("1", "one", "won"):
            return "play radio 1"
        if t in ("2", "two", "to", "too"):
            return "play radio 2"
        if t in ("3", "three", "tree"):
            return "play radio 3"
    return None


def build_context(history):
    if not history:
        return ""
    return "Recent context:\n" + "\n".join(f"- {line}" for line in history) + "\n\n"


def sanitize_tts_text(text):
    return text.replace("*", "").strip()


def _clip_text(text, limit=120):
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[:limit - 3] + "..."


def is_likely_english(text):
    if not text:
        return False
    if any(ord(ch) > 127 for ch in text):
        return False
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in COMMON_ENGLISH_WORDS)
    ratio = hits / len(tokens)
    if len(tokens) <= 3:
        return True
    return ratio >= 0.1


def wav_seconds(path):
    try:
        with wave.open(path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        return 0.0


def _play_wav_interruptible(path, interrupt_event):
    proc = subprocess.Popen(["aplay", "-q", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while True:
        if interrupt_event.is_set():
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return False
        if proc.poll() is not None:
            return True
        time.sleep(0.05)


def _play_sound_effect(path):
    if not path:
        return False
    if path.endswith(".mp3"):
        if not shutil.which("mpg123"):
            return False
        subprocess.run(["mpg123", "-q", path], check=False)
        return True
    if path.endswith(".wav"):
        subprocess.run(["aplay", "-q", path], check=False)
        return True
    return False


def _play_confused_sound():
    choices = [str(SOUND_DIR / name) for name in CONFUSED_SOUNDS]
    choices = [path for path in choices if os.path.exists(path)]
    if not choices:
        return False
    return _play_sound_effect(random.choice(choices))


def _softmaster_exists():
    try:
        result = subprocess.run(
            ["amixer", "-c", SOFTMASTER_CARD, "scontrols"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return False
    return "SoftMaster" in result.stdout


def _set_softmaster(percent):
    if percent is None:
        return False
    try:
        subprocess.run(
            ["amixer", "-c", SOFTMASTER_CARD, "sset", "SoftMaster", f"{percent}%"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _prime_softmaster():
    if not SOFTMASTER_ENABLE:
        return
    for _ in range(6):
        if _softmaster_exists():
            break
        subprocess.run(
            ["aplay", "-D", "softvol", "-q", "-f", "S16_LE", "-c", "1", "-r", "44100", "-d", "1", "/dev/zero"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.1)
    if _set_softmaster(SOFTMASTER_STARTUP_PERCENT):
        log_action("softmaster_set", percent=SOFTMASTER_STARTUP_PERCENT)


def estimate_costs(audio_in_s, audio_out_s, in_tokens, out_tokens):
    stt = (audio_in_s / 60.0) * STT_USD_PER_MIN
    tts = (audio_out_s / 60.0) * TTS_USD_PER_MIN
    llm = 0.0
    if in_tokens is not None and out_tokens is not None:
        llm = (in_tokens / 1_000_000.0) * LLM_IN_USD_PER_1M
        llm += (out_tokens / 1_000_000.0) * LLM_OUT_USD_PER_1M
    return stt, tts, llm, (stt + tts + llm)


def estimate_battery_percent(voltage, table=BATTERY_TABLE_2S):
    if voltage >= table[0][0]:
        return 100
    if voltage <= table[-1][0]:
        return 0
    for (v1, p1), (v2, p2) in zip(table, table[1:]):
        if v1 >= voltage >= v2:
            t = (voltage - v2) / (v1 - v2)
            return int(round(p2 + t * (p1 - p2)))
    return 0


def get_battery_status():
    voltage = float(rh_utils.get_battery_voltage())
    percent = estimate_battery_percent(voltage)
    return voltage, percent


def _read_pidfile(path):
    try:
        data = Path(path).read_text(encoding="ascii", errors="ignore").strip()
    except Exception:
        return None
    if not data:
        return None
    try:
        return int(data)
    except ValueError:
        return None


def _write_pidfile(path, pid):
    try:
        Path(path).write_text(f"{pid}\n", encoding="ascii")
    except Exception:
        pass


def _cleanup_pidfile(path):
    try:
        if Path(path).exists():
            Path(path).unlink()
    except Exception:
        pass


def _pid_is_running(pid):
    if pid <= 0:
        return False
    return Path(f"/proc/{pid}").exists()


def _is_pidog_process(pid):
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_text(encoding="ascii", errors="ignore")
    except Exception:
        return False
    return "pidog" in cmdline or "pidog.main" in cmdline or "pidog/main.py" in cmdline


def _terminate_pid(pid, timeout=2.0):
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _pid_is_running(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _terminate_other_pidog_processes():
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        if not _is_pidog_process(pid):
            continue
        _terminate_pid(pid)


def _ensure_single_instance(pidfile):
    pid = _read_pidfile(pidfile)
    if pid and pid != os.getpid():
        if _pid_is_running(pid) and _is_pidog_process(pid):
            _terminate_pid(pid)
    _terminate_other_pidog_processes()
    _write_pidfile(pidfile, os.getpid())
    atexit.register(_cleanup_pidfile, pidfile)


def _check_openai_available():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return False, "missing_api_key"
    try:
        import openai  # noqa: F401
    except ModuleNotFoundError:
        return False, "missing_openai_package"
    return True, None


def _init_stt_background(
    stt_ready,
    stt_error,
    stt_state,
    device,
    samplerate,
):
    try:
        log_action("stt_init_start", engine=STT_ENGINE, device=device, samplerate=samplerate)
        t0 = time.time()
        wake_stt = create_stt(language=STT_LANGUAGE, device=device, samplerate=samplerate)
        stt_model = None
        try:
            stt_model = wake_stt.get_model_name(wake_stt.language())
        except Exception:
            pass
        stt_state["init_stt_seconds"] = time.time() - t0
        asr_stt = wake_stt
        if STT_ENGINE == "whisper":
            t_whisper = time.time()
            asr_stt = create_whisper_stt(
                model=WHISPER_MODEL,
                language=WHISPER_LANGUAGE,
                samplerate=samplerate,
            )
            stt_state["init_whisper_seconds"] = time.time() - t_whisper
        stt_state["wake_stt"] = wake_stt
        stt_state["asr_stt"] = asr_stt
        stt_state["stt_model"] = stt_model
        log_action(
            "stt_init_ready",
            engine=STT_ENGINE,
            stt_model=stt_model,
            init_stt_seconds=stt_state.get("init_stt_seconds"),
            init_whisper_seconds=stt_state.get("init_whisper_seconds"),
        )
        print("STT: Vosk ready.")
    except Exception as exc:
        stt_error[0] = exc
        LOGGER.exception("stt_init_failed")
    finally:
        stt_ready.set()


def _start_stt_init(
    stt_ready,
    stt_error,
    stt_state,
    device,
    samplerate,
    async_mode,
):
    if stt_state.get("started"):
        return
    stt_state["started"] = True
    if async_mode:
        threading.Thread(
            target=_init_stt_background,
            args=(stt_ready, stt_error, stt_state, device, samplerate),
            daemon=True,
        ).start()
        log_action("stt_init", mode="async", device=device, samplerate=samplerate)
        print("STT: init in background.")
    else:
        _init_stt_background(stt_ready, stt_error, stt_state, device, samplerate)
        _await_stt(stt_ready, stt_error, stt_state)
        log_action("stt_init", mode="sync", device=device, samplerate=samplerate)


def _ensure_local_stt_ready(
    stt_ready,
    stt_error,
    stt_state,
    device,
    samplerate,
    async_mode=False,
):
    _start_stt_init(stt_ready, stt_error, stt_state, device, samplerate, async_mode)
    return _await_stt(stt_ready, stt_error, stt_state)


def _await_stt(stt_ready, stt_error, stt_state):
    if not stt_ready.is_set():
        print("STT: waiting for background init...")
        log_action("stt_waiting")
        stt_ready.wait()
    if stt_error[0] is not None:
        LOGGER.error("stt_init_error: %s", stt_error[0])
        raise stt_error[0]
    if not stt_state.get("printed"):
        init_stt = stt_state.get("init_stt_seconds")
        if init_stt is not None:
            print(f"Init: STT {init_stt:.2f}s")
        if STT_ENGINE == "whisper":
            suffix = f" (wake word: Vosk {stt_state.get('stt_model')})" if stt_state.get("stt_model") else " (wake word: Vosk)"
            print(f"STT: Whisper {WHISPER_MODEL}{suffix}")
            init_whisper = stt_state.get("init_whisper_seconds")
            if init_whisper is not None:
                print(f"Init: Whisper {init_whisper:.2f}s")
        else:
            model = stt_state.get("stt_model")
            print(f"STT: Vosk {model}" if model else "STT: Vosk")
        log_action(
            "stt_ready",
            engine=STT_ENGINE,
            stt_model=stt_state.get("stt_model"),
            whisper_model=WHISPER_MODEL if STT_ENGINE == "whisper" else None,
        )
        stt_state["printed"] = True
    return stt_state["wake_stt"], stt_state["asr_stt"], stt_state["stt_model"]


def load_m3u_stations(path):
    stations = []
    if not os.path.exists(path):
        return stations
    name = None
    with open(path, "r", encoding="ascii", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#EXTINF"):
                if "," in line:
                    name = line.split(",", 1)[1].strip()
                else:
                    name = None
                continue
            if line.startswith("#"):
                continue
            url = line
            if name is None:
                name = url
            stations.append({"name": name, "url": url})
            name = None
    return stations


def find_station_index(stations, url):
    for i, station in enumerate(stations):
        if station["url"] == url:
            return i
    return None


def get_default_station(stations):
    if stations:
        return stations[0]
    return {"name": "BBC World Service", "url": BBC_WORLD_SERVICE_URL}


def start_moc_stream(url):
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

    def _ensure_moc_server():
        if os.geteuid() == 0:
            cmd = [
                "sudo",
                "-u",
                "pat",
                "-H",
                "env",
                f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
                "mocp",
                "-S",
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return
        subprocess.run(["mocp", "-S"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    if os.geteuid() == 0:
        _ensure_moc_server()
        cmd = [
            "sudo",
            "-u",
            "pat",
            "-H",
            "env",
            f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
            "mocp",
            "-l",
            url,
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    _ensure_moc_server()
    subprocess.Popen(["mocp", "-l", url], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_moc_stream():
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    if os.geteuid() == 0:
        cmd = [
            "sudo",
            "-u",
            "pat",
            "-H",
            "env",
            f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
            "mocp",
            "-s",
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    subprocess.Popen(["mocp", "-s"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def adjust_moc_volume(delta):
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    op = f"{delta:+d}"
    if os.geteuid() == 0:
        cmd = [
            "sudo",
            "-u",
            "pat",
            "-H",
            "env",
            f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
            "mocp",
            "-v",
            op,
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    subprocess.Popen(["mocp", "-v", op], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_pet(touch):
    while True:
        val = touch.read()
        if val != TouchStyle.NONE:
            return val
    time.sleep(0.05)


def touch_resume_watcher(touch, leds, paused_event, resume_event, stop_event, resume_ready_time):
    last_val = TouchStyle.NONE
    last_tap_time = 0.0
    while not stop_event.is_set():
        if not paused_event.is_set():
            time.sleep(0.05)
            continue
        val = touch.read()
        if val in (TouchStyle.REAR, TouchStyle.FRONT) and last_val == TouchStyle.NONE:
            now = time.time()
            _pulse_feedback(leds, count=1, context="resume_tap")
            log_action("touch_tap", zone=_touch_label(val), context="resume")
            if now >= resume_ready_time[0]:
                if now - last_tap_time <= RESUME_TAP_WINDOW_SECONDS:
                    resume_event.set()
                    _pulse_feedback(leds, count=2, context="resume_double_tap")
                    log_action("touch_double_tap", context="resume")
                last_tap_time = now
        last_val = val
        time.sleep(0.05)


def touch_start_watcher(touch, leds, idle_event, start_event, interrupt_event, stop_event):
    last_val = TouchStyle.NONE
    last_tap_time = 0.0
    tap_count = 0
    while not stop_event.is_set():
        if not idle_event.is_set():
            time.sleep(0.05)
            continue
        val = touch.read()
        if val in (TouchStyle.REAR, TouchStyle.FRONT) and last_val == TouchStyle.NONE:
            now = time.time()
            if now - last_tap_time > TAP_WINDOW_SECONDS:
                tap_count = 0
            tap_count += 1
            last_tap_time = now
            _pulse_feedback(leds, count=1, context="idle_tap")
            log_action("touch_tap", zone=_touch_label(val), count=tap_count, context="idle_start")
            if tap_count >= 2:
                tap_count = 0
                start_event.set()
                interrupt_event.set()
                _pulse_feedback(leds, count=2, context="idle_double_tap")
                log_action("touch_double_tap", context="start_radio")
        last_val = val
        time.sleep(0.05)


def touch_answer_interrupt_watcher(touch, leds, answering_event, interrupt_event, stop_event, radio_mode_event, paused_event):
    last_val = TouchStyle.NONE
    while not stop_event.is_set():
        if not answering_event.is_set() or (radio_mode_event.is_set() and not paused_event.is_set()):
            time.sleep(0.05)
            continue
        val = touch.read()
        if val in (TouchStyle.REAR, TouchStyle.FRONT) and last_val == TouchStyle.NONE:
            _pulse_feedback(leds, count=1, context="answer_interrupt")
            log_action("touch_tap", zone=_touch_label(val), context="answer_interrupt")
            interrupt_event.reason = "tap"
            interrupt_event.set()
        last_val = val
        time.sleep(0.05)


def voice_answer_interrupt_watcher(answering_event, interrupt_event, stop_event, radio_mode_event, paused_event, device):
    cooldown_until = 0.0
    while not stop_event.is_set():
        if not answering_event.is_set() or (radio_mode_event.is_set() and not paused_event.is_set()):
            time.sleep(0.05)
            continue
        if time.time() < cooldown_until:
            time.sleep(0.05)
            continue
        listen_stop = threading.Event()

        def _watch_end():
            while not stop_event.is_set():
                if not answering_event.is_set() or (radio_mode_event.is_set() and not paused_event.is_set()):
                    listen_stop.set()
                    return
                time.sleep(0.05)

        threading.Thread(target=_watch_end, daemon=True).start()
        try:
            heard = listen_for_wake_word_picovoice(
                keyword=BARGE_IN_KEYWORD,
                sensitivity=BARGE_IN_SENSITIVITY,
                device=device,
                break_event=listen_stop,
            )
        except Exception as exc:
            log_action("barge_in_error", error=str(exc))
            time.sleep(1.0)
            continue
        if heard and answering_event.is_set() and (not radio_mode_event.is_set() or paused_event.is_set()):
            interrupt_event.reason = "voice"
            interrupt_event.set()
            log_action("barge_in_heard", word=heard)
            cooldown_until = time.time() + 1.0
            while answering_event.is_set() and not stop_event.is_set():
                time.sleep(0.05)


def radio_wake_watcher(radio_mode_event, paused_event, stop_event, device, wake_event):
    while not stop_event.is_set():
        if not radio_mode_event.is_set() or paused_event.is_set():
            time.sleep(0.05)
            continue
        listen_stop = threading.Event()

        def _watch_end():
            while not stop_event.is_set():
                if not radio_mode_event.is_set() or paused_event.is_set():
                    listen_stop.set()
                    return
                time.sleep(0.05)

        threading.Thread(target=_watch_end, daemon=True).start()
        try:
            heard = listen_for_wake_word_picovoice(
                keyword=RADIO_WAKE_KEYWORD,
                sensitivity=RADIO_WAKE_SENSITIVITY,
                device=device,
                break_event=listen_stop,
            )
        except RuntimeError as exc:
            log_action("radio_wake_error", error=str(exc))
            time.sleep(1.0)
            continue
        if heard and radio_mode_event.is_set():
            wake_event.set()
            log_action("radio_wake_heard", word=heard)
            # Wait until radio resumes (paused_event cleared) or radio mode ends.
            while not stop_event.is_set():
                if not radio_mode_event.is_set():
                    break
                if not paused_event.is_set():
                    break
                time.sleep(0.05)
            time.sleep(0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic-test", action="store_true", help="Print live STT partials from the mic")
    parser.add_argument("--list-audio", action="store_true", help="List audio devices and exit")
    parser.add_argument("--mic-device", type=int, help="Override mic device index for STT")
    parser.add_argument("--debug-latency", action="store_true", help="Print latency timings for pro mode")
    args = parser.parse_args()
    log_path = _setup_logging()
    log_action("startup", pid=os.getpid(), args=vars(args), log_path=str(log_path))
    log_action(
        "config_audio",
        max_utterance_seconds=MAX_UTTERANCE_SECONDS,
        silence_threshold=CHATGPT_PRO_SILENCE_THRESHOLD,
        silence_seconds=CHATGPT_PRO_SILENCE_SECONDS,
        notes="threshold=rms cutoff, silence=end-of-speech window",
    )

    if args.list_audio:
        devices = sd.query_devices()
        log_action("list_audio", count=len(devices), default_device=sd.default.device)
        print("Audio devices:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
        print(f"Default device: {sd.default.device}")
        return

    if USE_PICOVOICE_WAKE_WORD:
        if not os.environ.get("PICOVOICE_ACCESS_KEY", "").strip():
            raise RuntimeError("PICOVOICE_ACCESS_KEY is not set. Put it in /home/pat/pidog/.env.")

    _ensure_single_instance(PIDFILE)
    log_action("single_instance", pidfile=PIDFILE, pid=os.getpid())

    mic_device = MIC_DEVICE if args.mic_device is None else args.mic_device
    if args.mic_device is None and MIC_DEVICE_NAME:
        try:
            for i, dev in enumerate(sd.query_devices()):
                if MIC_DEVICE_NAME.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
                    mic_device = i
                    break
        except Exception:
            pass
    mic_device_name = None
    try:
        dev_info = sd.query_devices(mic_device, "input")
        mic_device_name = dev_info.get("name")
    except Exception:
        pass
    log_action("mic_device_selected", index=mic_device, name=mic_device_name)
    pro_available, pro_reason = _check_openai_available()
    log_action("pro_available", available=pro_available, reason=pro_reason)
    stt_ready = threading.Event()
    stt_error = [None]
    stt_state = {"printed": False, "started": False}
    if args.mic_test:
        _start_stt_init(stt_ready, stt_error, stt_state, mic_device, MIC_SAMPLE_RATE, async_mode=False)
    elif not USE_PICOVOICE_WAKE_WORD or not pro_available:
        _start_stt_init(stt_ready, stt_error, stt_state, mic_device, MIC_SAMPLE_RATE, async_mode=STT_INIT_ASYNC)
    else:
        log_action("stt_init", mode="deferred", reason="picovoice+pro")
        print("STT: init deferred (Picovoice + Pro).")
    print(f"TTS: {TTS_ENGINE} {TTS_MODEL}")
    print(f"LLM: {LLM_MODEL}")
    print(
        "Audio params: "
        f"max_utterance={MAX_UTTERANCE_SECONDS}s "
        f"silence_threshold={CHATGPT_PRO_SILENCE_THRESHOLD} "
        f"silence_seconds={CHATGPT_PRO_SILENCE_SECONDS}s"
    )
    print(
        "Audio params: "
        f"min_rms={CHATGPT_PRO_MIN_RMS} "
        f"wake_min_rms={CHATGPT_PRO_WAKE_MIN_RMS}"
    )
    print(
        "Audio params: max_utterance=cap, silence_threshold=rms cutoff, "
        "silence_seconds=end-of-speech, min_rms=skip quiet audio"
    )
    log_action("models", tts_engine=TTS_ENGINE, tts_model=TTS_MODEL, llm_model=LLM_MODEL)
    t1 = time.time()
    tts = create_tts(engine=TTS_ENGINE, model=TTS_MODEL, length_scale=TTS_LENGTH_SCALE)
    tts_init_s = time.time() - t1
    print(f"Init: TTS {tts_init_s:.2f}s")
    log_action("init_tts", seconds=tts_init_s, engine=TTS_ENGINE, model=TTS_MODEL)
    t2 = time.time()
    speaker = SpeechQueue(tts, output_device=SPEAKER_DEVICE)
    speaker.start()
    speaker_init_s = time.time() - t2
    print(f"Init: Speaker {speaker_init_s:.2f}s")
    log_action("init_speaker", seconds=speaker_init_s, device=SPEAKER_DEVICE)
    _prime_softmaster()
    t3 = time.time()
    leds = LedController()
    leds.start()
    leds_init_s = time.time() - t3
    print(f"Init: LEDs {leds_init_s:.2f}s")
    log_action("init_leds", seconds=leds_init_s)
    t4 = time.time()
    touch = DualTouch()
    touch_init_s = time.time() - t4
    print(f"Init: Touch {touch_init_s:.2f}s")
    log_action("init_touch", seconds=touch_init_s)
    try:
        voltage, percent = get_battery_status()
        if percent <= 5:
            print(f"\033[31mBattery: {voltage:.2f} V (~{percent}%)\033[0m")
        else:
            print(f"Battery: {voltage:.2f} V (~{percent}%)")
        log_action("battery_status", voltage=voltage, percent=percent)
    except Exception:
        print("Battery: unavailable")
        log_action("battery_status", error="unavailable")
    stations = load_m3u_stations(M3U_PATH)
    if not stations:
        stations = [
            {"name": "Radio 1", "url": RADIO_URLS["1"]},
            {"name": "Radio 2", "url": RADIO_URLS["2"]},
            {"name": "Radio 3", "url": RADIO_URLS["3"]},
            {"name": "BBC World Service", "url": BBC_WORLD_SERVICE_URL},
        ]
    log_action("stations_loaded", count=len(stations), names=[s["name"] for s in stations])
    radio_playing = False
    radio_paused = False
    radio_last_url = None
    radio_station_index = None
    radio_mode_event = threading.Event()
    radio_wake_event = threading.Event()
    radio_resume_on_no_followup = False
    resume_ready_time = [0.0]
    paused_event = threading.Event()
    resume_event = threading.Event()
    interrupt_event = threading.Event()
    answer_interrupt_event = threading.Event()
    answering_event = threading.Event()
    idle_event = threading.Event()
    start_event = threading.Event()
    watcher_stop = threading.Event()
    watcher_thread = threading.Thread(
        target=touch_resume_watcher,
        args=(touch, leds, paused_event, resume_event, watcher_stop, resume_ready_time),
        daemon=True,
    )
    watcher_thread.start()
    start_thread = threading.Thread(
        target=touch_start_watcher,
        args=(touch, leds, idle_event, start_event, interrupt_event, watcher_stop),
        daemon=True,
    )
    start_thread.start()
    answer_thread = threading.Thread(
        target=touch_answer_interrupt_watcher,
        args=(touch, leds, answering_event, answer_interrupt_event, watcher_stop, radio_mode_event, paused_event),
        daemon=True,
    )
    answer_thread.start()
    voice_answer_thread = None
    if BARGE_IN_ENABLED and USE_PICOVOICE_WAKE_WORD:
        voice_answer_thread = threading.Thread(
            target=voice_answer_interrupt_watcher,
            args=(answering_event, answer_interrupt_event, watcher_stop, radio_mode_event, paused_event, mic_device),
            daemon=True,
        )
        voice_answer_thread.start()
        log_action("voice_barge_in_started", keyword=BARGE_IN_KEYWORD, sensitivity=BARGE_IN_SENSITIVITY)
    radio_wake_thread = None
    if RADIO_WAKE_ENABLED and USE_PICOVOICE_WAKE_WORD:
        radio_wake_thread = threading.Thread(
            target=radio_wake_watcher,
            args=(radio_mode_event, paused_event, watcher_stop, mic_device, radio_wake_event),
            daemon=True,
        )
        radio_wake_thread.start()
        log_action("radio_wake_started", keyword=RADIO_WAKE_KEYWORD, sensitivity=RADIO_WAKE_SENSITIVITY)
    log_action("touch_watchers_started")
    video = VideoManager(
        port=CAMERA_PORT,
        vflip=CAMERA_VFLIP,
        hflip=CAMERA_HFLIP,
        path=CAMERA_URL_PATH,
    )

    if args.mic_test:
        wake_stt, asr_stt, stt_model = _ensure_local_stt_ready(
            stt_ready,
            stt_error,
            stt_state,
            mic_device,
            MIC_SAMPLE_RATE,
            async_mode=False,
        )
        log_action("mic_test_start", device=mic_device, engine=STT_ENGINE)
        print("Mic test: speak into the microphone (Ctrl+C to exit).")
        try:
            if STT_ENGINE == "whisper":
                while True:
                    utterance = record_utterance_whisper(
                        asr_stt,
                        device=mic_device,
                        max_seconds=MAX_UTTERANCE_SECONDS,
                        input_rate=MIC_INPUT_RATE,
                        input_dtype=MIC_INPUT_DTYPE,
                    )
                    final = utterance.get("text", "").strip()
                    if final:
                        print(f"final: {final}")
            else:
                for result in wake_stt.listen(stream=True, device=mic_device):
                    if not result:
                        continue
                    if result.get("done"):
                        final = result.get("final", "").strip()
                        if final:
                            print(f"final: {final}")
                    else:
                        partial = result.get("partial", "").strip()
                        if partial:
                            print(f"partial: {partial}", end="\r", flush=True)
        except KeyboardInterrupt:
            print("\nExiting mic test.")
            log_action("mic_test_stop")
        return

    history = deque(maxlen=MEMORY_SENTENCES)
    use_chatgpt = False
    use_chatgpt_pro = pro_available
    pro_cost_total = 0.0
    just_woke = False
    if use_chatgpt_pro:
        leds.set_state("speak")
        speaker.say("Smart dog pro mode is on.")
        speaker.wait_idle()
        print("Smart mode: pro")
        log_action("mode_smart_pro", enabled=True)
    else:
        leds.set_state("speak")
        speaker.say("Pro mode not available. Starting local LLM.")
        speaker.wait_idle()
        print("Smart mode: local")
        log_action("mode_smart_pro", enabled=False, reason=pro_reason)
        _start_stt_init(
            stt_ready,
            stt_error,
            stt_state,
            mic_device,
            MIC_SAMPLE_RATE,
            async_mode=STT_INIT_ASYNC,
        )

    if USE_PICOVOICE_WAKE_WORD:
        print(f"Wake word: {PICOVOICE_KEYWORD} (Picovoice)")
    else:
        print(f"Wake words: {WAKE_WORDS}")
    print("Ready.")
    log_action(
        "ready",
        wake_source="picovoice" if USE_PICOVOICE_WAKE_WORD else "vosk",
        wake_words=WAKE_WORDS if not USE_PICOVOICE_WAKE_WORD else None,
        picovoice_keyword=PICOVOICE_KEYWORD if USE_PICOVOICE_WAKE_WORD else None,
    )
    try:
        convo_deadline = None
        wake_word_delayed = False
        if (use_chatgpt_pro and STT_INIT_ASYNC and SKIP_WAKE_UNTIL_STT_READY
                and not USE_PICOVOICE_WAKE_WORD):
            convo_deadline = time.time() + CONVO_WINDOW_SECONDS
            wake_word_delayed = True
            print("Smart mode: pro (wake word enabled after STT init)")
        print(f"Convo window: {CONVO_WINDOW_SECONDS}s")
        log_action("convo_window", seconds=CONVO_WINDOW_SECONDS, wake_word_delayed=wake_word_delayed)

        def _after_answer_interrupt():
            nonlocal convo_deadline, just_woke
            convo_deadline = time.time() + CONVO_WINDOW_SECONDS
            just_woke = True
            log_action("followup_window", seconds=CONVO_WINDOW_SECONDS, source="answer_interrupt")

        while True:
            if radio_playing:
                leds.set_state("wake")
                last_tap_time = 0.0
                tap_count = 0
                last_touch_val = TouchStyle.NONE
                while True:
                    if radio_wake_event.is_set():
                        radio_wake_event.clear()
                        stop_moc_stream()
                        radio_playing = False
                        radio_paused = True
                        radio_resume_on_no_followup = True
                        _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                        paused_event.set()
                        resume_event.clear()
                        interrupt_event.clear()
                        resume_ready_time[0] = time.time() + 1.0
                        leds.set_state("speak")
                        if not _play_confused_sound():
                            speaker.say("Yes?")
                            speaker.wait_idle()
                        log_action("radio_stop", source="voice_wake")
                        convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                        just_woke = True
                        break
                    val = touch.read()
                    if val == TouchStyle.REAR_TO_FRONT:
                        adjust_moc_volume(VOLUME_STEP)
                        log_action(
                            "radio_volume",
                            delta=VOLUME_STEP,
                            source="touch_swipe",
                            direction="rear_to_front",
                        )
                    elif val == TouchStyle.FRONT_TO_REAR:
                        adjust_moc_volume(-VOLUME_STEP)
                        log_action(
                            "radio_volume",
                            delta=-VOLUME_STEP,
                            source="touch_swipe",
                            direction="front_to_rear",
                        )
                    elif val in (TouchStyle.REAR, TouchStyle.FRONT) and last_touch_val == TouchStyle.NONE:
                        now = time.time()
                        if now - last_tap_time > TAP_WINDOW_SECONDS:
                            tap_count = 0
                        tap_count += 1
                        last_tap_time = now
                        _pulse_feedback(leds, count=1, context="radio_tap")
                        log_action(
                            "touch_tap",
                            zone=_touch_label(val),
                            count=tap_count,
                            context="radio_playing",
                        )
                        if tap_count >= 3:
                            tap_count = 0
                            if stations:
                                if radio_station_index is None:
                                    radio_station_index = 0
                                radio_station_index = (radio_station_index + 1) % len(stations)
                                station = stations[radio_station_index]
                                start_moc_stream(station["url"])
                                radio_last_url = station["url"]
                                leds.set_state("speak")
                                speaker.say(f"Radio: {station['name']}.")
                                speaker.wait_idle()
                                _pulse_feedback(leds, count=3, context="radio_triple_tap")
                                log_action(
                                    "radio_station_next",
                                    name=station["name"],
                                    url=station["url"],
                                    index=radio_station_index,
                                )
                    last_touch_val = val
                    if tap_count == 2 and time.time() - last_tap_time > TAP_WINDOW_SECONDS:
                        stop_moc_stream()
                        radio_playing = False
                        radio_paused = True
                        radio_resume_on_no_followup = False
                        _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                        paused_event.set()
                        resume_event.clear()
                        interrupt_event.clear()
                        resume_ready_time[0] = time.time() + 1.0
                        leds.set_state("speak")
                        speaker.say("Radio stopped. Double-tap to resume.")
                        speaker.wait_idle()
                        _pulse_feedback(leds, count=2, context="radio_double_tap")
                        log_action("radio_stop", source="touch_double_tap")
                        break
                    time.sleep(0.05)
                continue

            if radio_paused and radio_last_url:
                if resume_event.is_set() and time.time() >= resume_ready_time[0]:
                    resume_event.clear()
                    interrupt_event.clear()
                    leds.set_state("speak")
                    speaker.say("Resuming radio.")
                    speaker.wait_idle()
                    start_moc_stream(radio_last_url)
                    radio_station_index = find_station_index(stations, radio_last_url)
                    radio_playing = True
                    radio_paused = False
                    radio_resume_on_no_followup = False
                    _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                    paused_event.clear()
                    log_action("radio_resume", source="touch_double_tap", url=radio_last_url)
                    continue

            if convo_deadline is not None and time.time() > convo_deadline:
                convo_deadline = None
                resumed_radio = False
                if radio_resume_on_no_followup and radio_paused and radio_last_url:
                    start_moc_stream(radio_last_url)
                    radio_station_index = find_station_index(stations, radio_last_url)
                    radio_playing = True
                    radio_paused = False
                    _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                    paused_event.clear()
                    log_action("radio_resume", source="voice_wake_timeout", url=radio_last_url)
                    resumed_radio = True
                radio_resume_on_no_followup = False
                if resumed_radio:
                    continue

            if convo_deadline is None or time.time() > convo_deadline:
                print("\nWaiting for wake word...")
                leds.set_state("wake")
                idle_event.set()
                log_action("wake_listen_start")
                if USE_PICOVOICE_WAKE_WORD:
                    heard = listen_for_wake_word_picovoice(
                        keyword=PICOVOICE_KEYWORD,
                        sensitivity=PICOVOICE_SENSITIVITY,
                        device=mic_device,
                        break_event=interrupt_event if (radio_paused or idle_event.is_set()) else None,
                    )
                else:
                    wake_stt, asr_stt, stt_model = _ensure_local_stt_ready(
                        stt_ready,
                        stt_error,
                        stt_state,
                        mic_device,
                        MIC_SAMPLE_RATE,
                        async_mode=False,
                    )
                    heard = listen_for_wake_word(
                        wake_stt,
                        WAKE_WORDS,
                        device=mic_device,
                        break_event=interrupt_event if (radio_paused or idle_event.is_set()) else None,
                        strict=WAKE_WORDS_STRICT,
                    )
                idle_event.clear()
                if heard is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                    just_woke = True
                    log_action(
                        "wake_word_heard",
                        word=heard,
                        source="picovoice" if USE_PICOVOICE_WAKE_WORD else "vosk",
                    )
            else:
                if stt_ready.is_set() or not SKIP_WAKE_UNTIL_STT_READY:
                    heard = "active"
                    print(f"Convo active: {int(convo_deadline - time.time())}s left")
                    log_action(
                        "convo_active",
                        seconds_left=int(convo_deadline - time.time()),
                    )
                else:
                    heard = "active"
                    print("Convo active (wake word pending STT init).")
                    log_action("convo_active", seconds_left=None, wake_word_pending=True)
            if start_event.is_set():
                start_event.clear()
                interrupt_event.clear()
                station = get_default_station(stations)
                leds.set_state("speak")
                speaker.say(f"Starting radio {station['name']}.")
                speaker.wait_idle()
                start_moc_stream(station["url"])
                radio_last_url = station["url"]
                radio_station_index = find_station_index(stations, station["url"])
                radio_playing = True
                radio_paused = False
                radio_resume_on_no_followup = False
                _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                paused_event.clear()
                log_action(
                    "radio_start",
                    source="touch_double_tap",
                    name=station["name"],
                    url=station["url"],
                )
                continue
            if heard is None and radio_paused and resume_event.is_set() and time.time() >= resume_ready_time[0]:
                resume_event.clear()
                interrupt_event.clear()
                leds.set_state("speak")
                speaker.say("Resuming radio.")
                speaker.wait_idle()
                start_moc_stream(radio_last_url)
                radio_playing = True
                radio_paused = False
                radio_resume_on_no_followup = False
                _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                paused_event.clear()
                log_action("radio_resume", source="touch_double_tap", url=radio_last_url)
                continue

            leds.set_state("listen")
            print("Listening...")
            log_action(
                "listen_start",
                mode="chatgpt_pro" if use_chatgpt_pro else STT_ENGINE,
                mic_device=mic_device,
            )
            if not use_chatgpt_pro:
                wake_stt, asr_stt, stt_model = _ensure_local_stt_ready(
                    stt_ready,
                    stt_error,
                    stt_state,
                    mic_device,
                    MIC_SAMPLE_RATE,
                    async_mode=False,
                )
            if use_chatgpt_pro:
                timing = {
                    "record": 0.0,
                    "stt": 0.0,
                    "llm": 0.0,
                    "tts": 0.0,
                    "play": 0.0,
                    "start_play": 0.0,
                }
                t_start = time.perf_counter()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp_path = tmp.name
                tmp.close()
                try:
                    t0 = time.perf_counter()
                    rec = record_audio_wav(
                        tmp_path,
                        device=mic_device,
                        max_seconds=MAX_UTTERANCE_SECONDS,
                        silence_threshold=CHATGPT_PRO_SILENCE_THRESHOLD,
                        silence_seconds=CHATGPT_PRO_SILENCE_SECONDS,
                        input_rate=MIC_INPUT_RATE,
                        input_dtype=MIC_INPUT_DTYPE,
                    )
                    timing["record"] = time.perf_counter() - t0
                    log_action(
                        "audio_recorded",
                        ok=rec.get("ok"),
                        duration=rec.get("duration"),
                        rms=rec.get("rms"),
                        mode="chatgpt_pro",
                    )
                    if CHATGPT_PRO_DEBUG_RECORDING:
                        print(f"Pro rec ok={rec.get('ok')} dur={rec.get('duration'):.2f}s rms={rec.get('rms'):.4f}")
                        if rec.get("ok"):
                            subprocess.run(["aplay", "-q", tmp_path], check=False)
                    if not rec.get("ok"):
                        leds.set_state("speak")
                        speaker.say("Sorry, can you repeat that?")
                        speaker.wait_idle()
                        log_action("prompt_repeat", reason="record_failed")
                        if convo_deadline is not None:
                            convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                        continue
                    min_rms = CHATGPT_PRO_WAKE_MIN_RMS if just_woke else CHATGPT_PRO_MIN_RMS
                    if rec.get("rms", 0.0) < min_rms:
                        if args.debug_latency:
                            print(
                                f"Pro: low RMS ({rec.get('rms', 0.0):.4f} < {min_rms:.4f}), skipping."
                            )
                        log_action("audio_low_rms", rms=rec.get("rms"), threshold=min_rms)
                        continue
                    just_woke = False
                    t0 = time.perf_counter()
                    text = openai_transcribe(
                        tmp_path,
                        model=CHATGPT_STT_MODEL,
                        language=CHATGPT_STT_LANGUAGE,
                    ).strip()
                    log_action("transcript", text=text, engine="openai", language=CHATGPT_STT_LANGUAGE)
                    if not is_likely_english(text):
                        log_action("transcript_dropped", reason="non_english", text=text)
                        print(f"Pro: dropped non-English transcript: {_clip_text(text)!r}")
                        continue
                    timing["stt"] = time.perf_counter() - t0
                    if CHATGPT_PRO_DEBUG_RECORDING:
                        print(f"Pro STT text: {text!r}")
                except RuntimeError as exc:
                    use_chatgpt_pro = False
                    use_chatgpt = False
                    _start_stt_init(
                        stt_ready,
                        stt_error,
                        stt_state,
                        mic_device,
                        MIC_SAMPLE_RATE,
                        async_mode=STT_INIT_ASYNC,
                    )
                    leds.set_state("speak")
                    speaker.say("Pro mode not available. Starting local LLM.")
                    speaker.wait_idle()
                    log_action("mode_smart_pro", enabled=False, reason=str(exc))
                    if convo_deadline is not None:
                        convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                    print("Smart mode: local")
                    continue
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                utterance = {"text": text, "confidence": 1.0 if text else 0.0}
            elif STT_ENGINE == "whisper":
                utterance = record_utterance_whisper(
                    asr_stt,
                    device=mic_device,
                    max_seconds=MAX_UTTERANCE_SECONDS,
                    input_rate=MIC_INPUT_RATE,
                    input_dtype=MIC_INPUT_DTYPE,
                )
            else:
                utterance = record_utterance(
                    asr_stt,
                    device=mic_device,
                    max_seconds=MAX_UTTERANCE_SECONDS,
                    stream=True,
                )
            text = utterance["text"].strip()
            if not use_chatgpt_pro:
                log_action(
                    "transcript",
                    text=text,
                    engine=STT_ENGINE,
                    confidence=utterance.get("confidence"),
                )
            if not text or utterance["confidence"] < CONFIDENCE_THRESHOLD:
                leds.set_state("speak")
                speaker.say("Sorry, can you repeat that?")
                speaker.wait_idle()
                log_action("prompt_repeat", reason="low_confidence")
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue

            command = repair_transcript(text)
            words = command.split()
            log_action(
                "command_received",
                raw=text,
                normalized=command,
                confidence=utterance.get("confidence"),
                mode="chatgpt_pro" if use_chatgpt_pro else ("chatgpt" if use_chatgpt else "local"),
            )
            if not (use_chatgpt or use_chatgpt_pro):
                if len(words) < 2 or (len(words) == 1 and words[0] in SHORT_UTTERANCE_WORDS):
                    leds.set_state("speak")
                    speaker.say("Sorry, can you repeat that?")
                    speaker.wait_idle()
                    log_action("command_rejected", reason="too_short", command=command)
                    if convo_deadline is not None:
                        convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                    continue
            radio_override = normalize_radio_command(command)
            if radio_override:
                command = radio_override
                log_action("command_normalized", normalized=command)
            if "play radio bbc" in command or "start radio bbc" in command:
                _begin_answering(answering_event, answer_interrupt_event, context="radio_start")
                leds.set_state("speak")
                speaker.say("Starting BBC World Service.")
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="radio_start")
                start_moc_stream(BBC_WORLD_SERVICE_URL)
                radio_playing = True
                radio_paused = False
                radio_resume_on_no_followup = False
                radio_last_url = BBC_WORLD_SERVICE_URL
                radio_station_index = find_station_index(stations, BBC_WORLD_SERVICE_URL)
                _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                paused_event.clear()
                interrupt_event.clear()
                log_action(
                    "radio_start",
                    source="voice",
                    name="BBC World Service",
                    url=BBC_WORLD_SERVICE_URL,
                )
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if "play radio" in command or "start radio" in command:
                if "start radio" in command and "play radio" not in command:
                    command = command.replace("start radio", "play radio")
                for word, key in RADIO_ALIASES.items():
                    if f"play radio {word}" in command:
                        command = command.replace(f"play radio {word}", f"play radio {key}")
                        break
                for key, url in RADIO_URLS.items():
                    if f"play radio {key}" in command:
                        _begin_answering(answering_event, answer_interrupt_event, context="radio_start")
                        leds.set_state("speak")
                        speaker.say(f"Starting radio {key}.")
                        _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                        _end_answering(answering_event, context="radio_start")
                        start_moc_stream(url)
                        radio_playing = True
                        radio_paused = False
                        radio_resume_on_no_followup = False
                        radio_last_url = url
                        radio_station_index = find_station_index(stations, url)
                        _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                        paused_event.clear()
                        interrupt_event.clear()
                        log_action(
                            "radio_start",
                            source="voice",
                            name=f"Radio {key}",
                            url=url,
                        )
                        if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                            convo_deadline = None
                            break
                        break
                else:
                    print(f"Unknown radio command: {text}")
                    log_action("radio_unknown", text=text, command=command)
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if "stop radio" in command:
                _begin_answering(answering_event, answer_interrupt_event, context="radio_stop")
                leds.set_state("speak")
                speaker.say("Stopping the radio.")
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="radio_stop")
                stop_moc_stream()
                radio_playing = False
                radio_paused = False
                radio_resume_on_no_followup = False
                _sync_radio_mode(radio_mode_event, radio_playing, radio_paused)
                paused_event.clear()
                interrupt_event.clear()
                log_action("radio_stop", source="voice")
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if (
                command == "camera"
                or "turn on camera" in command
                or "turn the camera on" in command
                or "turn camera on" in command
            ):
                _begin_answering(answering_event, answer_interrupt_event, context="camera_on")
                leds.set_state("speak")
                url, started, error = video.start()
                if url:
                    print(f"Camera feed: {url}")
                    msg = "Camera activated." if started else "Camera already active."
                else:
                    msg = "Sorry, I could not start the camera."
                speaker.say(msg)
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="camera_on")
                log_action(
                    "camera_start",
                    ok=bool(url),
                    started=started if url else None,
                    url=url,
                    error=error,
                )
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if (
                command == "camera off"
                or "turn the camera off" in command
                or "turn off camera" in command
                or "stop camera" in command
            ):
                _begin_answering(answering_event, answer_interrupt_event, context="camera_off")
                leds.set_state("speak")
                stopped = video.stop()
                if stopped:
                    msg = "Camera turned off."
                else:
                    msg = "Camera is not running."
                speaker.say(msg)
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="camera_off")
                log_action("camera_stop", ok=stopped)
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if "battery" in command:
                _begin_answering(answering_event, answer_interrupt_event, context="battery")
                leds.set_state("speak")
                try:
                    voltage, percent = get_battery_status()
                    speaker.say(f"Battery is at about {percent} percent.")
                    print(f"Battery: {voltage:.2f} V (~{percent}%)")
                    log_action("battery_query", voltage=voltage, percent=percent)
                except Exception:
                    speaker.say("Sorry, I cannot read the battery right now.")
                    print("Battery: unavailable")
                    log_action("battery_query", error="unavailable")
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="battery")
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue

            if "stop smart" in command:
                use_chatgpt = False
                use_chatgpt_pro = False
                _begin_answering(answering_event, answer_interrupt_event, context="smart_off")
                leds.set_state("speak")
                speaker.say("Switching to local model.")
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="smart_off")
                log_action("mode_smart", enabled=False)
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                print("Smart mode: off")
                continue
            if "smart mode pro" in command or "smart dog pro" in command:
                use_chatgpt = False
                use_chatgpt_pro = True
                _begin_answering(answering_event, answer_interrupt_event, context="smart_pro_on")
                leds.set_state("speak")
                speaker.say("Hello, I am a full LLM pipe to ChatGPT.")
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="smart_pro_on")
                log_action("mode_smart_pro", enabled=True)
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                print("Smart mode: pro")
                continue
            if "go smart" in command or "smart mode" in command or "smart dog" in command:
                use_chatgpt = True
                use_chatgpt_pro = False
                _begin_answering(answering_event, answer_interrupt_event, context="smart_on")
                leds.set_state("speak")
                speaker.say("ChatGPT here, how can I help?")
                _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                _end_answering(answering_event, context="smart_on")
                log_action("mode_smart", enabled=True)
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    convo_deadline = None
                    continue
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                print("Smart mode: on")
                continue

            print(f"User: {command}")
            prompt_text = f"{build_context(history)}{command}"
            buffer = ""
            full_response = ""
            _begin_answering(answering_event, answer_interrupt_event, context="response")
            leds.set_state("speak")
            answer_interrupted = False
            api_key_missing = False
            try:
                active_prompt = None if use_chatgpt else SYSTEM_PROMPT
                active_max_phrases = None if use_chatgpt else 4
                active_max_chars = None if use_chatgpt else 260
                active_max_words = None if use_chatgpt else 32
                if use_chatgpt_pro:
                    if answer_interrupt_event.is_set():
                        raise RuntimeError("answer_interrupt")
                    t0 = time.perf_counter()
                    response_text, usage = openai_chat_reply(
                        prompt_text,
                        model=CHATGPT_MODEL,
                        system_prompt=(
                            "You are a helpful robot dog. Reply in English only, no more than two short sentences."
                        ),
                    )
                    if answer_interrupt_event.is_set():
                        raise RuntimeError("answer_interrupt")
                    timing["llm"] = time.perf_counter() - t0
                    response_text = sanitize_tts_text(response_text)
                    sentences, remainder = split_sentences(response_text)
                    if len(sentences) >= 2:
                        response_text = " ".join(sentences[:2]).strip()
                    elif sentences:
                        response_text = sentences[0].strip()
                    elif remainder.strip():
                        response_text = remainder.strip()
                    audio_out_s = 0.0
                    wav_out_s = 0.0
                    if response_text:
                        print(response_text)
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        tmp_path = tmp.name
                        tmp.close()
                        try:
                            t0 = time.perf_counter()
                            openai_tts_to_file(
                                response_text,
                                tmp_path,
                                model=CHATGPT_TTS_MODEL,
                                voice=CHATGPT_TTS_VOICE,
                            )
                            if answer_interrupt_event.is_set():
                                raise RuntimeError("answer_interrupt")
                            timing["tts"] = time.perf_counter() - t0
                            wav_out_s = wav_seconds(tmp_path)
                            if wav_out_s > 120.0:
                                wav_out_s = 0.0
                            t0 = time.perf_counter()
                            timing["start_play"] = t0 - t_start
                            if not _play_wav_interruptible(tmp_path, answer_interrupt_event):
                                raise RuntimeError("answer_interrupt")
                            timing["play"] = time.perf_counter() - t0
                            audio_out_s = timing["play"] if timing["play"] > 0.1 else wav_out_s
                        except Exception as exc:
                            if str(exc) != "answer_interrupt":
                                print(f"Pro TTS failed: {exc}")
                            timing["start_play"] = time.perf_counter() - t_start
                            if response_text:
                                speaker.say(response_text)
                        finally:
                            try:
                                os.unlink(tmp_path)
                            except Exception:
                                pass
                    full_response = response_text
                    if args.debug_latency:
                        audio_in_s = float(rec.get("duration") or 0.0)
                        in_tokens = None
                        out_tokens = None
                        if usage:
                            in_tokens = usage.get("prompt_tokens")
                            out_tokens = usage.get("completion_tokens")
                        stt_c, tts_c, llm_c, total_c = estimate_costs(
                            audio_in_s,
                            audio_out_s,
                            in_tokens,
                            out_tokens,
                        )
                        pro_cost_total += total_c
                        total = time.perf_counter() - t_start
                        print(
                            "Pro latency: "
                            f"record={timing['record']:.2f}s "
                            f"stt={timing['stt']:.2f}s "
                            f"llm={timing['llm']:.2f}s "
                            f"tts={timing['tts']:.2f}s "
                            f"start_play={timing['start_play']:.2f}s "
                            f"play={timing['play']:.2f}s "
                            f"total={total:.2f}s"
                        )
                        print(
                            "Pro cost: "
                            f"stt=${stt_c:.6f} "
                            f"tts=${tts_c:.6f} "
                            f"llm=${llm_c:.6f} "
                            f"turn=${total_c:.6f} "
                            f"total=${pro_cost_total:.6f} "
                            f"(in={audio_in_s:.2f}s out={audio_out_s:.2f}s)"
                        )
                else:
                    if answer_interrupt_event.is_set():
                        raise RuntimeError("answer_interrupt")
                    for chunk in stream_reply(
                        prompt_text,
                        model=CHATGPT_MODEL if use_chatgpt else LLM_MODEL,
                        system_prompt=active_prompt,
                        max_phrases=active_max_phrases,
                        max_chars=active_max_chars,
                        max_words=active_max_words,
                    ):
                        if answer_interrupt_event.is_set():
                            raise RuntimeError("answer_interrupt")
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        buffer += chunk
                        full_response += chunk
                        sentences, buffer = split_sentences(buffer)
                    for sentence in sentences:
                        if answer_interrupt_event.is_set():
                            raise RuntimeError("answer_interrupt")
                        speaker.say(sanitize_tts_text(sentence))
            except RuntimeError as exc:
                if str(exc) == "answer_interrupt":
                    answer_interrupted = True
                elif "OPENAI_API_KEY" in str(exc):
                    use_chatgpt = False
                    use_chatgpt_pro = False
                    leds.set_state("speak")
                    speaker.say("ChatGPT mode not available. Starting local LLM.")
                    _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
                    log_action("mode_smart", enabled=False, reason="missing_api_key")
                    api_key_missing = True
                else:
                    raise

            if api_key_missing:
                _end_answering(answering_event, context="response")
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    _after_answer_interrupt()
                continue
            if answer_interrupted:
                if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                    _after_answer_interrupt()
                    continue

            if buffer.strip():
                cleaned = sanitize_tts_text(buffer)
                if cleaned:
                    speaker.say(cleaned)
                    full_response += cleaned
            _wait_for_speaker_or_interrupt(speaker, answer_interrupt_event)
            if full_response.strip():
                log_action(
                    "response",
                    text=full_response,
                    mode="chatgpt_pro" if use_chatgpt_pro else ("chatgpt" if use_chatgpt else "local"),
                )
            print()
            _end_answering(answering_event, context="response")
            if _handle_answer_interrupt(answer_interrupt_event, answering_event, speaker, leds):
                _after_answer_interrupt()
                continue
            convo_deadline = time.time() + CONVO_WINDOW_SECONDS
            final_sentences, _ = split_sentences(full_response)
            for sentence in final_sentences[-MEMORY_SENTENCES:]:
                history.append(sentence)
    except KeyboardInterrupt:
        print("\nExiting...")
        log_action("shutdown", reason="keyboard_interrupt")
    finally:
        watcher_stop.set()
        watcher_thread.join(timeout=1)
        start_thread.join(timeout=1)
        answer_thread.join(timeout=1)
        if voice_answer_thread is not None:
            voice_answer_thread.join(timeout=1)
        if radio_wake_thread is not None:
            radio_wake_thread.join(timeout=1)
        touch.close()
        leds.stop()
        speaker.stop()
        log_action("shutdown", reason="cleanup_complete")


if __name__ == "__main__":
    main()
