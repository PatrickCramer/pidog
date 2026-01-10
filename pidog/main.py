import argparse
import logging
from collections import deque
import atexit
import os
os.environ.setdefault("ORT_LOGGING_LEVEL", "4")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
import re
import subprocess
import sys
import threading
import time
import tempfile
import wave
import sounddevice as sd
import signal
from pathlib import Path

from .stt import (
    create_stt,
    create_whisper_stt,
    listen_for_wake_word,
    record_utterance,
    record_utterance_whisper,
    record_audio_wav,
)
from .leds import LedController
from .tts import create_tts, SpeechQueue
from .llm import stream_reply, openai_transcribe, openai_chat_reply, openai_tts_to_file
from .dual_touch import DualTouch, TouchStyle
from robot_hat import utils as rh_utils


WAKE_WORDS = ["ziggy", "pidog", "pie dog", "hi dog"]
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
CHATGPT_PRO_SILENCE_THRESHOLD = 0.003
CHATGPT_PRO_SILENCE_SECONDS = 1.5
CHATGPT_PRO_MIN_RMS = 0.004
CHATGPT_PRO_WAKE_MIN_RMS = 0.0

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
MAX_UTTERANCE_SECONDS = 12
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
CONVO_WINDOW_SECONDS = 15
PIDFILE = "/tmp/pidog.pid"
STT_INIT_ASYNC = True
SKIP_WAKE_UNTIL_STT_READY = True
TAP_PULSE_COLOR = [0, 140, 255]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mic-test", action="store_true", help="Print live STT partials from the mic")
    parser.add_argument("--list-audio", action="store_true", help="List audio devices and exit")
    parser.add_argument("--mic-device", type=int, help="Override mic device index for STT")
    parser.add_argument("--debug-latency", action="store_true", help="Print latency timings for pro mode")
    args = parser.parse_args()
    log_path = _setup_logging()
    log_action("startup", pid=os.getpid(), args=vars(args), log_path=str(log_path))

    if args.list_audio:
        devices = sd.query_devices()
        log_action("list_audio", count=len(devices), default_device=sd.default.device)
        print("Audio devices:")
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (in={dev['max_input_channels']}, out={dev['max_output_channels']})")
        print(f"Default device: {sd.default.device}")
        return

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
    stt_ready = threading.Event()
    stt_error = [None]
    stt_state = {"printed": False}
    if STT_INIT_ASYNC:
        threading.Thread(
            target=_init_stt_background,
            args=(stt_ready, stt_error, stt_state, mic_device, MIC_SAMPLE_RATE),
            daemon=True,
        ).start()
        log_action("stt_init", mode="async", device=mic_device, samplerate=MIC_SAMPLE_RATE)
        print("STT: init in background.")
    else:
        _init_stt_background(stt_ready, stt_error, stt_state, mic_device, MIC_SAMPLE_RATE)
        _await_stt(stt_ready, stt_error, stt_state)
        log_action("stt_init", mode="sync", device=mic_device, samplerate=MIC_SAMPLE_RATE)
    print(f"TTS: {TTS_ENGINE} {TTS_MODEL}")
    print(f"LLM: {LLM_MODEL}")
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
    resume_ready_time = [0.0]
    paused_event = threading.Event()
    resume_event = threading.Event()
    interrupt_event = threading.Event()
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
    log_action("touch_watchers_started")

    if args.mic_test:
        wake_stt, asr_stt, stt_model = _await_stt(stt_ready, stt_error, stt_state)
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
    use_chatgpt_pro = True
    pro_cost_total = 0.0
    just_woke = False
    if use_chatgpt_pro:
        leds.set_state("speak")
        speaker.say("Smart dog pro mode is on.")
        speaker.wait_idle()
        print("Smart mode: pro")
        log_action("mode_smart_pro", enabled=True)

    print(f"Wake words: {WAKE_WORDS}")
    print("Ready.")
    log_action("ready", wake_words=WAKE_WORDS)
    try:
        convo_deadline = None
        wake_word_delayed = False
        if use_chatgpt_pro and STT_INIT_ASYNC and SKIP_WAKE_UNTIL_STT_READY:
            convo_deadline = time.time() + CONVO_WINDOW_SECONDS
            wake_word_delayed = True
            print("Smart mode: pro (wake word enabled after STT init)")
        print(f"Convo window: {CONVO_WINDOW_SECONDS}s")
        log_action("convo_window", seconds=CONVO_WINDOW_SECONDS, wake_word_delayed=wake_word_delayed)
        while True:
            if radio_playing:
                leds.set_state("wake")
                last_tap_time = 0.0
                tap_count = 0
                last_touch_val = TouchStyle.NONE
                while True:
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
                    paused_event.clear()
                    log_action("radio_resume", source="touch_double_tap", url=radio_last_url)
                    continue

            if convo_deadline is not None and time.time() > convo_deadline:
                convo_deadline = None

            if convo_deadline is None or time.time() > convo_deadline:
                print("\nWaiting for wake word...")
                leds.set_state("wake")
                idle_event.set()
                log_action("wake_listen_start")
                wake_stt, asr_stt, stt_model = _await_stt(stt_ready, stt_error, stt_state)
                heard = listen_for_wake_word(
                    wake_stt,
                    WAKE_WORDS,
                    device=mic_device,
                    break_event=interrupt_event if (radio_paused or idle_event.is_set()) else None,
                )
                idle_event.clear()
                if heard is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                    just_woke = True
                    log_action("wake_word_heard", word=heard)
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
                wake_stt, asr_stt, stt_model = _await_stt(stt_ready, stt_error, stt_state)
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
                    leds.set_state("speak")
                    speaker.say("Smart mode needs an API key. Switching to local model.")
                    speaker.wait_idle()
                    log_action("mode_smart_pro", enabled=False, reason=str(exc))
                    if convo_deadline is not None:
                        convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                    print("Smart mode: off")
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
                leds.set_state("speak")
                speaker.say("Starting BBC World Service.")
                speaker.wait_idle()
                start_moc_stream(BBC_WORLD_SERVICE_URL)
                radio_playing = True
                radio_paused = False
                radio_last_url = BBC_WORLD_SERVICE_URL
                radio_station_index = find_station_index(stations, BBC_WORLD_SERVICE_URL)
                paused_event.clear()
                interrupt_event.clear()
                log_action(
                    "radio_start",
                    source="voice",
                    name="BBC World Service",
                    url=BBC_WORLD_SERVICE_URL,
                )
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
                        leds.set_state("speak")
                        speaker.say(f"Starting radio {key}.")
                        speaker.wait_idle()
                        start_moc_stream(url)
                        radio_playing = True
                        radio_paused = False
                        radio_last_url = url
                        radio_station_index = find_station_index(stations, url)
                        paused_event.clear()
                        interrupt_event.clear()
                        log_action(
                            "radio_start",
                            source="voice",
                            name=f"Radio {key}",
                            url=url,
                        )
                        break
                else:
                    print(f"Unknown radio command: {text}")
                    log_action("radio_unknown", text=text, command=command)
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if "stop radio" in command:
                leds.set_state("speak")
                speaker.say("Stopping the radio.")
                speaker.wait_idle()
                stop_moc_stream()
                radio_playing = False
                radio_paused = False
                paused_event.clear()
                interrupt_event.clear()
                log_action("radio_stop", source="voice")
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue
            if "battery" in command:
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
                speaker.wait_idle()
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                continue

            if "stop smart" in command:
                use_chatgpt = False
                use_chatgpt_pro = False
                leds.set_state("speak")
                speaker.say("Switching to local model.")
                speaker.wait_idle()
                log_action("mode_smart", enabled=False)
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                print("Smart mode: off")
                continue
            if "smart mode pro" in command or "smart dog pro" in command:
                use_chatgpt = False
                use_chatgpt_pro = True
                leds.set_state("speak")
                speaker.say("Hello, I am a full LLM pipe to ChatGPT.")
                speaker.wait_idle()
                log_action("mode_smart_pro", enabled=True)
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                print("Smart mode: pro")
                continue
            if "go smart" in command or "smart mode" in command or "smart dog" in command:
                use_chatgpt = True
                use_chatgpt_pro = False
                leds.set_state("speak")
                speaker.say("ChatGPT here, how can I help?")
                speaker.wait_idle()
                log_action("mode_smart", enabled=True)
                if convo_deadline is not None:
                    convo_deadline = time.time() + CONVO_WINDOW_SECONDS
                print("Smart mode: on")
                continue

            print(f"User: {command}")
            prompt_text = f"{build_context(history)}{command}"
            buffer = ""
            full_response = ""
            leds.set_state("speak")
            try:
                active_prompt = None if use_chatgpt else SYSTEM_PROMPT
                active_max_phrases = None if use_chatgpt else 4
                active_max_chars = None if use_chatgpt else 260
                active_max_words = None if use_chatgpt else 32
                if use_chatgpt_pro:
                    t0 = time.perf_counter()
                    response_text, usage = openai_chat_reply(
                        prompt_text,
                        model=CHATGPT_MODEL,
                        system_prompt=(
                            "You are a helpful robot dog. Reply in English only, no more than two short sentences."
                        ),
                    )
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
                            timing["tts"] = time.perf_counter() - t0
                            wav_out_s = wav_seconds(tmp_path)
                            if wav_out_s > 120.0:
                                wav_out_s = 0.0
                            t0 = time.perf_counter()
                            timing["start_play"] = t0 - t_start
                            subprocess.run(["aplay", "-q", tmp_path], check=False)
                            timing["play"] = time.perf_counter() - t0
                            audio_out_s = timing["play"] if timing["play"] > 0.1 else wav_out_s
                        except Exception as exc:
                            print(f"Pro TTS failed: {exc}")
                            timing["start_play"] = time.perf_counter() - t_start
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
                    for chunk in stream_reply(
                        prompt_text,
                        model=CHATGPT_MODEL if use_chatgpt else LLM_MODEL,
                        system_prompt=active_prompt,
                        max_phrases=active_max_phrases,
                        max_chars=active_max_chars,
                        max_words=active_max_words,
                    ):
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        buffer += chunk
                        full_response += chunk
                        sentences, buffer = split_sentences(buffer)
                    for sentence in sentences:
                        speaker.say(sanitize_tts_text(sentence))
            except RuntimeError as exc:
                if "OPENAI_API_KEY" in str(exc):
                    use_chatgpt = False
                    use_chatgpt_pro = False
                    leds.set_state("speak")
                    speaker.say("Smart mode needs an API key. Switching to local model.")
                    speaker.wait_idle()
                    log_action("mode_smart", enabled=False, reason="missing_api_key")
                    continue
                raise

            if buffer.strip():
                cleaned = sanitize_tts_text(buffer)
                if cleaned:
                    speaker.say(cleaned)
                    full_response += cleaned
            speaker.wait_idle()
            if full_response.strip():
                log_action(
                    "response",
                    text=full_response,
                    mode="chatgpt_pro" if use_chatgpt_pro else ("chatgpt" if use_chatgpt else "local"),
                )
            print()
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
        touch.close()
        leds.stop()
        speaker.stop()
        log_action("shutdown", reason="cleanup_complete")


if __name__ == "__main__":
    main()
