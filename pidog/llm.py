from robot_hat.llm import *

import os
import subprocess
import re
from pathlib import Path


def _load_dotenv(paths):
    for path in paths:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if not key:
                    continue
                if not os.environ.get(key):
                    os.environ[key] = value
        except Exception:
            continue


_load_dotenv([
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parent / ".env",
])


def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in /home/pat/pidog/.env.")
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "openai is not installed. Run `pip install openai` or install the "
            "project extra with `pip install -e .[openai]`."
        ) from exc
    return OpenAI(api_key=api_key)


def _stream_with_limits(chunks, max_phrases, max_chars, max_words):
    phrase_count = 0
    total_chars = 0
    word_count = 0
    in_word = False
    end_re = re.compile(r"[.!?,;]")
    for chunk in chunks:
        if max_phrases is None and max_chars is None and max_words is None:
            yield chunk
            continue

        out = ""
        for ch in chunk:
            out += ch
            total_chars += 1
            if ch.isspace():
                in_word = False
            else:
                if not in_word:
                    word_count += 1
                    in_word = True
            if max_chars is not None and total_chars >= max_chars:
                yield out
                return
            if max_words is not None and word_count >= max_words:
                yield out
                return
            if end_re.match(ch):
                phrase_count += 1
                if max_phrases is not None and phrase_count >= max_phrases:
                    yield out
                    return
        if out:
            yield out


def openai_transcribe(audio_path, model="gpt-4o-transcribe", language=None):
    client = _get_openai_client()
    with open(audio_path, "rb") as handle:
        kwargs = {"model": model, "file": handle}
        if language:
            kwargs["language"] = language
        result = client.audio.transcriptions.create(**kwargs)
    return getattr(result, "text", "") or ""


def openai_chat_reply(prompt_text, model="gpt-4o-mini", system_prompt=None):
    client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    content = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    usage_data = None
    if usage:
        usage_data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return content.strip(), usage_data


def openai_tts_to_file(text, output_path, model="gpt-4o-tts-1", voice="alloy"):
    client = _get_openai_client()
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format="wav",
    )
    if hasattr(response, "write_to_file"):
        response.write_to_file(output_path)
        return
    data = getattr(response, "content", None)
    if data:
        with open(output_path, "wb") as handle:
            handle.write(data)


def stream_reply(
    prompt_text,
    model="smollm:360m",
    system_prompt=None,
    max_phrases=2,
    max_chars=160,
    max_words=18,
):
    system_prompt = system_prompt or "You are a helpful dog assistant. Keep replies brief and practical."
    if model.startswith("gpt-"):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set. Put it in /home/pat/pidog/.env.")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ],
            stream=True,
            temperature=0.4,
        )
        def _chunks():
            for event in response:
                delta = event.choices[0].delta
                content = delta.content or ""
                if content:
                    yield content
        yield from _stream_with_limits(_chunks(), max_phrases, max_chars, max_words)
        return

    full_prompt = f"{system_prompt}\n\nUser: {prompt_text}\nAssistant:"

    env = os.environ.copy()
    env["OLLAMA_NOHISTORY"] = "1"

    proc = subprocess.Popen(
        ["ollama", "run", model, "--nowordwrap"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        proc.stdin.write(full_prompt)
        proc.stdin.close()
        def _chunks():
            while True:
                chunk = proc.stdout.read(64)
                if not chunk:
                    break
                yield chunk
        yield from _stream_with_limits(_chunks(), max_phrases, max_chars, max_words)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()
