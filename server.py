#!/usr/bin/env python3
"""Codex Local Voice MCP server.

Hands-free voice loop: listen (faster-whisper) -> send text to Codex -> speak reply (F5-TTS) -> resume.

Run:  python server.py
Or via Codex: codex mcp add local-voice -- python server.py
"""
from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from mcp.server import MCPServer
from mcp.server.stdio import stdio_server

# ---- Config (env overrides) ----
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SILENCE_MS = int(os.getenv("SILENCE_MS", "700"))
SAMPLE_RATE = 16000
REF_AUDIO = os.getenv("F5_REF_AUDIO", str(Path(__file__).parent / "assets" / "ref.wav"))
REF_TEXT = os.getenv("F5_REF_TEXT", str(Path(__file__).parent / "assets" / "ref.txt"))


@dataclass
class VoiceState:
    running: bool = False
    speaking: bool = False
    last_text: str = ""
    log: list[str] = field(default_factory=list)


STATE = VoiceState()
_audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
_whisper = None
_f5 = None


def _load_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    return _whisper


def _load_f5():
    global _f5
    if _f5 is None:
        try:
            from f5_tts.api import F5TTS
            _f5 = F5TTS(model="F5TTS_v1_Base", device="auto")
        except Exception as e:
            STATE.log.append(f"F5-TTS unavailable: {e}")
            _f5 = None
    return _f5


def _record_until_silence(max_seconds: float = 30.0) -> Optional[np.ndarray]:
    """Blocking record: start on voice, stop after SILENCE_MS of silence."""
    chunk = 0.05
    frames = []
    silent_for = 0.0
    started = False
    t0 = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while time.time() - t0 < max_seconds:
            data, _ = stream.read(int(SAMPLE_RATE * chunk))
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > 0.01:
                started = True
                silent_for = 0.0
                frames.append(data.copy())
            elif started:
                frames.append(data.copy())
                silent_for += chunk * 1000
                if silent_for >= SILENCE_MS:
                    break
    if not frames:
        return None
    return np.concatenate(frames, axis=0).flatten()


def _transcribe(audio: np.ndarray) -> str:
    model = _load_whisper()
    segments, _ = model.transcribe(audio, language="en", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def _speak(text: str) -> bool:
    f5 = _load_f5()
    if f5 is None or not Path(REF_AUDIO).exists():
        STATE.log.append("Cannot speak: F5-TTS or ref clip missing")
        return False
    STATE.speaking = True
    try:
        # mute mic conceptually by flag; actual ducking left to OS
        wav, sr, _ = f5.infer(ref_file=REF_AUDIO, ref_text=Path(REF_TEXT).read_text() if Path(REF_TEXT).exists() else "", gen_text=text)
        sd.play(wav, sr)
        sd.wait()
        return True
    except Exception as e:
        STATE.log.append(f"speak error: {e}")
        return False
    finally:
        STATE.speaking = False


def _loop():
    """Background hands-free loop."""
    while STATE.running:
        if STATE.speaking:
            time.sleep(0.1)
            continue
        audio = _record_until_silence()
        if audio is None:
            continue
        text = _transcribe(audio)
        if not text:
            continue
        STATE.last_text = text
        STATE.log.append(f"heard: {text}")
        # The MCP tool `voice_listen_once` returns text to the agent;
        # the agent decides what to do. For a true auto-send we'd need a
        # Codex hook; v1 returns text for the agent to act on.


# ---- MCP surface ----
mcp = MCPServer("codex-local-voice")


@mcp.tool()
def voice_start() -> str:
    """Start the hands-free voice loop."""
    if STATE.running:
        return "already running"
    STATE.running = True
    threading.Thread(target=_loop, daemon=True).start()
    return "voice loop started"


@mcp.tool()
def voice_stop() -> str:
    """Stop the hands-free voice loop."""
    STATE.running = False
    return "voice loop stopped"


@mcp.tool()
def voice_status() -> str:
    """Current voice state."""
    return f"running={STATE.running} speaking={STATE.speaking} last='{STATE.last_text}'"


@mcp.tool()
def voice_listen_once(max_seconds: float = 30.0) -> str:
    """Listen for one utterance, transcribe, return text. Does not loop."""
    audio = _record_until_silence(max_seconds)
    if audio is None:
        return ""
    return _transcribe(audio)


@mcp.tool()
def voice_speak(text: str) -> str:
    """Speak text aloud via F5-TTS."""
    ok = _speak(text)
    return "spoken" if ok else "failed"


@mcp.tool()
def voice_transcribe_file(path: str) -> str:
    """Transcribe a local audio file."""
    audio, _ = sf.read(path)
    return _transcribe(np.asarray(audio, dtype=np.float32))


async def main():
    async with stdio_server() as (read, write):
        await mcp.run(read, write, mcp.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
