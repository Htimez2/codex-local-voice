# Codex Local Voice

Hands-free local voice for Codex. Replaces OpenAI's built-in voice with a local MCP server.

- **STT**: faster-whisper (local, no cloud)
- **TTS**: F5-TTS (local, zero-shot cloning from a reference clip)
- **Transport**: MCP stdio (works in Codex CLI + app)
- **Mode**: Continuous / hands-free — one toggle, auto VAD, auto-send, auto-play, resume listening

## Why
OpenAI Codex voice burns the 5-hour window fast and has bad timing. This runs entirely on your machine.

## Quick start (Windows)

```powershell
# 1. Install Python 3.11+ and ffmpeg
winget install Python.Python.3.11
winget install Gyan.FFmpeg

# 2. Create venv + install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. (Optional) install F5-TTS
pip install f5-tts

# 4. Add a reference clip for cloning (your voice)
#    put a 5-10s clean WAV at assets/ref.wav and its transcript at assets/ref.txt

# 5. Register with Codex
codex mcp add local-voice -- .\.venv\Scripts\python.exe .\server.py
```

Then in Codex: enable the server, toggle voice on, talk. It listens, transcribes, sends your text, speaks replies, and keeps listening.

## Config

Edit `config.toml` or set env vars:

- `WHISPER_MODEL` (default: small) — tiny/base/small/medium/large-v3
- `WHISPER_DEVICE` (default: auto)
- `F5_REF_AUDIO` / `F5_REF_TEXT` — path to reference clip + transcript
- `SILENCE_MS` (default: 700) — how long of silence ends your turn
- `PLAYBACK_DEVICE` — output device index

## Tools exposed

- `voice_start` — begin hands-free loop
- `voice_stop` — end loop
- `voice_status` — is it listening/speaking?
- `voice_speak` — speak arbitrary text now (interrupts loop)
- `voice_transcribe_file` — one-shot file transcription

## Notes

- First run downloads Whisper + F5-TTS weights (~1.5GB).
- Needs a mic + speakers. Muting mic during playback prevents feedback.
- F5-TTS weights are CC-BY-NC; fine for personal use.
- Tested target: Windows 11 + NVIDIA GPU. CPU works but slower.
