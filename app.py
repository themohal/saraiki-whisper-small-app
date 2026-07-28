"""Saraiki Speech-to-Text — Streamlit app around themohal/saraiki-whisper-small.

Everything is processed in memory: uploaded/recorded bytes are decoded with
soundfile/librosa straight from a BytesIO buffer, never written to disk.
"""

from __future__ import annotations

import io
import time

import numpy as np
import streamlit as st

MODEL_ID = "themohal/saraiki-whisper-small"
TARGET_SR = 16_000
MAX_MINUTES = 30

st.set_page_config(page_title="Saraiki Speech-to-Text", page_icon="🎙️", layout="centered")


# --------------------------------------------------------------------------- model


@st.cache_resource(show_spinner=False)
def load_pipeline(model_id: str):
    """Build the ASR pipeline once per process and keep it warm."""
    import torch
    from transformers import pipeline

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,
        stride_length_s=5,
    )
    return asr, device


# --------------------------------------------------------------------------- audio


def decode_audio(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes to a mono float32 array at 16 kHz.

    Tries libsndfile first (wav/flac/ogg/mp3), then falls back to audioread via
    librosa for the containers libsndfile cannot open (m4a, some mp4/webm).
    """
    import librosa
    import soundfile as sf

    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
    except Exception:
        audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        audio = np.asarray(audio, dtype="float32")

    if audio.size == 0:
        raise ValueError("The audio stream is empty.")

    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    return audio, TARGET_SR


def transcribe(raw: bytes, *, timestamps: bool) -> dict:
    asr, _ = load_pipeline(MODEL_ID)
    audio, sr = decode_audio(raw)

    duration = len(audio) / sr
    if duration > MAX_MINUTES * 60:
        raise ValueError(f"Audio is {duration / 60:.1f} min; the limit is {MAX_MINUTES} min.")

    started = time.perf_counter()
    result = asr(
        {"array": audio, "sampling_rate": sr},
        return_timestamps=timestamps,
        generate_kwargs={"task": "transcribe"},
    )
    elapsed = time.perf_counter() - started

    return {
        "text": (result.get("text") or "").strip(),
        "chunks": result.get("chunks") or [],
        "duration": duration,
        "elapsed": elapsed,
    }


def fmt_ts(value) -> str:
    if value is None:
        return "--:--"
    minutes, seconds = divmod(float(value), 60)
    return f"{int(minutes):02d}:{seconds:05.2f}"


# --------------------------------------------------------------------------- ui


def render_result(res: dict, key: str) -> None:
    if not res["text"]:
        st.warning("No speech detected in this audio.")
        return

    st.subheader("Transcript")
    st.markdown(
        f"<div dir='rtl' style='font-size:1.25rem;line-height:2.1;text-align:right;"
        f"background:#f6f6f8;border-radius:8px;padding:1rem 1.2rem;'>{res['text']}</div>",
        unsafe_allow_html=True,
    )

    speed = res["duration"] / res["elapsed"] if res["elapsed"] else 0
    a, b, c = st.columns(3)
    a.metric("Audio length", f"{res['duration']:.1f} s")
    b.metric("Processing", f"{res['elapsed']:.1f} s")
    c.metric("Realtime factor", f"{speed:.1f}×")

    if res["chunks"]:
        with st.expander(f"Segments ({len(res['chunks'])})"):
            for chunk in res["chunks"]:
                start, end = (chunk.get("timestamp") or (None, None))[:2]
                st.markdown(f"`{fmt_ts(start)} → {fmt_ts(end)}`  {chunk.get('text', '').strip()}")

            srt = []
            for i, chunk in enumerate(res["chunks"], start=1):
                start, end = (chunk.get("timestamp") or (0, 0))[:2]
                srt.append(
                    f"{i}\n{_srt_time(start)} --> {_srt_time(end)}\n{chunk.get('text', '').strip()}\n"
                )
            st.download_button(
                "Download .srt", "\n".join(srt), file_name="transcript.srt",
                mime="text/plain", key=f"srt-{key}",
            )

    st.download_button(
        "Download transcript (.txt)", res["text"], file_name="transcript.txt",
        mime="text/plain", key=f"txt-{key}",
    )


def _srt_time(value) -> str:
    total = float(value or 0)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{millis:03d}"


def run(raw: bytes, key: str, timestamps: bool) -> None:
    with st.spinner("Transcribing…"):
        try:
            res = transcribe(raw, timestamps=timestamps)
        except Exception as exc:  # surface decode/inference failures in the UI
            st.error(f"Could not transcribe: {exc}")
            return
    st.session_state[f"result-{key}"] = res


st.title("🎙️ Saraiki Speech-to-Text")
st.caption(f"Fine-tuned Whisper-small · `{MODEL_ID}` · audio never touches disk")

with st.sidebar:
    st.header("Settings")
    timestamps = st.toggle("Segment timestamps", value=False)
    st.divider()
    if st.button("Load model now", use_container_width=True):
        with st.spinner("Downloading / loading weights…"):
            _, device = load_pipeline(MODEL_ID)
        st.success(f"Ready on **{device}**")
    st.caption(
        "First run downloads ~1 GB of weights from Hugging Face and caches them "
        "locally. Later runs start instantly."
    )

upload_tab, mic_tab = st.tabs(["📁 Upload a file", "🎤 Record with mic"])

with upload_tab:
    uploaded = st.file_uploader(
        "Audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg", "opus", "webm", "mp4"],
        help="Decoded in memory — nothing is saved to disk.",
    )
    if uploaded is not None:
        raw = uploaded.getvalue()
        st.audio(raw)
        st.caption(f"{uploaded.name} · {len(raw) / 1e6:.2f} MB")
        if st.button("Transcribe file", type="primary", use_container_width=True):
            run(raw, "upload", timestamps)
    if "result-upload" in st.session_state:
        render_result(st.session_state["result-upload"], "upload")

with mic_tab:
    recorded = st.audio_input("Press record, speak in Saraiki, then stop")
    if recorded is not None:
        raw = recorded.getvalue()
        st.caption(f"Recording captured · {len(raw) / 1e6:.2f} MB")
        if st.button("Transcribe recording", type="primary", use_container_width=True):
            run(raw, "mic", timestamps)
    if "result-mic" in st.session_state:
        render_result(st.session_state["result-mic"], "mic")
