import html
import io

import numpy as np
import soundfile as sf
import streamlit as st
from scipy.signal import resample_poly

MODEL_ID = "themohal/saraiki-whisper-small"
SR = 16_000

st.set_page_config(page_title="Saraiki Speech-to-Text (سرائیکی)", page_icon="🎙️")


@st.cache_resource(show_spinner="Loading model…")
def get_pipe():
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=MODEL_ID)


def read_audio(raw: bytes):
    """Decode audio bytes to mono float32 @ 16 kHz, entirely in memory."""
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        audio = resample_poly(audio, SR // g, sr // g).astype("float32")
    return audio


def transcribe(raw: bytes, timestamps: bool) -> dict:
    audio = read_audio(raw)
    out = get_pipe()({"array": audio, "sampling_rate": SR}, return_timestamps=timestamps)
    return {"text": out["text"].strip(), "chunks": out.get("chunks") or []}


def fmt(seconds) -> str:
    if seconds is None:
        return "--:--"
    m, s = divmod(float(seconds), 60)
    return f"{int(m):02d}:{s:05.2f}"


def show(raw: bytes, key: str, timestamps: bool):
    st.audio(raw)
    if st.button("Transcribe", type="primary", key=f"btn-{key}", use_container_width=True):
        with st.spinner("Transcribing…"):
            try:
                st.session_state[key] = transcribe(raw, timestamps)
            except Exception as exc:
                st.session_state.pop(key, None)
                st.error(f"Could not transcribe: {exc}")

    res = st.session_state.get(key)
    if res is None:
        return
    if not res["text"]:
        st.warning("No speech detected.")
        return

    st.markdown(
        "<div dir='rtl' style='font-size:1.2rem;line-height:2;text-align:right;"
        "background:rgba(128,128,128,0.12);border-radius:8px;padding:1rem;"
        "margin-bottom:1.25rem;overflow-wrap:break-word'>"
        f"{html.escape(res['text'])}</div>",
        unsafe_allow_html=True,
    )

    if res["chunks"]:
        with st.expander(f"Segments ({len(res['chunks'])})"):
            for c in res["chunks"]:
                start, end = (c.get("timestamp") or (None, None))[:2]
                st.markdown(f"`{fmt(start)} → {fmt(end)}`  {c.get('text', '').strip()}")

    st.download_button("Download .txt", res["text"], "transcript.txt", key=f"dl-{key}")


st.title("🎙️ Saraiki Speech-to-Text")
st.markdown(
    "Transcribe Saraiki speech into text using a Whisper-small model fine-tuned on "
    "Saraiki audio. Upload a recording or capture one with your mic — audio is "
    "decoded in memory and never written to disk."
)
st.caption(f"Model: [`{MODEL_ID}`](https://huggingface.co/{MODEL_ID}) · 244M params")

with st.sidebar:
    st.header("Settings")
    timestamps = st.toggle("Segment timestamps", value=False)
    st.caption(
        "Splits the transcript into timed segments so you can see **when** each "
        "phrase was spoken (`00:04 → 00:09`) instead of one unbroken block. "
        "Handy for locating a moment in a long recording or checking where the "
        "model misheard. Slightly slower, and timings are approximate."
    )

upload_tab, mic_tab = st.tabs(["📁 Upload", "🎤 Record"])

with upload_tab:
    f = st.file_uploader("Audio file", type=["wav", "flac", "ogg", "opus", "mp3"])
    if f:
        show(f.getvalue(), "upload", timestamps)

with mic_tab:
    rec = st.audio_input("Record, speak in Saraiki, then stop")
    if rec:
        show(rec.getvalue(), "mic", timestamps)
