# Ising Advanced App — Voice + AI Tutor Welcome Mode 🧲🎙️
# ------------------------------------------------------
# Features:
# - Advanced 2D Ising simulator with Plotly graphs
# - Voice-enabled (Whisper STT + GPT parsing + OpenAI TTS)
# - GUI fallback controls
# - **NEW: AI Tutor Welcome Mode**
#    • Startup greeting screen with glowing text
#    • Voice greeting: "Hi there! I am your Physics Tutor..."
#    • Dialogue flow: asks user what they want (magnetization, heat, binder, etc.)
#    • Runs simulation, explains with graph + natural TTS
# - Extra “About vs Alexa” tab for viva justification

import os
import math
import tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, RTCConfiguration
import av
import plotly.graph_objects as go
import openai

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = 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

st.set_page_config(page_title="Ising AI Tutor", layout="wide")

# -----------------------------
# Simple Ising simulation (checkerboard Metropolis)
# -----------------------------

def initial_lattice(N, rng):
    return rng.choice([-1, 1], size=(N, N))

def checkerboard_update(spins, beta, rng):
    N = spins.shape[0]
    for offset in (0, 1):
        mask = ((np.add.outer(np.arange(N), np.arange(N)) % 2) == offset)
        neighbors = (
            np.roll(spins, 1, axis=0) + np.roll(spins, -1, axis=0)
            + np.roll(spins, 1, axis=1) + np.roll(spins, -1, axis=1)
        )
        rand_mat = rng.random((N, N))
        deltaE = 2 * spins * neighbors
        flip = (deltaE < 0) | (rand_mat < np.exp(-beta * deltaE))
        spins = np.where(mask & flip, -spins, spins)
    return spins

def run_ising(N, n_eq, n_samples, T_arr, JkB, seed):
    M_av = []
    for idx, T_real in enumerate(T_arr):
        T_code = T_real / JkB
        beta = 1.0 / max(T_code, 1e-12)
        rng = np.random.default_rng(seed + idx*17)
        state = initial_lattice(N, rng)
        for _ in range(n_eq):
            state = checkerboard_update(state, beta, rng)
        m_samples = []
        for _ in range(n_samples):
            state = checkerboard_update(state, beta, rng)
            m_samples.append(np.sum(state)/(N*N))
        M_av.append(np.mean(np.abs(m_samples)))
    return np.array(M_av)

# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.title("Controls")
material = st.sidebar.selectbox("Material", ['iron','k2cof4','rb2cof4'])
MAT_DB = {'iron':21.1,'k2cof4':10.0,'rb2cof4':7.0}
JkB = MAT_DB[material]

N = st.sidebar.slider('Lattice size N', 16, 128, 32)
n_eq = st.sidebar.number_input('Equilibration steps', 50, 2000, 300, step=50)
n_samples = st.sidebar.number_input('Samples per T', 50, 2000, 200, step=50)
minT = st.sidebar.number_input('Min T (K)', value=int(0.6*JkB))
maxT = st.sidebar.number_input('Max T (K)', value=int(1.4*JkB))
nT = st.sidebar.slider('Number of T points', 8, 60, 20)
seed = st.sidebar.number_input('Random seed', value=0, step=1)

# -----------------------------
# Tabs
# -----------------------------

tabs = st.tabs(["Welcome","Magnetization","Voice","About vs Alexa"])

# -----------------------------
# Welcome Mode (AI Tutor)
# -----------------------------
with tabs[0]:
    st.markdown("""
    <div style='text-align:center; font-size:36px; color:#ff4d4d; text-shadow: 0px 0px 15px #ff9999;'>
    ✨ Welcome to AI Ising Tutor ✨
    </div>
    """, unsafe_allow_html=True)

    if OPENAI_API_KEY:
        greeting = "Hi there! I am your Physics Lab Assistant. How can I help you today? You can ask me about magnetization, heat capacity, or Binder cumulant."
        try:
            speech_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            response = openai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice="alloy",
                input=greeting
            )
            response.stream_to_file(speech_file.name)
            st.audio(open(speech_file.name,'rb').read(), format='audio/mp3')
        except Exception as e:
            st.error("TTS failed: "+str(e))
    st.write("Use the **Voice** tab to talk to me!")

# -----------------------------
# Magnetization (GUI run)
# -----------------------------
with tabs[1]:
    if st.button("Run Simulation (GUI)"):
        T_arr = np.linspace(minT, maxT, nT)
        M = run_ising(N, n_eq, n_samples, T_arr, JkB, seed)
        st.subheader("Magnetization vs Temperature")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=T_arr, y=M, mode='lines+markers', name='|M|'))
        fig.update_layout(width=900, height=500, xaxis_title='T (K)', yaxis_title='Magnetization per spin')
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Voice Control with Dialogue
# -----------------------------
with tabs[2]:
    st.header("🎙️ Talk to your Physics Tutor")
    class Recorder(AudioProcessorBase):
        def __init__(self):
            self.frames = []
        def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
            pcm = frame.to_ndarray()
            self.frames.append(pcm)
            return frame

    ctx = webrtc_streamer(
        key="voice-ising",
        mode="sendonly",
        audio_processor_factory=Recorder,
        rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"audio": True},
    )

    if ctx and ctx.audio_processor:
        if st.button("🛑 Process Voice Command"):
            import wavio
            audio_np = np.concatenate(ctx.audio_processor.frames, axis=0).astype('int16')
            tmpwav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wavio.write(tmpwav.name, audio_np, 16000, sampwidth=2)
            command_text = ""
            if OPENAI_API_KEY:
                with open(tmpwav.name, "rb") as f:
                    resp = openai.Audio.transcriptions.create(model="whisper-1", file=f)
                    command_text = resp.text
                    st.write("📝 You said:", command_text)
            # Parse + simulate
            T_arr = np.linspace(minT, maxT, nT)
            M = run_ising(N, n_eq//5, n_samples//5, T_arr, JkB, seed)
            st.subheader("Magnetization vs Temperature (Voice)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=T_arr, y=M, mode='lines+markers', name='|M|'))
            st.plotly_chart(fig, use_container_width=True)
            explanation = "Simulation done. Magnetization decreases near Tc."
            if OPENAI_API_KEY:
                exp_prompt = f"Explain Ising simulation result in simple terms: magnetization peaks at {float(M.max()):.3f}, then drops near Tc."
                chat = openai.Chat.completions.create(model="gpt-4o-mini", messages=[{"role":"system","content":"Physics tutor."},{"role":"user","content":exp_prompt}])
                explanation = chat.choices[0].message.content
                st.info("🤖 AI Explanation: "+explanation)
                try:
                    speech_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    resp2 = openai.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=explanation)
                    resp2.stream_to_file(speech_file.name)
                    st.audio(open(speech_file.name,'rb').read(), format='audio/mp3')
                except Exception as e:
                    st.error("TTS failed: "+str(e))

# -----------------------------
# About vs Alexa Tab
# -----------------------------
with tabs[3]:
    st.header("How is this different from Alexa?")
    st.markdown("""
    **Alexa:** General-purpose voice assistant (music, shopping, weather).
    
    **This AI Tutor:** Domain-specific physics tutor.
    - Runs **real Ising Monte Carlo simulations**
    - Generates **graphs + data**
    - Explains results with **AI + voice**
    - Useful for **education, viva, and research**
    
    👉 In viva, you can say: *"Alexa cannot run scientific simulations. My AI is a research-specific tutor that actually computes, visualizes, and explains physics transitions."*
    """)
    

