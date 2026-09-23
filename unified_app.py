"""
SIH Problem Statement 104: Real-Time Voice Cloning and Coercion Detection Pipeline
Unified Live Prototype with FastAPI + WebSocket High-Performance Web Dashboard
"""

import os
import sys
import json
import time
import queue
import threading
import asyncio
import numpy as np
import pyaudio
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------
# 1. System Constants & Risk Formula
# ---------------------------------------------------------
RATE = 16000
CHUNK_DUR = 0.5                      # 0.5s audio slices
BUFFER_DUR = 3.0                     # 3.0s sliding ring buffer
CHUNK_FRAMES = int(RATE * CHUNK_DUR)   # 8,000 samples
BUFFER_FRAMES = int(RATE * BUFFER_DUR) # 48,000 samples

WEIGHT_SYNTHETIC = 0.60
WEIGHT_COERCION = 0.40
CRITICAL_THRESHOLD = 0.70
WARNING_THRESHOLD = 0.40

MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"

SCAM_LEXICON = {
    "arrest": 0.40,
    "police": 0.35,
    "customs": 0.35,
    "cbi": 0.45,
    "ed": 0.40,
    "court": 0.30,
    "jail": 0.35,
    "warrant": 0.40,
    "upi": 0.35,
    "pin": 0.35,
    "otp": 0.40,
    "transfer": 0.30,
    "immediate": 0.25,
    "urgent": 0.25,
    "emergency": 0.30,
    "frozen": 0.35,
    "account": 0.20,
    "kyc": 0.35
}

# ---------------------------------------------------------
# 2. Pipeline State & Queues
# ---------------------------------------------------------
audio_queue = queue.Queue(maxsize=20)
active_websockets = []
latest_telemetry = {
    "p_synthetic": 0.0,
    "p_coercion": 0.0,
    "total_risk": 0.0,
    "status": "SAFE",
    "transcript": "",
    "flags": [],
    "circuit_breaker": False,
    "latency_ms": 0,
    "waveform": []
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Initializing ML Pipeline on {device}...")

# Load Acoustic Deepfake Detector
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
acoustic_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device)
acoustic_model.eval()
print("[*] Acoustic Classifier Loaded.")

# Load Vosk STT Engine (Graceful fallback if weights folder missing)
vosk_recognizer = None
try:
    from vosk import Model, KaldiRecognizer
    model_dir = "model"
    if not os.path.exists(model_dir):
        candidate_dirs = [d for d in os.listdir(".") if os.path.isdir(d) and "vosk" in d.lower()]
        if candidate_dirs:
            model_dir = candidate_dirs[0]

    if os.path.exists(model_dir):
        vosk_model = Model(model_dir)
        vosk_recognizer = KaldiRecognizer(vosk_model, RATE)
        print(f"[*] Vosk STT Engine initialized from '{model_dir}'.")
    else:
        print("[!] Warning: Vosk 'model/' directory not found. Coercion engine running in fallback mode.")
except Exception as e:
    print(f"[!] Vosk initialization notice: {e}")

# ---------------------------------------------------------
# 3. Audio Ingestion Worker (Producer Thread)
# ---------------------------------------------------------
def audio_capture_worker():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES
        )
    except Exception as e:
        print(f"[!] Audio stream open error: {e}")
        return

    print("[*] PyAudio background capture thread running.")
    while True:
        try:
            raw_bytes = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            chunk = np.frombuffer(raw_bytes, dtype=np.float32)
            if not audio_queue.full():
                audio_queue.put(chunk)
        except Exception:
            continue

# ---------------------------------------------------------
# 4. Dual-Engine Processing Loop (Consumer Thread)
# ---------------------------------------------------------
def ml_inference_worker():
    global latest_telemetry
    ring_buffer = np.zeros(BUFFER_FRAMES, dtype=np.float32)
    rolling_transcript = ""

    print("[*] Real-Time Inference Worker active.")

    while True:
        chunk = audio_queue.get()
        t_start = time.time()

        # Update 3.0s Ring Buffer
        ring_buffer[:-CHUNK_FRAMES] = ring_buffer[CHUNK_FRAMES:]
        ring_buffer[-CHUNK_FRAMES:] = chunk

        # -----------------------------------------------------
        # Branch A: Acoustic Synthetic Inference
        # -----------------------------------------------------
        p_synthetic = 0.0
        rms = np.sqrt(np.mean(ring_buffer**2))
        if rms > 0.005:  # Noise gate threshold
            norm_audio = (ring_buffer - np.mean(ring_buffer)) / (np.std(ring_buffer) + 1e-7)
            
            inputs = feature_extractor(
                norm_audio,
                sampling_rate=RATE,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(device)

            with torch.no_grad():
                logits = acoustic_model(input_values).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
                p_synthetic = float(probs[1])

        # -----------------------------------------------------
        # Branch B: Semantic Coercion Engine (Vosk)
        # -----------------------------------------------------
        p_coercion = 0.0
        matched_flags = []
        transcript_update = ""

        if vosk_recognizer is not None:
            chunk_int16 = (chunk * 32767).astype(np.int16).tobytes()
            if vosk_recognizer.AcceptWaveform(chunk_int16):
                res = json.loads(vosk_recognizer.Result())
                transcript_update = res.get("text", "").lower()
            else:
                partial = json.loads(vosk_recognizer.PartialResult())
                transcript_update = partial.get("partial", "").lower()

            if transcript_update:
                rolling_transcript = (rolling_transcript + " " + transcript_update).strip()
                if len(rolling_transcript.split()) > 40:
                    rolling_transcript = " ".join(rolling_transcript.split()[-40:])

        # Lexicon scoring over transcript
        search_space = (rolling_transcript + " " + transcript_update).lower()
        for word, score in SCAM_LEXICON.items():
            if word in search_space:
                p_coercion += score
                matched_flags.append(word.upper())
        p_coercion = min(p_coercion, 1.0)

        # -----------------------------------------------------
        # Risk Aggregation & Circuit Breaker Logic
        # -----------------------------------------------------
        total_risk = (WEIGHT_SYNTHETIC * p_synthetic) + (WEIGHT_COERCION * p_coercion)
        circuit_breaker = total_risk >= CRITICAL_THRESHOLD

        if circuit_breaker:
            status = "CRITICAL"
        elif total_risk >= WARNING_THRESHOLD:
            status = "WARNING"
        else:
            status = "SAFE"

        latency_ms = int((time.time() - t_start) * 1000)

        # Downsample waveform for lightweight visualization (64 points)
        downsample_step = len(chunk) // 64
        waveform_sample = chunk[::downsample_step].tolist()

        latest_telemetry = {
            "p_synthetic": round(p_synthetic * 100, 1),
            "p_coercion": round(p_coercion * 100, 1),
            "total_risk": round(total_risk * 100, 1),
            "status": status,
            "transcript": rolling_transcript if rolling_transcript else "[Listening for speech...]",
            "flags": list(set(matched_flags)),
            "circuit_breaker": circuit_breaker,
            "latency_ms": latency_ms,
            "waveform": waveform_sample
        }

# ---------------------------------------------------------
# 5. FastAPI & Glassmorphism Dashboard UI
# ---------------------------------------------------------
app = FastAPI(title="SIH-104 Real-Time Defense System")

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SIH-104: Dual-Engine Voice Cloning & Coercion Defense</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(18, 24, 38, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --safe: #10b981;
            --warning: #f59e0b;
            --critical: #ef4444;
            --accent: #3b82f6;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            background-image: radial-gradient(circle at 50% 0%, #172554 0%, transparent 60%);
            color: var(--text-main);
            font-family: 'Space Grotesk', sans-serif;
            min-height: 100vh;
            padding: 24px;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }
        .title h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.5px; }
        .title p { font-size: 0.85rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
        .badge {
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        .badge-safe { background: rgba(16, 185, 129, 0.2); color: var(--safe); border: 1px solid var(--safe); }
        .badge-warning { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
        .badge-critical { background: rgba(239, 68, 68, 0.25); color: var(--critical); border: 1px solid var(--critical); animation: pulse 1s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        
        .grid {
            display: grid;
            grid-template-columns: 1.2fr 1fr;
            gap: 20px;
        }
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 20px;
        }
        .card h2 {
            font-size: 0.95rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 16px;
        }
        .metric-hero {
            display: flex;
            align-items: baseline;
            gap: 12px;
            margin-bottom: 12px;
        }
        .metric-value {
            font-size: 3.5rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        .gauge-bar {
            height: 10px;
            background: rgba(255,255,255,0.08);
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 20px;
        }
        .gauge-fill {
            height: 100%;
            width: 0%;
            transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.3s;
        }
        .sub-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }
        .sub-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 14px;
        }
        .sub-card span { font-size: 0.75rem; color: var(--text-muted); }
        .sub-card h3 { font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
        
        .transcript-box {
            min-height: 110px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            line-height: 1.5;
            color: #cbd5e1;
            margin-bottom: 16px;
        }
        .flags-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .flag-tag {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.5);
            color: #fca5a5;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        canvas {
            width: 100%;
            height: 60px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            margin-top: 10px;
        }
        /* Circuit Breaker Overlay */
        #breaker-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            z-index: 100;
            justify-content: center;
            align-items: center;
        }
        .breaker-card {
            background: #1e1b2e;
            border: 2px solid var(--critical);
            border-radius: 16px;
            padding: 36px;
            text-align: center;
            max-width: 480px;
            box-shadow: 0 0 50px rgba(239, 68, 68, 0.4);
        }
        .breaker-card h2 { color: var(--critical); font-size: 1.8rem; margin-bottom: 12px; }
        .breaker-card p { font-size: 0.95rem; color: #cbd5e1; margin-bottom: 24px; line-height: 1.5; }
        .breaker-btn {
            background: var(--critical);
            color: white;
            border: none;
            padding: 12px 24px;
            font-weight: 700;
            border-radius: 8px;
            cursor: pointer;
            font-family: 'Space Grotesk', sans-serif;
        }
    </style>
</head>
<body>

    <div class="header">
        <div class="title">
            <h1>SIH-104: Dual-Engine Voice Cloning & Coercion Interceptor</h1>
            <p>Acoustic Model: Wav2Vec2 (Gary Stafford) | Semantic Engine: Vosk Kaldi</p>
        </div>
        <div id="system-badge" class="badge badge-safe">SYSTEM SECURE</div>
    </div>

    <div class="grid">
        <!-- Risk Aggregator Column -->
        <div class="card">
            <h2>Composite Real-Time Threat Score</h2>
            <div class="metric-hero">
                <div id="total-risk" class="metric-value">0.0%</div>
                <div style="color: var(--text-muted); font-size: 0.85rem;">Formula: 0.60×Acoustic + 0.40×Coercion</div>
            </div>
            <div class="gauge-bar">
                <div id="gauge-fill" class="gauge-fill"></div>
            </div>

            <div class="sub-metrics">
                <div class="sub-card">
                    <span>ACOUSTIC SYNTHETIC PROBABILITY</span>
                    <h3 id="p-synthetic">0.0%</h3>
                </div>
                <div class="sub-card">
                    <span>COERCION / EXTORTION PROBABILITY</span>
                    <h3 id="p-coercion">0.0%</h3>
                </div>
            </div>

            <div style="margin-top: 20px;">
                <h2>Audio Ingestion Oscilloscope</h2>
                <canvas id="waveform-canvas"></canvas>
            </div>
        </div>

        <!-- Coercion & Telemetry Column -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h2>Live Telephony Transcript Stream</h2>
                <span id="latency" style="font-size: 0.75rem; color: var(--text-muted); font-family: 'JetBrains Mono';">0 ms</span>
            </div>
            <div id="transcript-box" class="transcript-box">[Listening for speech...]</div>

            <h2>Triggered Scam Markers</h2>
            <div id="flags-container" class="flags-container">
                <span style="color: var(--text-muted); font-size: 0.8rem;">No coercive keywords detected.</span>
            </div>
        </div>
    </div>

    <!-- Circuit Breaker Modal -->
    <div id="breaker-overlay">
        <div class="breaker-card">
            <h2>CIRCUIT BREAKER TRIPPED</h2>
            <p>High-confidence synthetic voice cloning and coercive extortion pattern detected. Financial transactions and biometric authentication have been temporarily suspended.</p>
            <button class="breaker-btn" onclick="dismissBreaker()">Override / Step-Up Auth</button>
        </div>
    </div>

    <script>
        const ws = new WebSocket(`ws://${location.host}/ws`);
        const totalRiskEl = document.getElementById('total-risk');
        const gaugeFill = document.getElementById('gauge-fill');
        const pSynthEl = document.getElementById('p-synthetic');
        const pCoerceEl = document.getElementById('p-coercion');
        const systemBadge = document.getElementById('system-badge');
        const transcriptBox = document.getElementById('transcript-box');
        const flagsContainer = document.getElementById('flags-container');
        const latencyEl = document.getElementById('latency');
        const breakerOverlay = document.getElementById('breaker-overlay');
        const canvas = document.getElementById('waveform-canvas');
        const ctx = canvas.getContext('2d');

        let breakerDismissed = false;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            totalRiskEl.textContent = `${data.total_risk}%`;
            pSynthEl.textContent = `${data.p_synthetic}%`;
            pCoerceEl.textContent = `${data.p_coercion}%`;
            latencyEl.textContent = `${data.latency_ms} ms loop`;
            transcriptBox.textContent = data.transcript;

            // Gauge fill & color transitions
            gaugeFill.style.width = `${data.total_risk}%`;
            if (data.status === 'CRITICAL') {
                gaugeFill.style.backgroundColor = 'var(--critical)';
                systemBadge.className = 'badge badge-critical';
                systemBadge.textContent = 'CRITICAL THREAT';
                if (!breakerDismissed) breakerOverlay.style.display = 'flex';
            } else if (data.status === 'WARNING') {
                gaugeFill.style.backgroundColor = 'var(--warning)';
                systemBadge.className = 'badge badge-warning';
                systemBadge.textContent = 'CAUTION ADVISED';
                breakerOverlay.style.display = 'none';
                breakerDismissed = false;
            } else {
                gaugeFill.style.backgroundColor = 'var(--safe)';
                systemBadge.className = 'badge badge-safe';
                systemBadge.textContent = 'SYSTEM SECURE';
                breakerOverlay.style.display = 'none';
                breakerDismissed = false;
            }

            // Scam keywords
            if (data.flags.length > 0) {
                flagsContainer.innerHTML = data.flags.map(f => `<span class="flag-tag">${f}</span>`).join('');
            } else {
                flagsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">No coercive keywords detected.</span>';
            }

            // Draw oscilloscope
            drawWaveform(data.waveform);
        };

        function drawWaveform(points) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = 2;
            ctx.beginPath();
            const sliceWidth = canvas.width / (points.length || 1);
            let x = 0;
            for (let i = 0; i < points.length; i++) {
                const v = points[i] * 5;
                const y = (canvas.height / 2) + (v * (canvas.height / 2));
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
                x += sliceWidth;
            }
            ctx.stroke();
        }

        function dismissBreaker() {
            breakerDismissed = true;
            breakerOverlay.style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.get("/")
def get_dashboard():
    return HTMLResponse(content=HTML_DASHBOARD)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.send_text(json.dumps(latest_telemetry))
            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)

# ---------------------------------------------------------
# 6. Bootstrap Daemon
# ---------------------------------------------------------
if __name__ == "__main__":
    t_capture = threading.Thread(target=audio_capture_worker, daemon=True)
    t_capture.start()

    t_inference = threading.Thread(target=ml_inference_worker, daemon=True)
    t_inference.start()

    print("\n=======================================================")
    print(" SIH-104 DEFENSE PROTOCOL ACTIVE")
    print(" Dashboard: http://localhost:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")