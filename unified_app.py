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

# Audio stream and buffer settings
RATE = 16000
CHUNK_DUR = 0.5
BUFFER_DUR = 3.0
CHUNK_FRAMES = int(RATE * CHUNK_DUR)
BUFFER_FRAMES = int(RATE * BUFFER_DUR)

# Scoring weights and thresholds
WEIGHT_SYNTHETIC = 0.60
WEIGHT_COERCION = 0.40
CRITICAL_THRESHOLD = 0.70
WARNING_THRESHOLD = 0.40

MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"

# Keywords used to flag potential scams
SCAM_LEXICON = {
    "arrest": 0.35,
    "police": 0.35,
    "customs": 0.35,
    "cbi": 0.40,
    "ed": 0.35,
    "court": 0.30,
    "jail": 0.35,
    "warrant": 0.35,
    "upi": 0.35,
    "pin": 0.35,
    "otp": 0.40,
    "transfer": 0.30,
    "immediate": 0.25,
    "urgent": 0.25,
    "emergency": 0.25,
    "frozen": 0.35,
    "account": 0.20,
    "kyc": 0.30
}

# Queues and shared state
audio_queue = queue.Queue(maxsize=20)
active_websockets = []
reset_requested = False

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

# Load the acoustic model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
acoustic_model = AutoModelForAudioClassification.from_pretrained(MODEL_ID).to(device)
acoustic_model.eval()

# Load Vosk for speech recognition
vosk_model = None
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
except Exception as e:
    print(f"Vosk load error: {e}")

# Read microphone input in the background
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
        print(f"Mic error: {e}")
        return

    while True:
        try:
            raw_bytes = stream.read(CHUNK_FRAMES, exception_on_overflow=False)
            chunk = np.frombuffer(raw_bytes, dtype=np.float32)
            if not audio_queue.full():
                audio_queue.put(chunk)
        except Exception:
            continue

# Process audio chunks and run analysis
def ml_inference_worker():
    global latest_telemetry, reset_requested, vosk_recognizer
    ring_buffer = np.zeros(BUFFER_FRAMES, dtype=np.float32)
    rolling_transcript = ""

    while True:
        chunk = audio_queue.get()

        # Handle full system reset request
        if reset_requested:
            ring_buffer = np.zeros(BUFFER_FRAMES, dtype=np.float32)
            rolling_transcript = ""
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break
            if vosk_model is not None:
                from vosk import KaldiRecognizer
                vosk_recognizer = KaldiRecognizer(vosk_model, RATE)
            
            latest_telemetry = {
                "p_synthetic": 0.0,
                "p_coercion": 0.0,
                "total_risk": 0.0,
                "status": "SAFE",
                "transcript": "[System Reset: Listening for speech...]",
                "flags": [],
                "circuit_breaker": False,
                "latency_ms": 0,
                "waveform": []
            }
            reset_requested = False
            continue

        t_start = time.time()

        # Update sliding 3-second buffer
        ring_buffer[:-CHUNK_FRAMES] = ring_buffer[CHUNK_FRAMES:]
        ring_buffer[-CHUNK_FRAMES:] = chunk

        # Check for synthetic voice artifacts
        p_synthetic = 0.0
        rms = np.sqrt(np.mean(ring_buffer**2))
        
        if rms > 0.012:
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
                raw_synthetic_prob = float(probs[1])
                
                if raw_synthetic_prob < 0.35:
                    p_synthetic = raw_synthetic_prob * 0.5
                else:
                    p_synthetic = raw_synthetic_prob

        # Run offline speech-to-text
        p_coercion = 0.0
        matched_flags = []
        current_display = rolling_transcript

        if vosk_recognizer is not None:
            chunk_int16 = (chunk * 32767).astype(np.int16).tobytes()
            
            if vosk_recognizer.AcceptWaveform(chunk_int16):
                res = json.loads(vosk_recognizer.Result())
                final_text = res.get("text", "").strip()
                if final_text:
                    rolling_transcript = (rolling_transcript + " " + final_text).strip()
                    words = rolling_transcript.split()
                    if len(words) > 30:
                        rolling_transcript = " ".join(words[-30:])
                current_display = rolling_transcript
            else:
                partial = json.loads(vosk_recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text:
                    current_display = (rolling_transcript + " " + partial_text).strip()

        # Count keyword occurrences
        search_words = current_display.lower().split()
        for word, weight in SCAM_LEXICON.items():
            count = search_words.count(word)
            if count > 0:
                p_coercion += weight * count
                matched_flags.append(f"{word.upper()} (x{count})" if count > 1 else word.upper())

        p_coercion = min(p_coercion, 1.0)

        # Calculate combined threat score
        total_risk = (WEIGHT_SYNTHETIC * p_synthetic) + (WEIGHT_COERCION * p_coercion)
        circuit_breaker = total_risk >= CRITICAL_THRESHOLD

        if circuit_breaker:
            status = "CRITICAL"
        elif total_risk >= WARNING_THRESHOLD:
            status = "WARNING"
        else:
            status = "SAFE"

        latency_ms = int((time.time() - t_start) * 1000)

        # Downsample waveform for UI rendering
        downsample_step = len(chunk) // 64
        waveform_sample = chunk[::downsample_step].tolist()

        latest_telemetry = {
            "p_synthetic": round(p_synthetic * 100, 1),
            "p_coercion": round(p_coercion * 100, 1),
            "total_risk": round(total_risk * 100, 1),
            "status": status,
            "transcript": current_display if current_display else "[Listening for speech...]",
            "flags": list(set(matched_flags)),
            "circuit_breaker": circuit_breaker,
            "latency_ms": latency_ms,
            "waveform": waveform_sample
        }

# Web application and dashboard
app = FastAPI()

HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Voice Interceptor Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(18, 24, 38, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --safe: #10b981;
            --warning: #f59e0b;
            --critical: #ef4444;
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
        .title h1 { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.5px; }
        .title p { font-size: 0.82rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
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
            font-size: 0.88rem;
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
        .sub-card span { font-size: 0.72rem; color: var(--text-muted); font-weight: 600; }
        .sub-card h3 { font-size: 1.6rem; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
        
        .transcript-box {
            min-height: 110px;
            background: rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
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
            font-size: 0.72rem;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }
        canvas {
            width: 100%;
            height: 60px;
            background: rgba(0,0,0,0.25);
            border-radius: 8px;
            margin-top: 10px;
        }

        #breaker-overlay {
            display: none;
            position: fixed;
            inset: 0;
            background: rgba(10, 14, 26, 0.95);
            backdrop-filter: blur(12px);
            z-index: 999999;
            justify-content: center;
            align-items: center;
            user-select: none;
        }
        .breaker-card {
            background: #191428;
            border: 2px solid var(--critical);
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            max-width: 520px;
            box-shadow: 0 0 60px rgba(239, 68, 68, 0.5);
            animation: shake 0.5s ease;
        }
        @keyframes shake {
            0%, 100% { transform: translate(0, 0); }
            20%, 60% { transform: translate(-8px, 0); }
            40%, 80% { transform: translate(8px, 0); }
        }
        .breaker-card h2 { color: var(--critical); font-size: 1.8rem; margin-bottom: 10px; }
        .breaker-card p { font-size: 0.92rem; color: #cbd5e1; margin-bottom: 20px; line-height: 1.5; }
        .lockdown-status {
            font-family: 'JetBrains Mono', monospace;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.4);
            color: #fca5a5;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.8rem;
            margin-bottom: 24px;
            display: inline-block;
        }
        .breaker-btn {
            background: rgba(255, 255, 255, 0.08);
            color: #64748b;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 12px 24px;
            font-weight: 700;
            border-radius: 8px;
            cursor: not-allowed;
            font-family: 'Space Grotesk', sans-serif;
            transition: all 0.3s ease;
        }
        .breaker-btn.active {
            background: var(--critical);
            color: white;
            cursor: pointer;
            border: none;
            box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
        }
    </style>
</head>
<body>

    <div class="header">
        <div class="title">
            <h1>Voice Interceptor Dashboard</h1>
            <p>Real-time speech analysis & automated protection</p>
        </div>
        <div id="system-badge" class="badge badge-safe">SYSTEM SECURE</div>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Current Risk Level</h2>
            <div class="metric-hero">
                <div id="total-risk" class="metric-value">0.0%</div>
                <div style="color: var(--text-muted); font-size: 0.82rem;">Weighted: 60% Acoustic + 40% Keyword</div>
            </div>
            <div class="gauge-bar">
                <div id="gauge-fill" class="gauge-fill"></div>
            </div>

            <div class="sub-metrics">
                <div class="sub-card">
                    <span>SYNTHETIC VOICE SCORE</span>
                    <h3 id="p-synthetic">0.0%</h3>
                </div>
                <div class="sub-card">
                    <span>COERCION INTENT SCORE</span>
                    <h3 id="p-coercion">0.0%</h3>
                </div>
            </div>

            <div style="margin-top: 20px;">
                <h2>Microphone Input</h2>
                <canvas id="waveform-canvas"></canvas>
            </div>
        </div>

        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h2>Live Transcript</h2>
                <span id="latency" style="font-size: 0.72rem; color: var(--text-muted); font-family: 'JetBrains Mono';">0 ms loop</span>
            </div>
            <div id="transcript-box" class="transcript-box">[Listening for speech...]</div>

            <h2>Detected Keywords</h2>
            <div id="flags-container" class="flags-container">
                <span style="color: var(--text-muted); font-size: 0.8rem;">No threat keywords detected.</span>
            </div>
        </div>
    </div>

    <div id="breaker-overlay">
        <div class="breaker-card">
            <h2>SUSPICIOUS CALL DETECTED</h2>
            <p>Synthetic voice patterns and high-pressure phrases were detected. Sensitive actions are temporarily locked.</p>
            <div class="lockdown-status" id="lockdown-status">DEVICE LOCKED: 5s COOLDOWN ACTIVE</div>
            <br>
            <button id="breaker-btn" class="breaker-btn" disabled onclick="dismissBreaker()">
                Lockdown Active (5s)
            </button>
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
        const breakerBtn = document.getElementById('breaker-btn');
        const lockdownStatus = document.getElementById('lockdown-status');
        const canvas = document.getElementById('waveform-canvas');
        const ctx = canvas.getContext('2d');

        let isLockdownActive = false;
        let countdownTimer = null;

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);

            totalRiskEl.textContent = `${data.total_risk}%`;
            pSynthEl.textContent = `${data.p_synthetic}%`;
            pCoerceEl.textContent = `${data.p_coercion}%`;
            latencyEl.textContent = `${data.latency_ms} ms loop`;
            transcriptBox.textContent = data.transcript;

            gaugeFill.style.width = `${data.total_risk}%`;
            if (data.status === 'CRITICAL') {
                gaugeFill.style.backgroundColor = 'var(--critical)';
                systemBadge.className = 'badge badge-critical';
                systemBadge.textContent = 'CRITICAL THREAT';
                
                if (!isLockdownActive) {
                    triggerLockdown();
                }
            } else if (data.status === 'WARNING') {
                gaugeFill.style.backgroundColor = 'var(--warning)';
                systemBadge.className = 'badge badge-warning';
                systemBadge.textContent = 'CAUTION ADVISED';
            } else {
                gaugeFill.style.backgroundColor = 'var(--safe)';
                systemBadge.className = 'badge badge-safe';
                systemBadge.textContent = 'SYSTEM SECURE';
            }

            if (data.flags.length > 0) {
                flagsContainer.innerHTML = data.flags.map(f => `<span class="flag-tag">${f}</span>`).join('');
            } else {
                flagsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">No threat keywords detected.</span>';
            }

            drawWaveform(data.waveform);
        };

        function triggerLockdown() {
            isLockdownActive = true;
            breakerOverlay.style.display = 'flex';
            breakerBtn.disabled = true;
            breakerBtn.className = 'breaker-btn';
            
            let timeLeft = 5;
            breakerBtn.textContent = `Lockdown Active (${timeLeft}s)`;
            lockdownStatus.textContent = `DEVICE LOCKED (${timeLeft}s)`;

            clearInterval(countdownTimer);
            countdownTimer = setInterval(() => {
                timeLeft -= 1;
                if (timeLeft > 0) {
                    breakerBtn.textContent = `Lockdown Active (${timeLeft}s)`;
                    lockdownStatus.textContent = `DEVICE LOCKED (${timeLeft}s)`;
                } else {
                    clearInterval(countdownTimer);
                    breakerBtn.disabled = false;
                    breakerBtn.className = 'breaker-btn active';
                    breakerBtn.textContent = 'Acknowledge & Dismiss';
                    lockdownStatus.textContent = 'COOLDOWN EXPIRED - STEP-UP AUTH REQUIRED';
                }
            }, 1000);
        }

        async function dismissBreaker() {
            if (breakerBtn.disabled) return;
            
            // Send reset command to backend
            await fetch('/reset', { method: 'POST' });

            // Instantly clear UI elements
            breakerOverlay.style.display = 'none';
            totalRiskEl.textContent = '0.0%';
            pSynthEl.textContent = '0.0%';
            pCoerceEl.textContent = '0.0%';
            gaugeFill.style.width = '0%';
            gaugeFill.style.backgroundColor = 'var(--safe)';
            systemBadge.className = 'badge badge-safe';
            systemBadge.textContent = 'SYSTEM SECURE';
            transcriptBox.textContent = '[System Reset: Listening for speech...]';
            flagsContainer.innerHTML = '<span style="color: var(--text-muted); font-size: 0.8rem;">No threat keywords detected.</span>';

            // 5 second cooldown before breaker can trip again
            setTimeout(() => {
                isLockdownActive = false;
            }, 5000);
        }

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
    </script>
</body>
</html>
"""

@app.get("/")
def get_dashboard():
    return HTMLResponse(content=HTML_DASHBOARD)

# API endpoint to clear history and reset buffers
@app.post("/reset")
def reset_system():
    global reset_requested
    reset_requested = True
    return {"status": "ok"}

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

# Start background workers and launch server
if __name__ == "__main__":
    t_capture = threading.Thread(target=audio_capture_worker, daemon=True)
    t_capture.start()

    t_inference = threading.Thread(target=ml_inference_worker, daemon=True)
    t_inference.start()

    print("Server running at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")