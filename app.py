"""
Driver Wellness AI — Hugging Face Spaces entry point (Gradio).

This file is intentionally thin. All model/adapter/fusion/orchestrator logic
lives in `wellness_core.py` (a faithful port of the Colab notebook, with the
updated Option-A exponential risk fusion + score smoothing). `app.py` only:

  1. builds the module manager once at startup,
  2. exposes `analyze_video(path)` for the recorded-video tab, and
  3. exposes a live webcam streaming tab that reuses the same orchestrator,
     one browser-streamed frame at a time.

Public surface used from wellness_core.py:
    build_manager()                         -> loads all 5 models, returns manager
    run_recorded_video(manager, video_path) -> (annotated_video_path, summary_dict)
    start_live_session(fps=None)            -> reset orchestrator for a new live run
    process_live_frame(frame_rgb, fps=None) -> (annotated_rgb, summary_text)
    stop_live_session()                     -> final summary string
"""

import re
import traceback

import gradio as gr

# ----------------------------------------------------------------------
# `spaces` is only present on Hugging Face ZeroGPU hardware. Guard the
# import so the app also runs on a plain CPU/GPU Space (or locally) where
# the package may be absent — the decorator then becomes a no-op.
# ----------------------------------------------------------------------
try:
    import spaces  # noqa: F401

    def gpu(duration=120):
        return spaces.GPU(duration=duration)
except Exception:  # pragma: no cover
    def gpu(duration=120):
        def _wrap(fn):
            return fn
        return _wrap

# ----------------------------------------------------------------------
# Optional: pull weights from a separate HF model repo instead of shipping
# them inside this Space. Uncomment and set MODEL_REPO to use this path.
# ----------------------------------------------------------------------
# import os
# from huggingface_hub import hf_hub_download
# MODEL_REPO = "your-username/driver-wellness-weights"
# os.makedirs("models", exist_ok=True)
# for fname in [
#     "Video_Fatigue.pth", "Landmark_Fatigue.pt", "Driver_Activity.pth",
#     "Smoking_And_Drinking.pt", "SeatBelt_And_Phone.pt",
#     "m4_normalization_stats_ws45.csv", "face_landmarker.task",
# ]:
#     hf_hub_download(repo_id=MODEL_REPO, filename=fname, local_dir="models")

# ----------------------------------------------------------------------
# Build the manager ONCE (loading 5 models is expensive — never per request)
# ----------------------------------------------------------------------
import wellness_core as core

print("Loading models... (first startup can take a minute)")
MANAGER = core.build_manager()
print("Models loaded.")


# ----------------------------------------------------------------------
# Theming — "Cockpit HUD": dark glassmorphism, neon indigo/cyan/rose glow,
# animated aurora backdrop, Orbitron display font for numerals/headings.
# ----------------------------------------------------------------------
THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="rose",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

HEAD_HTML = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&display=swap" rel="stylesheet">
"""

CUSTOM_CSS = """
:root {
    --dw-bg:       #05070f;
    --dw-panel:    rgba(18, 22, 38, 0.60);
    --dw-panel-2:  rgba(18, 22, 38, 0.85);
    --dw-border:   rgba(120, 130, 255, 0.20);
    --dw-indigo:   #6366f1;
    --dw-cyan:     #22d3ee;
    --dw-rose:     #f43f5e;
    --dw-text:     #e6e8f5;
    --dw-text-dim: #9aa1c4;

    --body-background-fill: var(--dw-bg) !important;
    --background-fill-primary: var(--dw-panel) !important;
    --background-fill-secondary: var(--dw-panel-2) !important;
    --body-text-color: var(--dw-text) !important;
    --body-text-color-subdued: var(--dw-text-dim) !important;
    --border-color-primary: var(--dw-border) !important;
    --block-background-fill: var(--dw-panel) !important;
    --block-border-color: var(--dw-border) !important;
    --block-label-text-color: var(--dw-cyan) !important;
    --input-background-fill: rgba(8, 10, 20, 0.65) !important;
    --button-primary-background-fill: linear-gradient(90deg, var(--dw-indigo), var(--dw-cyan)) !important;
    --button-primary-background-fill-hover: linear-gradient(90deg, var(--dw-cyan), var(--dw-indigo)) !important;
    --button-primary-text-color: #05070f !important;
    --button-secondary-background-fill: rgba(255,255,255,0.06) !important;
}

body, .gradio-container {
    background: var(--dw-bg) !important;
    color: var(--dw-text) !important;
}

/* Slowly drifting aurora glow behind everything */
body::before {
    content: ""; position: fixed; inset: -20%; z-index: -2; pointer-events: none;
    background:
        radial-gradient(circle at 15% 20%, rgba(99,102,241,0.30), transparent 40%),
        radial-gradient(circle at 85% 15%, rgba(244,63,94,0.24), transparent 42%),
        radial-gradient(circle at 50% 90%, rgba(34,211,238,0.20), transparent 45%);
    filter: blur(40px);
    animation: dw-drift 22s ease-in-out infinite alternate;
}
@keyframes dw-drift {
    0%   { transform: translate(0,0) scale(1); }
    50%  { transform: translate(-3%, 2%) scale(1.08); }
    100% { transform: translate(2%, -3%) scale(1.03); }
}
/* Faint scanline texture for the HUD feel */
body::after {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background: repeating-linear-gradient(
        to bottom, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px,
        transparent 1px, transparent 3px
    );
}

.gradio-container { max-width: 1220px !important; margin: 0 auto !important; }

/* ---------------- Header ---------------- */
#dw-header { text-align: center; padding: 30px 12px 6px; }
#dw-header h1 {
    font-family: 'Orbitron', sans-serif; font-weight: 800; letter-spacing: 3px;
    font-size: 2.5rem; text-transform: uppercase; margin-bottom: 6px;
    background: linear-gradient(90deg, var(--dw-indigo), var(--dw-cyan), var(--dw-rose), var(--dw-indigo));
    background-size: 300% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: dw-gradient-shift 6s linear infinite;
    filter: drop-shadow(0 0 18px rgba(99,102,241,0.45));
}
@keyframes dw-gradient-shift { to { background-position: 300% center; } }
#dw-header p { color: var(--dw-text-dim); font-size: 0.95rem; letter-spacing: 0.4px; margin-top: 0; }

/* ---------------- Status chips ---------------- */
#dw-status { text-align: center; margin-bottom: 16px; }
.status-chip {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 15px; margin: 4px; border-radius: 999px;
    font-size: 0.8rem; font-weight: 700; letter-spacing: 0.3px;
    border: 1px solid var(--dw-border);
    background: var(--dw-panel); backdrop-filter: blur(10px);
    transition: transform 0.2s ease;
}
.status-chip:hover { transform: translateY(-2px); }
.status-dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.status-ok .status-dot { background: #22c55e; box-shadow: 0 0 8px 2px #22c55e; animation: dw-pulse-green 2s ease-in-out infinite; }
.status-missing .status-dot { background: #ef4444; box-shadow: 0 0 8px 2px #ef4444; animation: dw-pulse-red 1.2s ease-in-out infinite; }
.status-missing { opacity: 0.8; }
@keyframes dw-pulse-green { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes dw-pulse-red   { 0%,100% { opacity: 1; } 50% { opacity: 0.25; } }

/* ---------------- Glass cards ---------------- */
.dw-card {
    border-radius: 18px !important;
    background: var(--dw-panel) !important;
    border: 1px solid var(--dw-border) !important;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
    transition: box-shadow 0.3s ease;
}
.dw-card:hover { box-shadow: 0 8px 36px rgba(99,102,241,0.25); }

.dw-live-frame { animation: dw-glow-cycle 5s ease-in-out infinite; }
@keyframes dw-glow-cycle {
    0%,100% { box-shadow: 0 0 28px 3px rgba(99,102,241,0.55); }
    33%     { box-shadow: 0 0 28px 3px rgba(34,211,238,0.55); }
    66%     { box-shadow: 0 0 28px 3px rgba(244,63,94,0.55); }
}

/* ---------------- Tabs ---------------- */
.tab-nav button { font-weight: 700 !important; letter-spacing: 0.4px; border-radius: 12px 12px 0 0 !important; }
.tab-nav button.selected { color: var(--dw-cyan) !important; box-shadow: inset 0 -2px 0 var(--dw-cyan); }

/* ---------------- Buttons ---------------- */
button.primary {
    border: none !important; font-weight: 800 !important; letter-spacing: 0.6px;
    text-transform: uppercase; box-shadow: 0 0 20px rgba(99,102,241,0.5) !important;
    animation: dw-btn-pulse 2.4s ease-in-out infinite;
}
@keyframes dw-btn-pulse {
    0%,100% { box-shadow: 0 0 16px rgba(99,102,241,0.45); }
    50%     { box-shadow: 0 0 26px rgba(34,211,238,0.65); }
}
button.stop {
    background: linear-gradient(90deg, #f43f5e, #f59e0b) !important;
    border: none !important; color: #05070f !important; font-weight: 800 !important;
    text-transform: uppercase; letter-spacing: 0.6px;
    box-shadow: 0 0 20px rgba(244,63,94,0.55) !important;
}

/* ---------------- Live banner ---------------- */
#dw-live-banner {
    border-radius: 14px; padding: 12px 18px; margin-top: 8px;
    background: linear-gradient(90deg, rgba(99,102,241,0.12), rgba(244,63,94,0.12));
    border: 1px solid var(--dw-border); font-size: 0.9rem; color: var(--dw-text-dim);
}

/* ---------------- Risk gauge ---------------- */
.dw-gauge-wrap { position: relative; width: 240px; margin: 14px auto 6px; }
.dw-gauge-track { fill: none; stroke: rgba(255,255,255,0.08); stroke-width: 14; stroke-linecap: round; }
.dw-gauge-fill  { fill: none; stroke-width: 14; stroke-linecap: round; transition: stroke-dasharray 0.5s ease; }
.dw-gauge-center { position: absolute; left: 50%; bottom: 4px; transform: translateX(-50%); text-align: center; }
.dw-gauge-score { font-family: 'Orbitron', sans-serif; font-size: 2.6rem; font-weight: 800; line-height: 1; }
.dw-gauge-label { font-size: 0.72rem; letter-spacing: 2.5px; text-transform: uppercase; color: var(--dw-text-dim); margin-top: 4px; }

footer { display: none !important; }
"""


def _status_chips_html() -> str:
    """Renders a chip per model — pulsing green dot if loaded, pulsing red if missing."""
    chips = []
    for row in core.system_status:
        ok = row["Status"] == "Available"
        css_class = "status-ok" if ok else "status-missing"
        icon = "✓" if ok else "✕"
        chips.append(
            f'<span class="status-chip {css_class}">'
            f'<span class="status-dot"></span>{row["Module"]} {icon}</span>'
        )
    return f'<div id="dw-status">{"".join(chips)}</div>'


def _risk_gauge_html(score, risk_level: str = "N/A") -> str:
    """Renders a glowing semicircular SVG risk gauge (0-100), color-coded by risk bucket."""
    try:
        score = max(0.0, min(100.0, float(score)))
    except (TypeError, ValueError):
        score = 0.0
    if score <= 25:
        color, glow = "#22c55e", "34,197,94"
    elif score <= 50:
        color, glow = "#f59e0b", "245,158,11"
    elif score <= 75:
        color, glow = "#f97316", "249,115,22"
    else:
        color, glow = "#ef4444", "239,68,68"
    arc = "M15 105 A85 85 0 0 1 185 105"
    return f"""
<div class="dw-gauge-wrap">
  <svg viewBox="0 0 200 115">
    <path d="{arc}" class="dw-gauge-track" pathLength="100"></path>
    <path d="{arc}" class="dw-gauge-fill" pathLength="100"
          stroke="{color}" stroke-dasharray="{score:.1f} {100 - score:.1f}"
          style="filter: drop-shadow(0 0 10px rgba({glow},0.85));"></path>
  </svg>
  <div class="dw-gauge-center">
    <div class="dw-gauge-score" style="color:{color}; text-shadow:0 0 20px rgba({glow},0.9);">{score:.0f}</div>
    <div class="dw-gauge-label">{str(risk_level).upper()}</div>
  </div>
</div>
""".strip()


_SCORE_RE = re.compile(r"Overall Score\s*:\s*([\d.]+)")
_LEVEL_RE = re.compile(r"Risk Level\s*:\s*([^\n]+)")


def _parse_score_and_level(summary_text: str):
    """Pulls the numeric score + risk-level label back out of the plain-text summary."""
    score_match = _SCORE_RE.search(summary_text or "")
    level_match = _LEVEL_RE.search(summary_text or "")
    score = float(score_match.group(1)) if score_match else 0.0
    level = level_match.group(1).strip() if level_match else "N/A"
    return score, level


# ======================================================================
# Recorded-video tab
# ======================================================================
@gpu(duration=120)
def analyze_video(video_path):
    if not video_path:
        return None, "Please upload a video first.", _risk_gauge_html(0.0, "No Input")
    try:
        annotated_path, summary = core.run_recorded_video(MANAGER, video_path)
        lines = [f"{k}: {v}" for k, v in summary.items()]
        score_text = str(summary.get("Overall Score", "0 / 100"))
        score = float(score_text.split("/")[0].strip() or 0.0)
        risk_level = str(summary.get("Risk Level", "N/A"))
        return annotated_path, "\n".join(lines), _risk_gauge_html(score, risk_level)
    except Exception:
        return None, "Error during analysis:\n" + traceback.format_exc(), _risk_gauge_html(0.0, "Error")


# ======================================================================
# Live webcam tab (browser streams frames -> same orchestrator)
# ======================================================================
def live_start():
    """Reset the orchestrator for a fresh live session."""
    core.start_live_session()
    return None, "Live session started — grant camera access and stay in frame.", _risk_gauge_html(0.0, "Standby")


def live_stream(frame):
    """Handle ONE streamed webcam frame; return (annotated_rgb, summary_text, gauge_html)."""
    if frame is None:
        return None, "Waiting for webcam frames...", _risk_gauge_html(0.0, "Waiting")
    try:
        annotated, summary_text = core.process_live_frame(frame)
        score, risk_level = _parse_score_and_level(summary_text)
        return annotated, summary_text, _risk_gauge_html(score, risk_level)
    except Exception:
        return frame, "Error during live analysis:\n" + traceback.format_exc(), _risk_gauge_html(0.0, "Error")


def live_stop():
    """Finalise the live session and return the fused summary + a final gauge."""
    try:
        summary_text = core.stop_live_session()
        score, risk_level = _parse_score_and_level(summary_text)
        return summary_text, _risk_gauge_html(score, risk_level)
    except Exception:
        return "Error finalising session:\n" + traceback.format_exc(), _risk_gauge_html(0.0, "Error")


with gr.Blocks(title="Driver Wellness AI") as demo:
    gr.HTML(
        '<div id="dw-header">'
        "<h1>🚗 Driver Wellness AI</h1>"
        "<p>Five models fused into one live wellness/risk score — "
        "Common Driver Risk Score Framework (Option A, exponential)</p>"
        "</div>"
    )
    gr.HTML(_status_chips_html())

    with gr.Tab("📹 Recorded video"):
        gr.Markdown("Upload a short driving clip to get a fused, annotated analysis.")
        with gr.Row():
            with gr.Column():
                inp = gr.Video(label="Driving clip", sources=["upload"], elem_classes=["dw-card"])
                btn = gr.Button("▶ Analyze", variant="primary", size="lg")
            with gr.Column():
                out_video = gr.Video(label="Processed Video", elem_classes=["dw-card"])
                out_gauge = gr.HTML(_risk_gauge_html(0.0, "Awaiting Analysis"), elem_classes=["dw-card"])
                out_text = gr.Textbox(label="📋 Session summary", lines=14, elem_classes=["dw-card"])
        btn.click(analyze_video, inputs=inp, outputs=[out_video, out_text, out_gauge])

    with gr.Tab("🔴 Live webcam"):
        gr.HTML(
            '<div id="dw-live-banner">Click <b>Start</b>, allow camera access, and the '
            "annotated feed + live risk gauge update continuously. Click <b>Finish</b> "
            "for the session summary.<br><i>On CPU hardware this runs slower — upgrade to "
            "GPU for smoother real-time performance.</i></div>"
        )
        with gr.Row():
            with gr.Column():
                cam = gr.Image(
                    label="Webcam",
                    sources=["webcam"],
                    streaming=True,
                    type="numpy",
                    elem_classes=["dw-card", "dw-live-frame"],
                )
                with gr.Row():
                    start_btn = gr.Button("▶ Start / Reset session", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹ Finish & summarize", variant="stop", size="lg")
            with gr.Column():
                live_out = gr.Image(label="Annotated live feed", type="numpy", elem_classes=["dw-card", "dw-live-frame"])
                live_gauge = gr.HTML(_risk_gauge_html(0.0, "Standby"), elem_classes=["dw-card"])
                live_text = gr.Textbox(label="📊 Live status", lines=12, elem_classes=["dw-card"])
        final_gauge = gr.HTML(_risk_gauge_html(0.0, "No Session"), elem_classes=["dw-card"])
        final_text = gr.Textbox(label="📋 Session summary", lines=10, elem_classes=["dw-card"])

        start_btn.click(live_start, inputs=None, outputs=[live_out, live_text, live_gauge])
        # Stream each captured frame through the orchestrator. concurrency_limit=1
        # keeps the shared orchestrator state serialized (frames processed in order).
        cam.stream(
            live_stream,
            inputs=[cam],
            outputs=[live_out, live_text, live_gauge],
            stream_every=0.1,
            concurrency_limit=1,
            show_progress="hidden",
        )
        stop_btn.click(live_stop, inputs=None, outputs=[final_text, final_gauge])


if __name__ == "__main__":
    demo.queue().launch(theme=THEME, css=CUSTOM_CSS, head=HEAD_HTML, ssr_mode=False)
