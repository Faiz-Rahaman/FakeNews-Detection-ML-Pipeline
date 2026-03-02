import html
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from .ensemble import StackingEnsemble, WeightedEnsemble
    from .feature_engineering import HybridFeatureBuilder
    import src.ensemble as _ensemble_mod
    import src.feature_engineering as _feature_mod
except ImportError:
    from ensemble import StackingEnsemble, WeightedEnsemble
    from feature_engineering import HybridFeatureBuilder
    import ensemble as _ensemble_mod
    import feature_engineering as _feature_mod

sys.modules.setdefault("ensemble", _ensemble_mod)
sys.modules.setdefault("feature_engineering", _feature_mod)


MODEL_PATH = Path("models/research_fast/final_model.joblib")
_SOURCE_CUE_RE = re.compile(
    r"\b(according to|reported by|stated by|announced by|official said|sources said|study found)\b",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"(\[\d+\]|\(\d{4}\)|source[s]?:|data from)", re.IGNORECASE)


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and str(artifact.get("version", "")).startswith(("2.", "3.")):
        return artifact
    if isinstance(artifact, dict) and "model" in artifact:
        return {
            "version": "legacy",
            "model": artifact["model"],
            "threshold": float(artifact.get("threshold", 0.5)),
            "label_map": artifact.get("label_map", {0: "REAL", 1: "FAKE"}),
            "model_name": artifact.get("model_name", "legacy_pipeline"),
        }
    return {
        "version": "legacy",
        "model": artifact,
        "threshold": 0.5,
        "label_map": {0: "REAL", 1: "FAKE"},
        "model_name": "legacy_pipeline",
    }


app = FastAPI(title="Veritas Detector")
artifact: dict[str, Any] | None = None
bert_runtime: dict[str, Any] = {"tokenizer": None, "model": None, "device": "cpu", "model_dir": None}


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    model: str
    probability_fake: float
    probability_real: float
    confidence: float
    threshold: float
    prediction: str
    explanation: list[str]


def _load_bert_runtime(model_dir: Path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    global bert_runtime
    if bert_runtime["model"] is not None and bert_runtime["model_dir"] == str(model_dir):
        return bert_runtime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()
    bert_runtime = {"tokenizer": tokenizer, "model": model, "device": device, "model_dir": str(model_dir)}
    return bert_runtime


def _bert_probabilities(model_dir: Path, texts: list[str]) -> np.ndarray:
    import torch

    rt = _load_bert_runtime(model_dir)
    tokenizer = rt["tokenizer"]
    model = rt["model"]
    device = rt["device"]

    enc = tokenizer(texts, truncation=True, padding=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def _text_signals(text: str) -> dict[str, float]:
    words = re.findall(r"\b\w+\b", text)
    caps = sum(1 for c in text if c.isupper())
    alpha = sum(1 for c in text if c.isalpha())
    exclam = text.count("!")
    questions = text.count("?")
    numbers = len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", text))
    source_cues = len(_SOURCE_CUE_RE.findall(text))
    citations = len(_CITATION_RE.findall(text))
    caps_ratio = (caps / alpha) if alpha else 0.0

    return {
        "word_count": float(len(words)),
        "caps_ratio": float(caps_ratio),
        "exclam": float(exclam),
        "questions": float(questions),
        "numbers": float(numbers),
        "source_cues": float(source_cues),
        "citations": float(citations),
    }


def _build_explanation(prob_fake: float, prediction: str, text: str) -> list[str]:
    s = _text_signals(text)
    lines: list[str] = []

    confidence = max(prob_fake, 1.0 - prob_fake)
    if confidence < 0.60:
        lines.append("Low-confidence classification: the text is close to the model decision boundary.")

    if prediction == "FAKE":
        lines.append(f"Model estimates a high fake likelihood (P(fake)={prob_fake:.2%}).")
        if s["exclam"] >= 2 or s["questions"] >= 2:
            lines.append("The tone appears emotionally loaded (multiple exclamation/question markers).")
        if s["source_cues"] == 0 and s["citations"] == 0:
            lines.append("The text includes few verifiable sourcing/citation cues.")
        if s["caps_ratio"] > 0.12:
            lines.append("Unusually high capitalization can be a sensational-style signal.")
    else:
        lines.append(f"Model estimates a low fake likelihood (P(fake)={prob_fake:.2%}).")
        if s["source_cues"] > 0:
            lines.append("The text contains source-reporting cues (e.g., 'according to', 'reported by').")
        if s["citations"] > 0 or s["numbers"] > 0:
            lines.append("Presence of citations/data-like details supports factual reporting structure.")
        if s["exclam"] <= 1 and s["questions"] <= 1:
            lines.append("Tone appears relatively neutral and less sensational.")

    if not lines:
        lines.append("Prediction is based on learned linguistic and semantic patterns from training data.")
    return lines


def infer_prob_fake(text: str) -> tuple[float, str, float]:
    assert artifact is not None

    threshold = float(artifact.get("threshold", 0.5))
    version = str(artifact.get("version", "legacy"))

    if version.startswith(("2.", "3.")):
        feature_builder = HybridFeatureBuilder.from_artifacts(artifact["feature_artifacts"])
        X = feature_builder.transform([text])
        p_linear = artifact["linear_svc"].predict_proba(X)[:, 1]

        bert_dir = Path(artifact.get("bert_model_dir", ""))
        if bert_dir.exists():
            p_bert = _bert_probabilities(bert_dir, [text])
        else:
            p_bert = np.array([0.5], dtype=np.float64)

        if "stacking_meta_model" in artifact:
            stacker = StackingEnsemble()
            stacker.meta_model = artifact["stacking_meta_model"]
            stacker.constant_prob = artifact.get("stacking_constant_prob", None)
            stacker.use_average = bool(artifact.get("stacking_use_average", False))
            prob_fake = float(stacker.predict_proba(np.column_stack([p_linear, p_bert]))[0])
            model_name = "research-stacking"
        else:
            p_xgb = artifact["xgboost"].predict_proba(X)[:, 1] if "xgboost" in artifact else np.array([0.5])
            w = artifact.get("ensemble_weights", {"linear": 0.5, "xgboost": 0.0, "bert": 0.5})
            ensemble = WeightedEnsemble(w["linear"], w["xgboost"], w["bert"])
            prob_fake = float(ensemble.predict_proba(p_linear, p_xgb, p_bert)[0])
            model_name = "research-weighted"
        return prob_fake, model_name, threshold

    model = artifact["model"]
    prob_fake = float(model.predict_proba([text])[0][1])
    return prob_fake, str(artifact.get("model_name", "legacy_pipeline")), threshold


def infer(text: str) -> PredictResponse:
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    prob_fake, model_name, threshold = infer_prob_fake(text)
    pred = int(prob_fake >= threshold)
    label_map = artifact.get("label_map", {0: "REAL", 1: "FAKE"}) if artifact else {0: "REAL", 1: "FAKE"}
    prediction = str(label_map.get(pred, "FAKE" if pred == 1 else "REAL"))
    explanation = _build_explanation(prob_fake, prediction, text)

    return PredictResponse(
        model=model_name,
        probability_fake=prob_fake,
        probability_real=1.0 - prob_fake,
        confidence=max(prob_fake, 1.0 - prob_fake),
        threshold=threshold,
        prediction=prediction,
        explanation=explanation,
    )


@app.on_event("startup")
def startup_event():
    global artifact
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Train first with src/train.py or update MODEL_PATH in src/app.py."
        )
    artifact = load_artifact(MODEL_PATH)


def render_page(text: str = "", result: PredictResponse | None = None, error: str | None = None) -> str:
    safe_text = html.escape(text)
    error_html = f"<p class='error'>{html.escape(error)}</p>" if error else ""

    result_html = ""
    if result is not None:
        tone = "fake" if result.prediction.upper() == "FAKE" else "real"
        meter = max(1, min(100, int(round(result.confidence * 100))))
        explain_items = "".join(f"<li>{html.escape(x)}</li>" for x in result.explanation)

        result_html = f"""
        <section class='result {tone}'>
          <div class='badge-row'>
            <span class='badge'>Model: {html.escape(result.model)}</span>
            <span class='badge'>Threshold: {result.threshold:.2f}</span>
          </div>
          <h2>{html.escape(result.prediction)}</h2>
          <div class='score-grid'>
            <div class='score'>
              <span>P(fake)</span>
              <strong>{result.probability_fake:.4f}</strong>
            </div>
            <div class='score'>
              <span>P(real)</span>
              <strong>{result.probability_real:.4f}</strong>
            </div>
            <div class='score'>
              <span>Confidence</span>
              <strong>{result.confidence:.2%}</strong>
            </div>
          </div>
          <div class='meter'><div class='fill' style='width:{meter}%'></div></div>
          <h3>Why This Prediction?</h3>
          <ul>{explain_items}</ul>
        </section>
        """

    return f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <title>Veritas Detector</title>
  <link rel='preconnect' href='https://fonts.googleapis.com'>
  <link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>
  <link href='https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Libre+Baskerville:wght@400;700&display=swap' rel='stylesheet'>
  <style>
    :root {{
      --bg:#f1efe8;
      --ink:#13151c;
      --muted:#566173;
      --card:#ffffffd6;
      --line:#1d243214;
      --primary:#1f63d7;
      --good:#118a5a;
      --bad:#c54141;
      --shadow:0 18px 42px rgba(9,19,32,0.16);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      min-height:100vh;
      font-family:'Space Grotesk','Segoe UI',sans-serif;
      color:var(--ink);
      background:
        radial-gradient(900px 500px at 8% -5%, #bed9fb 0%, transparent 55%),
        radial-gradient(850px 450px at 90% 5%, #f8d9be 0%, transparent 50%),
        linear-gradient(135deg,#f7f2e7,var(--bg));
    }}
    .wrap {{ width:min(980px,92vw); margin:2.2rem auto; }}
    .shell {{ background:var(--card); border:1px solid var(--line); border-radius:20px; box-shadow:var(--shadow); padding:1.2rem; }}
    h1 {{ margin:.1rem 0 .35rem; font-family:'Libre Baskerville',serif; font-size:clamp(1.9rem,4vw,2.9rem); }}
    .lead {{ margin:0 0 1rem; color:var(--muted); }}
    .credit {{
      margin: 0 0 .35rem;
      color: #2c3e59;
      font-size: .95rem;
      display: inline-flex;
      gap: .45rem;
      align-items: center;
      flex-wrap: wrap;
      background: #f6f9ff;
      border: 1px solid #d9e6ff;
      border-radius: 999px;
      padding: .3rem .72rem;
    }}
    .credit b {{ color: #123b85; }}
    .credit a {{ color: #1355be; text-decoration: none; font-weight: 600; }}
    .credit a:hover {{ text-decoration: underline; }}
    textarea {{ width:100%; min-height:220px; border:1px solid var(--line); border-radius:12px; padding:.9rem; font-size:1rem; line-height:1.45; }}
    textarea:focus {{ outline:none; border-color:#4a8df7; box-shadow:0 0 0 4px #4a8df722; }}
    .actions {{ display:flex; gap:.65rem; margin-top:.85rem; flex-wrap:wrap; }}
    .btn {{ border:none; border-radius:999px; padding:.64rem 1.1rem; font-weight:700; cursor:pointer; }}
    .btn.primary {{ background:linear-gradient(120deg,var(--primary),#3f8cff); color:#fff; }}
    .btn.secondary {{ background:#eef4ff; color:#1c57be; border:1px solid #c8dbff; text-decoration:none; display:inline-flex; align-items:center; }}
    .error {{ margin-top:.8rem; color:#8a1d1d; background:#ffe8e8; border:1px solid #ffcaca; border-radius:10px; padding:.55rem .7rem; }}
    .result {{ margin-top:1rem; border:1px solid var(--line); border-radius:14px; padding:1rem; background:#fff; }}
    .result.real {{ border-left:6px solid var(--good); }}
    .result.fake {{ border-left:6px solid var(--bad); }}
    .badge-row {{ display:flex; gap:.5rem; flex-wrap:wrap; margin-bottom:.3rem; }}
    .badge {{ border:1px solid var(--line); border-radius:999px; background:#f8faff; padding:.22rem .6rem; font-size:.82rem; color:#354252; }}
    h2 {{ margin:.3rem 0 .65rem; font-size:1.6rem; }}
    h3 {{ margin:.85rem 0 .35rem; font-size:1.05rem; }}
    .score-grid {{ display:grid; grid-template-columns:repeat(3,minmax(120px,1fr)); gap:.55rem; }}
    .score {{ border:1px solid var(--line); border-radius:10px; padding:.55rem .65rem; background:#fcfdff; }}
    .score span {{ font-size:.82rem; color:var(--muted); display:block; }}
    .score strong {{ font-size:1.08rem; }}
    .meter {{ margin:.75rem 0; height:11px; border-radius:999px; background:#e6edf7; overflow:hidden; }}
    .fill {{ height:100%; background:linear-gradient(90deg,#2d82ff,#16a071); }}
    ul {{ margin:.25rem 0 0 1.05rem; padding:0; }}
    li {{ margin:.26rem 0; color:#303b49; }}
    @media (max-width:680px) {{ .score-grid {{ grid-template-columns:1fr; }} textarea {{ min-height:180px; }} }}
  </style>
</head>
<body>
  <main class='wrap'>
    <section class='shell'>
      <h1>Veritas Detector</h1>
      <p class='credit'><b>Created by Faiz Rahaman</b> <span>•</span> <a href='mailto:faizr3712@gmail.com'>faizr3712@gmail.com</a></p>
      <p class='lead'>Paste a news article and get prediction, confidence scores, and an interpretable reason summary.</p>
      <form method='post' action='/predict'>
        <textarea name='text' placeholder='Paste full headline + body text...'>{safe_text}</textarea>
        <div class='actions'>
          <button class='btn primary' type='submit'>Analyze</button>
          <a class='btn secondary' href='/docs'>API Docs</a>
        </div>
      </form>
      {error_html}
      {result_html}
    </section>
  </main>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifact is not None, "model_path": str(MODEL_PATH)}


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(payload: PredictRequest):
    return infer(payload.text)


@app.get("/", response_class=HTMLResponse)
def home():
    return render_page()


@app.post("/predict", response_class=HTMLResponse)
def predict_form(text: str = Form(...)):
    try:
        result = infer(text)
        return render_page(text=text, result=result)
    except HTTPException as exc:
        return render_page(text=text, error=str(exc.detail))
