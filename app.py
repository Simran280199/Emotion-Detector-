"""
EmotiSense — Emotion Detection App
Loads pre-trained models from models/ folder (instant startup).
Run train_and_save.ipynb ONCE to generate models/, then use this app.
"""

import streamlit as st
import numpy as np
import pickle, re, string, os, warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="EmotiSense | Emotion Detection",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #111118 !important;
    border-right: 1px solid #2a2a3a !important;
}
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label { color: #e8e8f0 !important; }
[data-testid="stMain"] { background: #0a0a0f !important; }
.main .block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1100px !important; }
p, span, div, label, li, td, th, h1, h2, h3, h4 { color: #e8e8f0 !important; }

.stRadio label { color: #e8e8f0 !important; font-size: 0.95rem !important; }
.stTextArea textarea {
    background: #1a1a2e !important; color: #ffffff !important;
    border: 1.5px solid #3a3a5c !important; border-radius: 10px !important;
    font-size: 15px !important; padding: 12px !important;
}
.stTextArea textarea:focus { border-color: #7c5cbf !important; }
.stTextArea label { color: #c8c8e0 !important; font-weight: 500 !important; }
.stButton > button {
    background: linear-gradient(135deg, #7c5cbf 0%, #4a90d9 100%) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; padding: 0.65rem 1.5rem !important;
    font-size: 15px !important; font-weight: 600 !important; width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stTabs [data-baseweb="tab-list"] {
    background: #111118 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #9090b0 !important;
    border-radius: 8px !important; font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
}
.stTabs [aria-selected="true"] { background: #7c5cbf !important; color: #ffffff !important; }
.stSuccess { background: #0d2a1a !important; border: 1px solid #1a5c38 !important; border-radius: 10px !important; }
.stSuccess * { color: #4ade80 !important; }
.stError { background: #2a0d0d !important; border: 1px solid #5c1a1a !important; border-radius: 10px !important; }
.stError * { color: #f87171 !important; }
.streamlit-expanderHeader { background: #1a1a2e !important; color: #e8e8f0 !important; border-radius: 8px !important; }
.streamlit-expanderContent { background: #13131f !important; border: 1px solid #2a2a3a !important; }
hr { border-color: #2a2a3a !important; margin: 1.2rem 0 !important; }
[data-testid="metric-container"] {
    background: #1a1a2e !important; border: 1px solid #2a2a3a !important;
    border-radius: 12px !important; padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: #a78bfa !important; font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #9090b0 !important; font-size: 0.82rem !important; }

.hero-card {
    background: linear-gradient(135deg, #1a1030 0%, #0d1a30 50%, #101a10 100%);
    border: 1px solid #3a2a5a; border-radius: 20px;
    padding: 2.5rem 2rem; text-align: center; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;
}
.hero-card::before {
    content: ''; position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(124,92,191,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem !important; font-weight: 700 !important;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important;
    background-clip: text !important; margin-bottom: 0.5rem !important;
}
.stat-pill {
    display: inline-block; background: rgba(124,92,191,0.15);
    border: 1px solid rgba(124,92,191,0.3); color: #c4b5fd !important;
    padding: 0.3rem 0.9rem; border-radius: 999px;
    font-size: 0.82rem; font-weight: 600; margin: 0.3rem 0.2rem;
}
.result-card {
    background: linear-gradient(135deg, #1a1030, #0d1a30);
    border: 1.5px solid #3a2a5a; border-radius: 16px;
    padding: 2rem 1.5rem; text-align: center; margin: 1rem 0;
}
.bar-bg { background: #1e1e32; border-radius: 6px; height: 10px; overflow: hidden; margin: 4px 0 2px 0; }
.bar-fg { height: 100%; border-radius: 6px; }
.info-card { background: #111118; border: 1px solid #2a2a3a; border-radius: 14px; padding: 1.4rem 1.5rem; margin: 0.7rem 0; }
.info-card-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 1rem !important; font-weight: 600 !important; color: #a78bfa !important; margin-bottom: 0.6rem !important; }
.section-title { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.3rem !important; font-weight: 600 !important; color: #e8e8f0 !important; margin: 1.5rem 0 0.8rem 0 !important; }
.model-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; border-bottom: 1px solid #1e1e32; font-size: 0.9rem; }
.model-row:last-child { border-bottom: none !important; }
.badge-dl { background: rgba(124,92,191,0.2); color: #c4b5fd !important; padding: 1px 7px; border-radius: 4px; font-size: 0.72rem; }
.badge-ml { background: rgba(74,144,217,0.2); color: #93c5fd !important; padding: 1px 7px; border-radius: 4px; font-size: 0.72rem; }
.emo-chip { display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; margin: 3px; }
.sidebar-label { font-size: 0.7rem !important; font-weight: 700 !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; color: #5050a0 !important; margin-bottom: 0.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_META = {
    "anger":    {"emoji": "😠", "color": "#FF4B4B", "bg": "rgba(255,75,75,0.12)",   "desc": "Strong displeasure or hostility toward a situation or person."},
    "fear":     {"emoji": "😨", "color": "#7C6FE8", "bg": "rgba(124,111,232,0.12)", "desc": "Anxiety or apprehension caused by perceived danger or threat."},
    "joy":      {"emoji": "😄", "color": "#F5C842", "bg": "rgba(245,200,66,0.12)",  "desc": "A feeling of great pleasure, happiness, and delight."},
    "love":     {"emoji": "❤️", "color": "#FF69B4", "bg": "rgba(255,105,180,0.12)", "desc": "Deep affection, warmth, and fondness for someone or something."},
    "sadness":  {"emoji": "😢", "color": "#4FC3F7", "bg": "rgba(79,195,247,0.12)",  "desc": "Feeling of unhappiness, sorrow, or grief."},
    "surprise": {"emoji": "😲", "color": "#4ADE80", "bg": "rgba(74,222,128,0.12)",  "desc": "Feeling of mild astonishment caused by something unexpected."},
}
MODEL_RESULTS = [
    ("Bidirectional LSTM",  90.35, "DL"),
    ("LightGBM",            85.70, "ML"),
    ("Logistic Regression", 84.90, "ML"),
    ("XGBoost",             84.80, "ML"),
    ("Random Forest",       84.65, "ML"),
    ("CatBoost",            67.40, "ML"),
    ("Simple RNN",          34.75, "DL"),
    ("LSTM",                34.75, "DL"),
    ("GRU",                 34.75, "DL"),
    ("Stacked LSTM",        34.75, "DL"),
]
DATASET_STATS = {
    "train": 16000, "test": 2000, "val": 2000, "total": 20000,
    "classes": ["anger","fear","joy","love","sadness","surprise"],
    "dist": {"joy": 5362, "sadness": 4666, "anger": 2159, "fear": 1937, "love": 1304, "surprise": 572},
    "avg_len": 19.8, "max_len": 66,
}

# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════
def clean_text_ml(text):
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    sw = set(stopwords.words("english"))
    s, l = PorterStemmer(), WordNetLemmatizer()
    return " ".join(l.lemmatize(s.stem(w)) for w in text.split() if w not in sw)

def clean_text_dl(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\d+", "", text)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD SAVED MODELS (instant — no training)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    import nltk
    for pkg in ["stopwords", "wordnet", "omw-1.4"]:
        nltk.download(pkg, quiet=True)

    errors = []
    # ── BiLSTM ────────────────────────────────────────────────────────────────
    try:
        from tensorflow.keras.models import load_model
        bilstm = load_model("models/bilstm_model.h5")
        with open("models/tokenizer.pkl", "rb") as f:
            tok = pickle.load(f)
        bilstm_ok = True
    except Exception as e:
        bilstm, tok, bilstm_ok = None, None, False
        errors.append(f"BiLSTM: {e}")

    # ── CatBoost ──────────────────────────────────────────────────────────────
    try:
        with open("models/catboost_model.pkl", "rb") as f:
            cat = pickle.load(f)
        with open("models/tfidf.pkl", "rb") as f:
            tfidf = pickle.load(f)
        cat_ok = True
    except Exception as e:
        cat, tfidf, cat_ok = None, None, False
        errors.append(f"CatBoost: {e}")

    # ── Label encoder ─────────────────────────────────────────────────────────
    try:
        with open("models/label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        le_ok = True
    except Exception as e:
        le, le_ok = None, False
        errors.append(f"LabelEncoder: {e}")

    return dict(bilstm=bilstm, tok=tok, cat=cat, tfidf=tfidf, le=le,
                bilstm_ok=bilstm_ok, cat_ok=cat_ok, le_ok=le_ok, errors=errors,
                MAX_LEN=200)

def predict_bilstm(text, s):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq = s["tok"].texts_to_sequences([clean_text_dl(text)])
    pad = pad_sequences(seq, maxlen=s["MAX_LEN"], padding="post")
    return s["le"].classes_, s["bilstm"].predict(pad, verbose=0)[0]

def predict_cat(text, s):
    vec = s["tfidf"].transform([clean_text_ml(text)])
    return s["le"].classes_, s["cat"].predict_proba(vec)[0]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<p style='font-family:Space Grotesk;font-size:1.2rem;font-weight:700;color:#a78bfa !important;margin-bottom:0;'>🎭 EmotiSense</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.78rem;color:#5050a0 !important;margin-top:2px;'>NLP Emotion Detection</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<p class='sidebar-label'>Model Selection</p>", unsafe_allow_html=True)
    model_choice = st.radio("", [
        "🧠 Bidirectional LSTM (90.35%)",
        "⚡ CatBoost ML Model"
    ], index=0, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("<p class='sidebar-label'>All Model Results</p>", unsafe_allow_html=True)
    for name, acc, mtype in MODEL_RESULTS:
        badge = "<span class='badge-dl'>DL</span>" if mtype == "DL" else "<span class='badge-ml'>ML</span>"
        color = "#a78bfa" if acc == 90.35 else ("#60a5fa" if acc >= 80 else "#6060a0")
        st.markdown(f"""
<div class='model-row'>
  <span style='color:#c8c8e0 !important;'>{badge} {name}</span>
  <span style='font-weight:700;color:{color} !important;font-family:Space Grotesk;'>{acc}%</span>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p class='sidebar-label'>Emotion Classes</p>", unsafe_allow_html=True)
    for emo, m in EMOTION_META.items():
        st.markdown(f"<span class='emo-chip' style='background:{m['bg']};border:1px solid {m['color']}55;color:{m['color']} !important;'>{m['emoji']} {emo.capitalize()}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='font-size:0.75rem;color:#404070 !important;text-align:center;'>Built by Simran · NLP Assignment 2<br>Dataset: Emotions NLP (Kaggle)</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Loading models…"):
    state = load_models()

if state["errors"]:
    st.error(f"⚠️ Could not load some models. Make sure you ran **train_and_save.ipynb** first.\n\n`{'  |  '.join(state['errors'])}`")
    if not state["le_ok"]:
        st.stop()
else:
    st.success("✅ Models loaded instantly from saved files!")

# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='hero-card'>
  <div class='hero-title'>🎭 EmotiSense</div>
  <p style='color:#9090b0 !important;font-size:1.05rem;max-width:520px;margin:0 auto 1rem auto;'>
    AI-powered emotion detection using Deep Learning &amp; Machine Learning
  </p>
  <div>
    <span class='stat-pill'>🏆 Best: 90.35% Accuracy</span>
    <span class='stat-pill'>📊 20,000 Samples</span>
    <span class='stat-pill'>🏷️ 6 Emotion Classes</span>
    <span class='stat-pill'>🤖 10 Models Trained</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔍  Detect Emotion", "📊  Dataset & EDA", "🤖  Model Performance"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DETECT
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-title'>💬 Try an Example</div>", unsafe_allow_html=True)
    examples = {
        "😄 Joy":      "I am so happy today! Everything feels wonderful and bright.",
        "😠 Anger":    "This is absolutely unacceptable! I am completely furious right now.",
        "😢 Sadness":  "I feel completely lost and devastated. Nothing seems right.",
        "😨 Fear":     "I am terrified. I don't know what's going to happen next.",
        "❤️ Love":     "I deeply love my family. They make every day worth living.",
        "😲 Surprise": "I can't believe it! I had absolutely no idea this would happen!",
    }
    cols = st.columns(6)
    selected = None
    for i, (label, ex) in enumerate(examples.items()):
        if cols[i].button(label, key=f"ex_{i}", use_container_width=True):
            selected = ex

    st.markdown("<div class='section-title'>✍️ Enter Your Text</div>", unsafe_allow_html=True)
    user_text = st.text_area("", value=selected or "", height=130,
                              placeholder="Type any sentence — e.g. 'I feel really excited about tomorrow!'",
                              label_visibility="collapsed")

    use_dl = "LSTM" in model_choice
    model_ready = state["bilstm_ok"] if use_dl else state["cat_ok"]
    cur_model   = "Bidirectional LSTM" if use_dl else "CatBoost"
    cur_acc     = "90.35" if use_dl else "67.40"

    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        analyze = st.button("🔍 Analyze Emotion", use_container_width=True, disabled=not model_ready)
    with col_info:
        st.markdown(f"<p style='color:#6060a0;font-size:0.85rem;padding-top:0.7rem;'>Model: {cur_model} · Accuracy: {cur_acc}%</p>", unsafe_allow_html=True)

    if analyze and user_text.strip():
        with st.spinner("Analyzing…"):
            if use_dl:
                classes, probs = predict_bilstm(user_text, state)
            else:
                classes, probs = predict_cat(user_text, state)

        idx     = int(np.argmax(probs))
        emotion = classes[idx]
        conf    = float(probs[idx])
        meta    = EMOTION_META.get(emotion, {"emoji": "🤔", "color": "#aaa", "bg": "#222", "desc": ""})

        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"""
<div class='result-card' style='border-color:{meta["color"]}55;'>
  <div style='font-size:5rem;line-height:1;margin-bottom:0.5rem;'>{meta["emoji"]}</div>
  <div style='font-family:Space Grotesk;font-size:2rem;font-weight:700;color:{meta["color"]} !important;margin-bottom:0.3rem;'>
    {emotion.upper()}
  </div>
  <div style='font-size:2.8rem;font-weight:800;color:#ffffff !important;margin:0.3rem 0;'>
    {conf*100:.1f}%
  </div>
  <div style='color:#5050a0 !important;font-size:0.78rem;letter-spacing:1px;margin-bottom:0.8rem;'>CONFIDENCE</div>
  <div style='background:{meta["bg"]};border:1px solid {meta["color"]}33;border-radius:8px;padding:0.7rem;'>
    <p style='color:#c8c8e0 !important;font-size:0.88rem;margin:0;'>{meta["desc"]}</p>
  </div>
  <p style='color:#404070 !important;font-size:0.75rem;margin-top:0.8rem;'>{cur_model} · {cur_acc}% accuracy</p>
</div>""", unsafe_allow_html=True)

        with right:
            st.markdown("<p style='font-family:Space Grotesk;font-weight:600;color:#a78bfa !important;font-size:0.95rem;margin-bottom:0.8rem;margin-top:1rem;'>📊 All Emotion Probabilities</p>", unsafe_allow_html=True)
            for emo, prob in sorted(zip(classes, probs), key=lambda x: x[1], reverse=True):
                m = EMOTION_META.get(emo, {"emoji": "🔹", "color": "#aaa"})
                pct   = prob * 100
                is_top = emo == emotion
                st.markdown(f"""
<div style='margin-bottom:0.65rem;'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;'>
    <span style='font-size:0.88rem;font-weight:{"700" if is_top else "400"};color:{"#ffffff" if is_top else "#9090b0"} !important;'>
      {m["emoji"]} {emo.capitalize()}</span>
    <span style='font-size:0.85rem;font-weight:700;color:{m["color"]} !important;'>{pct:.1f}%</span>
  </div>
  <div class='bar-bg'>
    <div class='bar-fg' style='width:{pct:.1f}%;background:{m["color"]};opacity:{"1" if is_top else "0.55"};'></div>
  </div>
</div>""", unsafe_allow_html=True)

            with st.expander("🔬 Preprocessed text"):
                st.code(clean_text_dl(user_text), language="text")

    elif analyze and not user_text.strip():
        st.warning("⚠️ Please enter some text first.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATASET & EDA
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-title'>📦 About the Dataset</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='info-card'>
  <p class='info-card-title'>🗂️ Emotions NLP Dataset — Kaggle</p>
  <p style='color:#9090b0 !important;font-size:0.88rem;line-height:1.8;'>
    A text classification dataset of English Twitter messages labelled with 6 basic emotions.
    Compiled for NLP research in affective computing and sentiment analysis.
    Each sample is a short sentence with a single emotion label.
  </p>
  <div style='margin-top:0.8rem;'>
    <span class='stat-pill'>📎 Source: Kaggle (praveengovi)</span>
    <span class='stat-pill'>🌐 Language: English</span>
    <span class='stat-pill'>📝 Format: Text ; Label (semicolon-separated)</span>
    <span class='stat-pill'>📂 Files: train.txt / test.txt / val.txt</span>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📐 Dataset Split</div>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏋️ Train Set",  "16,000")
    m2.metric("🧪 Test Set",   "2,000")
    m3.metric("✅ Val Set",    "2,000")
    m4.metric("📊 Total",      "20,000")

    st.markdown("<div class='section-title'>🏷️ Emotion Class Distribution (Train Set)</div>", unsafe_allow_html=True)
    total = sum(DATASET_STATS["dist"].values())
    for emo in sorted(DATASET_STATS["dist"], key=DATASET_STATS["dist"].get, reverse=True):
        count = DATASET_STATS["dist"][emo]
        pct   = count / total * 100
        m     = EMOTION_META.get(emo, {"emoji": "🔹", "color": "#aaa"})
        st.markdown(f"""
<div style='margin-bottom:0.8rem;'>
  <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
    <span style='font-size:0.92rem;font-weight:600;color:#e8e8f0 !important;'>{m["emoji"]} {emo.capitalize()}</span>
    <span style='font-size:0.88rem;color:{m["color"]} !important;font-weight:700;'>{count:,} samples &nbsp;·&nbsp; {pct:.1f}%</span>
  </div>
  <div class='bar-bg' style='height:16px;'>
    <div class='bar-fg' style='width:{pct:.1f}%;background:{m["color"]};'></div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📏 Text Statistics</div>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("📝 Avg Words / Text", "19.8")
    s2.metric("📏 Max Words / Text", "66")
    s3.metric("🏷️ Emotion Classes",  "6")
    s4.metric("📖 Vocab Size",       "~20,000")

    st.markdown("<div class='section-title'>🧹 Preprocessing Pipeline</div>", unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Lowercasing",         "Convert all text to lowercase for uniformity"),
        ("2️⃣", "Punctuation Removal", "Strip all punctuation marks using str.translate()"),
        ("3️⃣", "Number Removal",      "Remove all digits with regex re.sub()"),
        ("4️⃣", "Stopword Removal",    "Remove common English stopwords (NLTK stopwords corpus)"),
        ("5️⃣", "Stemming",            "Reduce words to root form using PorterStemmer"),
        ("6️⃣", "Lemmatization",       "Normalize words to dictionary base form (WordNetLemmatizer)"),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(steps):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
<div class='info-card' style='padding:0.9rem 1.2rem;margin:0.4rem 0;'>
  <div style='display:flex;align-items:flex-start;gap:10px;'>
    <span style='font-size:1.2rem;'>{icon}</span>
    <div>
      <p style='font-weight:600;color:#c4b5fd !important;font-size:0.9rem;margin:0;'>{title}</p>
      <p style='color:#6060a0 !important;font-size:0.82rem;margin:0;'>{desc}</p>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚙️ Feature Engineering</div>", unsafe_allow_html=True)
    fe1, fe2, fe3 = st.columns(3)
    for col, (icon, title, desc, color) in zip([fe1, fe2, fe3], [
        ("📦", "Bag of Words",    "CountVectorizer\n5,000 features", "#60a5fa"),
        ("📊", "TF-IDF Vectors", "TfidfVectorizer\n5,000 features — used for ML", "#a78bfa"),
        ("🔢", "Word Embeddings","Keras Embedding\n128-dim — used for DL", "#4ade80"),
    ]):
        col.markdown(f"""
<div class='info-card' style='text-align:center;height:140px;display:flex;flex-direction:column;justify-content:center;'>
  <div style='font-size:2rem;margin-bottom:0.4rem;'>{icon}</div>
  <p style='font-weight:700;color:{color} !important;font-size:0.9rem;margin:0;'>{title}</p>
  <p style='color:#5050a0 !important;font-size:0.78rem;margin-top:4px;white-space:pre-line;'>{desc}</p>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-title'>🏆 Best Model</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='info-card' style='border-color:#7c5cbf44;background:linear-gradient(135deg,#1a1030,#0d1030);'>
  <div style='display:flex;align-items:center;gap:1.2rem;'>
    <div style='font-size:3.5rem;'>🥇</div>
    <div>
      <p style='font-family:Space Grotesk;font-size:1.4rem;font-weight:700;color:#a78bfa !important;margin:0;'>Bidirectional LSTM</p>
      <p style='color:#7070a0 !important;font-size:0.88rem;margin:4px 0;'>Deep Learning · Keras / TensorFlow</p>
      <p style='color:#4ade80 !important;font-size:1.8rem;font-weight:800;margin:0;'>90.35% Test Accuracy</p>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>📋 Full Model Leaderboard</div>", unsafe_allow_html=True)
    for rank, (name, acc, mtype) in enumerate(MODEL_RESULTS, 1):
        pct   = acc / 90.35 * 100
        color = "#a78bfa" if rank == 1 else ("#60a5fa" if acc >= 80 else ("#9090b0" if acc >= 50 else "#404060"))
        medal = "🥇" if rank == 1 else ("🥈" if rank == 2 else ("🥉" if rank == 3 else f"#{rank} "))
        badge = "<span class='badge-dl'>Deep Learning</span>" if mtype == "DL" else "<span class='badge-ml'>ML · TF-IDF</span>"
        st.markdown(f"""
<div style='background:#111118;border:1px solid #2a2a3a;border-radius:10px;padding:0.8rem 1rem;margin-bottom:0.45rem;'>
  <div style='display:flex;align-items:center;gap:0.8rem;'>
    <span style='font-size:1rem;width:2rem;text-align:center;'>{medal}</span>
    <div style='flex:1;'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;'>
        <span style='font-weight:600;color:#e8e8f0 !important;'>{name} &nbsp;{badge}</span>
        <span style='font-weight:800;color:{color} !important;font-family:Space Grotesk;font-size:1rem;'>{acc}%</span>
      </div>
      <div class='bar-bg'>
        <div class='bar-fg' style='width:{pct:.1f}%;background:{color};opacity:0.8;'></div>
      </div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🏗️ Bidirectional LSTM Architecture</div>", unsafe_allow_html=True)
    arch = [
        ("Input",               "Padded sequences, max length 200 tokens",  "#6060a0"),
        ("Embedding Layer",     "20,000 vocab size × 128 dimensions",       "#a78bfa"),
        ("Bidirectional LSTM",  "128 units, dropout=0.3, recurrent_dropout=0.3", "#60a5fa"),
        ("Dense Layer",         "64 units · ReLU activation",               "#4ade80"),
        ("Dropout",             "Rate: 0.5 — prevents overfitting",         "#f59e0b"),
        ("Output Layer",        "6 units · Softmax → probability per class","#f472b6"),
    ]
    for layer, detail, color in arch:
        st.markdown(f"""
<div style='display:flex;align-items:center;gap:1rem;background:#111118;border:1px solid #2a2a3a;
     border-left:3px solid {color};border-radius:8px;padding:0.7rem 1rem;margin-bottom:0.4rem;'>
  <div>
    <p style='font-weight:600;color:{color} !important;font-size:0.9rem;margin:0;'>{layer}</p>
    <p style='color:#6060a0 !important;font-size:0.8rem;margin:2px 0 0 0;'>{detail}</p>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>⚙️ Training Configuration</div>", unsafe_allow_html=True)
    tc1, tc2, tc3, tc4, tc5 = st.columns(5)
    tc1.metric("📦 Batch Size",   "64")
    tc2.metric("🔄 Max Epochs",   "10")
    tc3.metric("⚡ Optimizer",    "Adam")
    tc4.metric("🛑 Early Stop",   "Patience 3")
    tc5.metric("📉 Loss",         "Cat. CrossEntropy")
