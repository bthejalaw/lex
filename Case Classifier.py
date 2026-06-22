import streamlit as st
import re
import math
import io
import os
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ── Optional parsers ──────────────────────────────────────────────────────────
try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    from docx import Document
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LexAI – Legal Intelligence Suite",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #080f18; }
[data-testid="stSidebar"]          { background: #0b141f !important; border-right: 1px solid #142032; }
.block-container { padding-top: 0 !important; max-width: 1400px; }

/* Hero */
.hero {
    background: linear-gradient(120deg, #0a1929 0%, #0e2a45 60%, #102840 100%);
    border-bottom: 1px solid #1a3550;
    padding: 1.6rem 2rem 1.4rem;
    margin-bottom: 1.4rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content: ""; position: absolute; top: -40px; right: -40px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, #1565c044 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow { font-size:.65rem; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color:#4a90d9; margin-bottom:.4rem; }
.hero-title   { font-size:2.1rem; font-weight:900; color:#e8f4ff; letter-spacing:-.03em; line-height:1.1; margin-bottom:.35rem; }
.hero-title span { color:#4a90d9; }
.hero-sub     { font-size:.88rem; color:#6a9bc4; max-width:700px; }
.hero-badges  { display:flex; gap:.5rem; flex-wrap:wrap; margin-top:.8rem; }
.hero-badge   { font-size:.68rem; font-weight:600; letter-spacing:.06em; padding:.2rem .55rem; border-radius:4px; border:1px solid #1e4a72; color:#5aaad9; background:#0d2035; }

/* Section labels */
.section-head { font-size:.65rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#3d6a94; margin-bottom:.55rem; padding-bottom:.3rem; border-bottom:1px solid #14283d; }

/* Result card */
.result-card  { background:linear-gradient(135deg,#0d2540 0%,#0f3460 100%); color:white; padding:1.3rem 1.6rem; border-radius:12px; margin-bottom:.8rem; border:1px solid #1e4a72; box-shadow:0 4px 24px #00000044; }
.result-label { font-size:.62rem; letter-spacing:.14em; text-transform:uppercase; opacity:.6; margin-bottom:.35rem; }
.result-category { font-size:1.6rem; font-weight:800; line-height:1.2; }
.conf-pill    { font-size:.72rem; font-weight:700; padding:.18rem .6rem; border-radius:999px; letter-spacing:.04em; }
.override-badge { font-size:.72rem; color:#e67e22; background:#2a1500; border:1px solid #e67e2244; border-radius:6px; padding:.25rem .7rem; margin-bottom:.7rem; display:inline-block; }

/* Desc / info boxes */
.desc-box  { background:#0c1e30; border-left:3px solid #1e4a72; padding:.6rem 1rem; border-radius:0 8px 8px 0; color:#7fb3d9; font-size:.83rem; margin-bottom:1rem; }
.info-box  { background:#0a1d2e; border-left:3px solid #1e4a72; padding:.6rem .9rem; border-radius:0 8px 8px 0; font-size:.8rem; color:#6a9bc4; margin-bottom:.8rem; }
.warn-box  { background:#1c0e0e; border-left:3px solid #8b2020; padding:.6rem .9rem; border-radius:0 8px 8px 0; font-size:.8rem; color:#d97575; margin-bottom:.8rem; }
.ok-box    { background:#0a1e13; border-left:3px solid #1e7a3a; padding:.6rem .9rem; border-radius:0 8px 8px 0; font-size:.8rem; color:#5ab37a; margin-bottom:.8rem; }

/* Risk cards */
.risk-card { border-radius:12px; padding:1.3rem 1.6rem; margin-bottom:1rem; border:1px solid; }
.risk-low    { background:#0a1e13; border-color:#1e7a3a55; }
.risk-medium { background:#1c1100; border-color:#e67e2255; }
.risk-high   { background:#1c0e0e; border-color:#8b202055; }
.risk-score  { font-size:3rem; font-weight:900; line-height:1; }
.risk-label  { font-size:.75rem; letter-spacing:.12em; text-transform:uppercase; font-weight:700; margin-top:.2rem; }
.bail-rec    { font-size:1rem; font-weight:700; margin-top:.7rem; padding:.5rem .9rem; border-radius:8px; display:inline-block; }

/* Probability bars */
.prob-title  { font-size:.68rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#3d6a94; margin-bottom:.55rem; }
.prob-row    { display:flex; align-items:center; gap:.5rem; margin-bottom:.45rem; }
.prob-label  { font-size:.78rem; color:#8db8d9; min-width:220px; flex-shrink:0; }
.prob-bar-bg { flex:1; background:#0d2035; border-radius:4px; height:7px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:4px; transition:width .4s ease; }
.prob-pct    { font-size:.75rem; color:#5a8ab3; min-width:38px; text-align:right; }
.token-note  { font-size:.72rem; color:#3d6a94; margin-top:.9rem; padding-top:.6rem; border-top:1px solid #0f2337; }

/* Transcript / preview */
.transcript-box { background:#090f18; border:1px solid #14283d; border-radius:8px; padding:.85rem; font-size:.83rem; color:#b0cfe8; max-height:150px; overflow-y:auto; margin-bottom:.8rem; line-height:1.55; }
.engine-badge   { display:inline-block; background:#0d2035; color:#6a9bc4; font-size:.7rem; padding:.15rem .5rem; border-radius:999px; margin-bottom:.5rem; border:1px solid #1e4a72; }

/* Sidebar */
.cat-group-title { font-size:.65rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; color:#3d6a94; margin:.9rem 0 .3rem 0; }
.sidebar-title   { font-size:1rem; font-weight:800; color:#d0e8ff; letter-spacing:-.01em; margin-bottom:.1rem; }
.sidebar-sub     { font-size:.72rem; color:#3d6a94; margin-bottom:.7rem; }
.my-divider      { border-top:1px solid #0f2337; margin:.9rem 0; }
.placeholder     { background:#090f18; border:1px dashed #14283d; border-radius:12px; padding:3rem 1.5rem; text-align:center; color:#264d6e; font-size:.86rem; }
.placeholder-icon { font-size:2.2rem; margin-bottom:.6rem; display:block; }

/* Tabs */
[data-testid="stTabs"] button { color:#4a7a9b !important; font-size:.85rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#d0e8ff !important; border-bottom-color:#1e6abf !important; }

/* Buttons */
[data-testid="stButton"] > button[kind="primary"] {
    background:linear-gradient(135deg,#1565c0 0%,#0d47a1 100%) !important;
    border:none !important; border-radius:8px !important;
    font-weight:700 !important; letter-spacing:.04em !important;
    padding:.6rem 1rem !important; font-size:.9rem !important;
    box-shadow:0 2px 12px #1565c033 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background:linear-gradient(135deg,#1976d2 0%,#1565c0 100%) !important;
}

/* Dataset info table */
.dataset-table { width:100%; border-collapse:collapse; font-size:.8rem; color:#8db8d9; }
.dataset-table th { background:#0d2035; color:#4a90d9; text-transform:uppercase; letter-spacing:.08em; font-size:.68rem; padding:.5rem .8rem; border-bottom:1px solid #1e4a72; text-align:left; }
.dataset-table td { padding:.45rem .8rem; border-bottom:1px solid #0f2337; }
.dataset-table tr:hover td { background:#0c1e30; }

/* Slider label */
.slider-hint  { font-size:.72rem; color:#3d6a94; margin-top:-.4rem; margin-bottom:.6rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING DATA (NaïveBayes)
# ══════════════════════════════════════════════════════════════════════════════

TRAINING_DATA = {
    "Civil – Contract Dispute": [
        "breach of contract agreement terms conditions plaintiff defendant damages",
        "unpaid debt invoice payment failure goods services delivery",
        "contractual obligation violation compensation monetary relief",
        "non-performance of agreement liquidated damages penalty clause",
        "contract enforcement specific performance injunction",
        "sale agreement purchase price default buyer seller",
        "service contract termination wrongful breach notice",
        "promissory note loan repayment default interest",
        "partnership deed dispute dissolution winding up",
    ],
    "Civil – Family Law": [
        "divorce dissolution marriage alimony maintenance spouse",
        "child custody guardianship visitation rights minor",
        "matrimonial property division settlement decree",
        "domestic violence protection order restraining",
        "adoption guardianship ward court",
        "inheritance succession will probate estate",
        "dowry harassment matrimonial home eviction",
        "divorce petition mutual consent irretrievable breakdown",
        "child support custody joint sole primary",
    ],
    "Civil – Property / Land": [
        "property title land ownership dispute encroachment",
        "landlord tenant rent eviction possession",
        "ancestral property partition suit coparcener",
        "mortgage foreclosure deed registration",
        "easement right of way boundary wall",
        "trespass property possession adverse",
        "real estate fraud misrepresentation sale deed",
        "tenancy agreement rent arrears eviction notice",
        "land acquisition compensation government",
    ],
    "Civil – Tort / Personal Injury": [
        "negligence personal injury road accident civil damages compensation",
        "medical malpractice hospital doctor negligent treatment civil suit",
        "slip fall premises liability duty of care civil claim",
        "motor vehicle accident civil negligence compensation claim court",
        "defamation libel slander reputation civil damages",
        "product liability defective goods consumer civil harm",
        "wrongful death dependants civil compensation tortfeasor",
        "nuisance interference property enjoyment civil damages",
        "civil suit personal injury claim compensation court",
        "rash negligent driving civil liability compensation",
    ],
    "Civil – Commercial / IP": [
        "intellectual property trademark copyright infringement",
        "patent invention trade secret misappropriation",
        "antitrust competition unfair trade practices",
        "corporate shareholder director dispute derivative",
        "merger acquisition due diligence breach",
        "franchise agreement royalty licence infringement",
        "passing off brand name goodwill",
        "commercial arbitration dispute resolution award",
        "insolvency liquidation company winding up creditor",
    ],
    "Criminal – Felony": [
        "robbery armed knife weapon demanded handed stole fled accused FIR",
        "robbery physical assault knife threatened demanded wallet phone watch stolen",
        "dacoity armed gang robbery IPC 392 394 395 grievous hurt accused",
        "snatching bag chain robbery bus stand road street accused arrested",
        "armed robbery knife threatened victim demanded belongings fled FIR police",
        "robbery accused approached threatened knife demanded mobile cash fled scene",
        "murder homicide accused FIR charge sheet conviction arrested",
        "attempt murder accused attacked victim critical injury IPC 307 arrested",
        "kidnapping abduction ransom accused FIR criminal charges arrested",
        "rape sexual assault victim survivor prosecution accused FIR arrested",
        "drug trafficking narcotics NDPS Act possession accused arrested seized",
        "fraud forgery cheating criminal breach trust IPC accused arrested FIR",
        "terrorism sedition national security UAPA accused arrested charged",
        "money laundering PMLA proceeds crime accused arrested ED investigation",
        "FIR registered police complaint accused arrested criminal offence serious",
        "accused persons fled scene witnesses present police investigation FIR",
        "stolen property recovered accused arrested criminal charges filed court",
    ],
    "Criminal – Misdemeanor / Bailable": [
        "trespass unlawful assembly bailable offence",
        "petty theft shoplifting minor offence fine",
        "vandalism public property damage section",
        "public nuisance order disturbance police",
        "defamation criminal complaint IPC 499 500",
        "cheating deception misrepresentation minor loss",
        "simple hurt minor scuffle bailable offence",
        "stalking harassment cybercrime minor offence",
        "dishonoured cheque section 138 NI Act",
    ],
    "Criminal – Traffic / Infraction": [
        "traffic violation motor vehicle act fine challan",
        "reckless driving drunk DUI alcohol breath test",
        "licence suspension speeding red light",
        "hit and run accident vehicular negligence",
        "road rage minor offence traffic police",
        "overloading truck penalty transport",
    ],
    "Specialized – Administrative Law": [
        "government agency ruling administrative tribunal",
        "licence permit cancellation government order",
        "social security benefits pension claim rejection",
        "public authority action writ mandamus certiorari",
        "service matter government employee disciplinary",
        "regulatory authority SEBI TRAI CCI order",
        "right to information RTI public authority",
        "environmental clearance pollution board order NGT",
    ],
    "Specialized – Constitutional Law": [
        "fundamental rights Article Constitution violation",
        "writ petition High Court Supreme Court",
        "unconstitutional law arbitrary discrimination",
        "freedom speech expression Article 19 restriction",
        "right equality Article 14 classification",
        "due process life liberty Article 21",
        "judicial review legislative act invalid",
        "PIL public interest litigation constitutional bench",
        "directive principles state policy",
    ],
    "Specialized – Bankruptcy / Insolvency": [
        "bankruptcy insolvency IBC NCLT resolution plan",
        "debt repayment creditor liquidation assets",
        "corporate insolvency resolution process CIRP",
        "personal insolvency fresh start discharge",
        "financial creditor operational creditor claim",
        "moratorium automatic stay assets protection",
        "resolution professional committee creditors",
        "voluntary insolvency winding up petition",
    ],
}

FELONY_OVERRIDE_KEYWORDS = {
    "robbery","robbed","dacoity","dacoit","snatched","snatching",
    "knife","knifepoint","gunpoint","pistol","revolver","firearm","weapon",
    "fir","chargesheet","charge sheet",
    "kidnapped","kidnapping","abducted","abduction","ransom",
    "murdered","murder","homicide","raped","rape",
    "following me","pre-planned","preplanned",
    "accused fled","fled from the scene",
}

def has_felony_override(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in FELONY_OVERRIDE_KEYWORDS)

STOP_WORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","was","are","were","be","been","being","have","has","had","do",
    "does","did","will","would","shall","should","may","might","can","could",
    "this","that","these","those","it","its","he","she","they","we","i",
    "his","her","their","our","my","your","not","no","by","from","as","if",
    "so","all","any","each","every","also","than","then","when","where",
    "which","who","whom","what","how","there","here","both","such","into",
    "about","above","after","before","between","through","under","over",
    "other","another","same","more","most","some","just","very","too",
}

def tokenize(text: str) -> list:
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior = {}
        self.word_log_likelihood = {}
        self.vocab = set()
        self.classes = []

    def fit(self, data: dict):
        word_counts  = defaultdict(lambda: defaultdict(int))
        class_totals = defaultdict(int)
        total_docs   = 0
        for label, sentences in data.items():
            for sentence in sentences:
                tokens = tokenize(sentence)
                for tok in tokens:
                    word_counts[label][tok] += 1
                    self.vocab.add(tok)
                class_totals[label] += len(tokens)
            total_docs += len(sentences)
        self.classes = list(data.keys())
        V = len(self.vocab)
        for label, sentences in data.items():
            self.class_log_prior[label] = math.log(len(sentences) / total_docs)
        for label in self.classes:
            self.word_log_likelihood[label] = {}
            total = class_totals[label]
            for word in self.vocab:
                count = word_counts[label].get(word, 0)
                self.word_log_likelihood[label][word] = math.log(
                    (count + self.alpha) / (total + self.alpha * V)
                )

    def predict_proba(self, text: str) -> dict:
        tokens = tokenize(text)
        scores = {}
        for label in self.classes:
            score = self.class_log_prior[label]
            for tok in tokens:
                if tok in self.vocab:
                    score += self.word_log_likelihood[label][tok]
            scores[label] = score
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        return {k: v / total for k, v in exp_scores.items()}

    def predict(self, text: str) -> tuple:
        if has_felony_override(text):
            proba = self.predict_proba(text)
            felony_key = "Criminal – Felony"
            boosted = {k: v * 0.05 for k, v in proba.items()}
            boosted[felony_key] = max(proba.get(felony_key, 0), 0.70)
            total = sum(boosted.values())
            proba = {k: v / total for k, v in boosted.items()}
            return felony_key, proba
        proba = self.predict_proba(text)
        return max(proba, key=proba.get), proba

@st.cache_resource
def load_nb_model():
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(TRAINING_DATA)
    return clf

# ══════════════════════════════════════════════════════════════════════════════
#  COMPAS DATASET & RANDOM FOREST BAIL MODEL
# ══════════════════════════════════════════════════════════════════════════════

COMPAS_PATH = "/mnt/user-data/uploads/compas-scores-two-years-violent.csv"

@st.cache_data
def load_compas_data():
    """Load and preprocess the COMPAS violent recidivism dataset."""
    df = pd.read_csv(COMPAS_PATH)

    # Keep only needed columns; drop rows with critical missing values
    keep_cols = [
        "age", "sex", "race",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
        "priors_count", "c_charge_degree",
        "v_decile_score",    # COMPAS violent risk score 1-10
        "is_violent_recid",  # ground-truth label
        "two_year_recid",
    ]
    # The CSV has duplicate column names — pandas auto-suffixes them; pick the last priors_count
    try:
        df = df[keep_cols].copy()
    except KeyError:
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()

    df.dropna(subset=["age", "priors_count", "is_violent_recid"], inplace=True)

    # Encode sex
    df["sex_enc"] = (df["sex"].str.strip().str.lower() == "male").astype(int)

    # Encode charge degree: F=1 (felony), M=0 (misdemeanor), else 0
    df["charge_felony"] = (df["c_charge_degree"].str.strip().str.upper() == "F").astype(int)

    # Juvenile total arrests proxy
    for col in ["juv_fel_count", "juv_misd_count", "juv_other_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["juv_total"] = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]

    # COMPAS decile → flight risk proxy (score ≥ 7 = high, bucketized 0/1/2)
    df["v_decile_score"] = pd.to_numeric(df["v_decile_score"], errors="coerce").fillna(5)
    df["compas_risk_tier"] = pd.cut(
        df["v_decile_score"], bins=[0, 3, 6, 10],
        labels=[0, 1, 2], include_lowest=True
    ).astype(int)

    # Target: violent recidivism within 2 years
    df["target"] = pd.to_numeric(df["is_violent_recid"], errors="coerce").fillna(0).astype(int)

    feature_cols = [
        "age", "sex_enc", "priors_count", "juv_total",
        "charge_felony", "compas_risk_tier",
    ]
    return df, feature_cols

@st.cache_resource
def train_bail_model():
    df, feature_cols = load_compas_data()
    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm     = confusion_matrix(y_test, y_pred)
    feature_imp = dict(zip(feature_cols, rf.feature_importances_))
    return rf, report, cm, feature_imp, len(df), X_test, y_test

def risk_score_from_proba(proba_high: float) -> int:
    """Map RF probability of violent recidivism to 0–100 risk score."""
    return min(100, int(round(proba_high * 100)))

def risk_tier(score: int):
    if score < 35:
        return "Low", "#27ae60", "risk-low", "🟢"
    elif score < 65:
        return "Medium", "#e67e22", "risk-medium", "🟡"
    else:
        return "High", "#e74c3c", "risk-high", "🔴"

def bail_recommendation(score: int, severity: str) -> tuple[str, str]:
    """Return (recommendation_text, badge_css_class)"""
    if score < 35:
        return "✅ Bail Likely Grantable — Low flight/recidivism risk", "#27ae60"
    elif score < 55:
        return "⚠️ Conditional Bail — Consider supervision / surety bond", "#f39c12"
    elif score < 75:
        return "🔶 Bail with Strict Conditions — Regular reporting required", "#e67e22"
    else:
        if severity == "Felony":
            return "❌ Bail Not Recommended — High violent recidivism risk, serious charge", "#e74c3c"
        else:
            return "🔶 High-Risk Conditional Bail — Intensive supervision required", "#e67e22"

# ══════════════════════════════════════════════════════════════════════════════
#  TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if HAS_PYMUPDF:
            doc = fitz.open(stream=raw, filetype="pdf")
            return "\n".join(page.get_text() for page in doc)
        return "[PDF parsing requires PyMuPDF. Run: pip install pymupdf]"
    if name.endswith(".docx"):
        if HAS_DOCX:
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        return "[DOCX parsing requires python-docx. Run: pip install python-docx]"
    return raw.decode("utf-8", errors="ignore")

def transcribe_audio(audio_bytes, filename, progress_cb=None):
    if not HAS_SR:
        raise RuntimeError("SpeechRecognition not installed.")
    if not HAS_PYDUB:
        raise RuntimeError("pydub not installed.")
    ext = filename.rsplit(".", 1)[-1].lower()
    fmt_map = {"mp3":"mp3","mp4":"mp4","m4a":"mp4","ogg":"ogg","flac":"flac","webm":"webm","wav":"wav"}
    fmt = fmt_map.get(ext, "mp3")
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
    audio = audio.set_frame_rate(16000).set_channels(1)
    target_dBFS = -20.0
    change = target_dBFS - audio.dBFS
    if abs(change) > 1:
        audio = audio.apply_gain(change)
    CHUNK_MS = 20_000; OVERLAP_MS = 1_000
    total_ms = len(audio)
    starts   = list(range(0, total_ms, CHUNK_MS - OVERLAP_MS))
    chunks   = [audio[s: s + CHUNK_MS] for s in starts]
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 200
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8
    transcripts = []; engine_used = "Google Web Speech"
    for idx, chunk in enumerate(chunks):
        if progress_cb: progress_cb(idx, len(chunks))
        if chunk.dBFS < -50: continue
        buf = io.BytesIO(); chunk.export(buf, format="wav"); buf.seek(0)
        with sr.AudioFile(buf) as source:
            recognizer.adjust_for_ambient_noise(source, duration=min(0.3, len(chunk)/2000))
            audio_data = recognizer.record(source)
        text = None
        for lang in ("en-IN","en-US","en-GB"):
            try:
                text = recognizer.recognize_google(audio_data, language=lang)
                engine_used = f"Google Web Speech ({lang})"; break
            except sr.UnknownValueError: continue
            except sr.RequestError as e: raise RuntimeError(f"Google Speech API error: {e}")
        if text: transcripts.append(text.strip())
    if not transcripts:
        raise RuntimeError("Speech could not be understood.")
    return " ".join(transcripts), engine_used

# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORY METADATA
# ══════════════════════════════════════════════════════════════════════════════

DESCRIPTIONS = {
    "Civil – Contract Dispute": "Disputes from breach of agreements, unpaid debts, or failure to deliver goods/services.",
    "Civil – Family Law": "Matrimonial matters: divorce, custody, alimony, adoption, and inheritance.",
    "Civil – Property / Land": "Landlord-tenant conflicts, land title disputes, ancestral partition, mortgage cases.",
    "Civil – Tort / Personal Injury": "Negligence claims — medical malpractice, road accidents, defamation, product liability.",
    "Civil – Commercial / IP": "Trademark/copyright infringement, antitrust, corporate disputes, arbitration.",
    "Criminal – Felony": "Serious offences: robbery, murder, rape, kidnapping, drug trafficking, terrorism.",
    "Criminal – Misdemeanor / Bailable": "Lesser offences: petty theft, trespass, cheating, stalking, dishonoured cheques (s.138).",
    "Criminal – Traffic / Infraction": "Motor Vehicle Act violations, DUI, reckless driving, hit-and-run cases.",
    "Specialized – Administrative Law": "Challenges to government/agency orders, RTI, service matters, regulatory decisions.",
    "Specialized – Constitutional Law": "Fundamental rights violations, writ petitions, PILs, judicial review.",
    "Specialized – Bankruptcy / Insolvency": "IBC/NCLT proceedings, CIRP, liquidation, creditor claims, personal insolvency.",
}

ICONS = {
    "Civil – Contract Dispute":"📝","Civil – Family Law":"👨‍👩‍👧",
    "Civil – Property / Land":"🏠","Civil – Tort / Personal Injury":"⚖️",
    "Civil – Commercial / IP":"💼","Criminal – Felony":"🔴",
    "Criminal – Misdemeanor / Bailable":"🟡","Criminal – Traffic / Infraction":"🚦",
    "Specialized – Administrative Law":"🏛️","Specialized – Constitutional Law":"📜",
    "Specialized – Bankruptcy / Insolvency":"💰",
}

def confidence_colour(pct):
    if pct >= 70: return "#27ae60"
    if pct >= 45: return "#f39c12"
    return "#c0392b"

GROUPS = {
    "⚖️ Civil": ["Civil – Contract Dispute","Civil – Family Law","Civil – Property / Land","Civil – Tort / Personal Injury","Civil – Commercial / IP"],
    "🔴 Criminal": ["Criminal – Felony","Criminal – Misdemeanor / Bailable","Criminal – Traffic / Infraction"],
    "🏛️ Specialized": ["Specialized – Administrative Law","Specialized – Constitutional Law","Specialized – Bankruptcy / Insolvency"],
}

# ══════════════════════════════════════════════════════════════════════════════
#  RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_classifier_results(clf, text, source_label=""):
    prediction, proba = clf.predict(text)
    confidence = proba[prediction] * 100
    icon  = ICONS.get(prediction, "⚖️")
    desc  = DESCRIPTIONS.get(prediction, "")
    cc    = confidence_colour(confidence)
    override_note = ""
    if has_felony_override(text):
        override_note = '<div class="override-badge">🔒 Hard-matched via criminal keywords</div>'
    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Predicted Category{' · ' + source_label if source_label else ''}</div>
        <div class="result-category">{icon} {prediction}</div>
        <div style="display:flex;align-items:center;gap:.6rem;margin-top:.4rem;">
            <span class="conf-pill" style="background:{cc}22;color:{cc};border:1px solid {cc}55;">
                {confidence:.1f}% confidence
            </span>
        </div>
    </div>
    {override_note}
    """, unsafe_allow_html=True)
    st.markdown(f'<div class="desc-box">📌 {desc}</div>', unsafe_allow_html=True)
    st.markdown('<div class="prob-title">Top 5 probabilities</div>', unsafe_allow_html=True)
    for cat, prob in sorted(proba.items(), key=lambda x: x[1], reverse=True)[:5]:
        ico = ICONS.get(cat,""); pct = prob*100; w = max(pct,2)
        col = "#2d6a9f" if cat != prediction else "#1e88e5"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label">{ico} {cat}</div>
            <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{w}%;background:{col};"></div></div>
            <div class="prob-pct">{pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    tokens = [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOP_WORDS and len(t) > 2]
    st.markdown(
        f'<div class="token-note">ℹ️ {len(tokens)} meaningful tokens · Multinomial Naïve Bayes (α=1)</div>',
        unsafe_allow_html=True,
    )

def render_risk_gauge(score, label, color, icon):
    """Render a styled risk gauge using matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 2.2), facecolor="none")
    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=14, color="#0d2035", solid_capstyle="round")
    # Filled arc proportional to score
    fill_theta = np.linspace(np.pi, np.pi - (np.pi * score / 100), 200)
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=14, color=color, solid_capstyle="round")
    # Score text
    ax.text(0, -0.15, str(score), ha="center", va="center",
            fontsize=42, fontweight="bold", color=color)
    ax.text(0, -0.62, f"{icon} {label} Risk", ha="center", va="center",
            fontsize=11, fontweight="bold", color=color)
    ax.text(0, 0.45, "Risk Score / 100", ha="center", va="center",
            fontsize=8, color="#3d6a94")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.9, 1.1)
    ax.axis("off")
    ax.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    return fig

def render_feature_importance(feature_imp):
    labels = {
        "age":"Age", "sex_enc":"Gender (Male=1)",
        "priors_count":"Prior Convictions", "juv_total":"Juvenile Arrests",
        "charge_felony":"Charge Severity (Felony)", "compas_risk_tier":"COMPAS Risk Tier",
    }
    names  = [labels.get(k, k) for k in feature_imp]
    values = list(feature_imp.values())
    sorted_pairs = sorted(zip(values, names), reverse=True)
    vals, nms = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(5, 3), facecolor="none")
    colors = ["#1e88e5" if v == max(vals) else "#1a4a7a" for v in vals]
    bars = ax.barh(nms, vals, color=colors, height=0.55, edgecolor="none")
    ax.set_xlabel("Importance", color="#6a9bc4", fontsize=8)
    ax.tick_params(colors="#6a9bc4", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.set_facecolor("none"); fig.patch.set_alpha(0.0)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#5aaad9", fontsize=7.5)
    ax.tick_params(axis='x', colors="#3d6a94")
    for label in ax.get_yticklabels():
        label.set_color("#8db8d9")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-title">⚖️ LexAI Suite</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Legal Intelligence · Naïve Bayes + Random Forest</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Two AI modules: <b>Case Classifier</b> (Naïve Bayes) and '
        '<b>Bail Risk Assessment</b> (Random Forest on COMPAS dataset).</div>',
        unsafe_allow_html=True,
    )

    missing = []
    if not HAS_SR:      missing.append("`SpeechRecognition`")
    if not HAS_PYDUB:   missing.append("`pydub`")
    if not HAS_PYMUPDF: missing.append("`pymupdf`")
    if not HAS_DOCX:    missing.append("`python-docx`")
    if missing:
        st.markdown(
            '<div class="warn-box">⚠️ Missing packages:<br>' + " · ".join(missing) +
            "<br><code>pip install SpeechRecognition pydub pymupdf python-docx</code></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="ok-box">✅ All packages installed</div>', unsafe_allow_html=True)

    st.markdown('<div class="my-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📚 Category Reference")
    for group_label, cats in GROUPS.items():
        st.markdown(f'<div class="cat-group-title">{group_label}</div>', unsafe_allow_html=True)
        for cat in cats:
            with st.expander(f"{ICONS.get(cat,'')} {cat}"):
                st.caption(DESCRIPTIONS.get(cat, ""))

    st.markdown('<div class="my-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🗄️ COMPAS Dataset")
    st.caption(
        "**Source:** ProPublica's COMPAS Recidivism dataset — Broward County, Florida.\n\n"
        "**Records:** ~4,700 defendants screened 2013–2014.\n\n"
        "**Target:** `is_violent_recid` — violent re-offense within 2 years.\n\n"
        "**Features used:** Age, Sex, Prior Convictions, Juvenile Arrests, "
        "Charge Severity, COMPAS Risk Tier."
    )
    st.caption("⚠️ *For educational/research demonstration only. Not for use in real bail decisions.*")

# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════

nb_clf   = load_nb_model()
rf_model, rf_report, rf_cm, feature_imp, n_records, X_test, y_test = train_bail_model()
df_full, feature_cols = load_compas_data()

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI-Powered Legal Intelligence Suite</div>
    <div class="hero-title">Lex<span>AI</span> — Classifier · Bail Risk · COMPAS Analytics</div>
    <div class="hero-sub">
        Two AI modules in one platform: a Multinomial Naïve Bayes <b>case classifier</b>
        and a Random Forest <b>bail risk assessment tool</b> trained on the ProPublica COMPAS
        violent-recidivism dataset (Broward County, FL · 4,700+ defendants).
    </div>
    <div class="hero-badges">
        <span class="hero-badge">📝 TXT · PDF · DOCX</span>
        <span class="hero-badge">🎙️ Voice / Audio</span>
        <span class="hero-badge">🧮 Naïve Bayes</span>
        <span class="hero-badge">🌲 Random Forest</span>
        <span class="hero-badge">🗄️ COMPAS Dataset</span>
        <span class="hero-badge">🔒 No External API</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_classifier, tab_bail, tab_dataset = st.tabs([
    "⚖️  Case Classifier",
    "🔍  Bail Risk Assessment",
    "📊  Dataset & Model Insights",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — CASE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
with tab_classifier:
    col_input, col_result = st.columns([1, 1], gap="large")

    if "classified_text"  not in st.session_state: st.session_state.classified_text  = ""
    if "classified_label" not in st.session_state: st.session_state.classified_label = ""
    if "audio_transcript" not in st.session_state: st.session_state.audio_transcript = ""
    if "audio_engine"     not in st.session_state: st.session_state.audio_engine     = ""
    if "error_msg"        not in st.session_state: st.session_state.error_msg        = ""

    with col_input:
        st.markdown('<div class="section-head">Input</div>', unsafe_allow_html=True)
        tab_paste, tab_upload, tab_audio = st.tabs(["✏️ Paste Text", "📂 Upload File", "🎙️ Audio"])
        active_text = ""; active_label = ""

        with tab_paste:
            pasted = st.text_area(
                "Case description", height=240,
                placeholder="e.g. The petitioner challenges the eviction notice…",
                label_visibility="collapsed",
            )
            if pasted.strip():
                active_text = pasted.strip(); active_label = "Pasted Text"

        with tab_upload:
            uploaded = st.file_uploader("Upload case file", type=["txt","pdf","docx"], label_visibility="collapsed")
            if uploaded:
                with st.spinner("Extracting text…"):
                    extracted = extract_text(uploaded)
                st.markdown(
                    f'<div class="ok-box">✅ Extracted <b>{len(extracted):,}</b> characters from <b>{uploaded.name}</b></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Preview extracted text"):
                    st.markdown(
                        f'<div class="transcript-box">{extracted[:1500]}{"…" if len(extracted)>1500 else ""}</div>',
                        unsafe_allow_html=True,
                    )
                active_text = extracted; active_label = f"File: {uploaded.name}"

        with tab_audio:
            if not HAS_SR or not HAS_PYDUB:
                st.markdown(
                    '<div class="warn-box">⚠️ Audio requires: <code>pip install SpeechRecognition pydub</code> '
                    'and <code>ffmpeg</code></div>', unsafe_allow_html=True,
                )
            else:
                audio_file = st.file_uploader(
                    "Upload audio", type=["mp3","wav","m4a","mp4","ogg","flac","webm"],
                    label_visibility="collapsed", key="audio_uploader",
                )
                if audio_file:
                    st.audio(audio_file)
                    if st.button("🔊 Transcribe Audio", use_container_width=True):
                        progress_bar = st.progress(0, text="Starting…")
                        def on_progress(idx, total):
                            progress_bar.progress(int((idx/total)*100), text=f"Chunk {idx+1}/{total}…")
                        try:
                            t, eng = transcribe_audio(audio_file.read(), audio_file.name, on_progress)
                            progress_bar.progress(100, text="Done!")
                            st.session_state.audio_transcript = t
                            st.session_state.audio_engine = eng
                            st.session_state.error_msg = ""
                        except Exception as e:
                            progress_bar.empty()
                            st.session_state.error_msg = str(e)
                    if st.session_state.audio_transcript:
                        st.markdown(f'<span class="engine-badge">🔧 {st.session_state.audio_engine}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="transcript-box">{st.session_state.audio_transcript}</div>', unsafe_allow_html=True)
                        active_text = st.session_state.audio_transcript; active_label = "Audio → Text"
                    if st.session_state.error_msg:
                        st.markdown(f'<div class="warn-box">❌ {st.session_state.error_msg}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Classify Case", type="primary", use_container_width=True):
            if active_text.strip():
                st.session_state.classified_text  = active_text
                st.session_state.classified_label = active_label
            else:
                st.session_state.classified_text  = ""
                st.warning("Please provide input first.")

    with col_result:
        st.markdown('<div class="section-head">Classification Result</div>', unsafe_allow_html=True)
        if st.session_state.classified_text:
            render_classifier_results(nb_clf, st.session_state.classified_text, st.session_state.classified_label)
        else:
            st.markdown(
                '<div class="placeholder"><span class="placeholder-icon">⚖️</span>'
                "Results will appear here after classification.<br><br>"
                "<span style='font-size:.78rem;color:#1e4060;'>Paste text · Upload file · Transcribe audio → click Classify Case</span>"
                "</div>", unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — BAIL RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_bail:
    st.markdown("""
    <div class="info-box">
    🌲 <b>Random Forest Bail Risk Tool</b> — Trained on the ProPublica COMPAS violent recidivism
    dataset (~4,700 Broward County defendants). The model predicts the probability of violent
    re-offense within 2 years and maps it to a 0–100 risk score with bail recommendations.<br>
    <br><i>⚠️ For academic/educational demonstration only. Not for real judicial use.</i>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_output = st.columns([1, 1.1], gap="large")

    with col_form:
        st.markdown('<div class="section-head">Defendant Input Parameters</div>', unsafe_allow_html=True)

        age = st.slider("🎂 Age", min_value=16, max_value=80, value=28, step=1)
        st.markdown('<div class="slider-hint">Age of the defendant at time of screening</div>', unsafe_allow_html=True)

        gender = st.selectbox("👤 Gender", ["Male", "Female"])

        employment = st.selectbox(
            "💼 Employment Status",
            ["Employed (Full-time)", "Employed (Part-time)", "Unemployed", "Student", "Retired"],
        )
        # Map employment to a community-ties proxy (not a direct RF feature but shown for UX)
        emp_community_score = {
            "Employed (Full-time)": 0.9, "Employed (Part-time)": 0.7,
            "Student": 0.75, "Retired": 0.8, "Unemployed": 0.3,
        }[employment]

        prev_convictions = st.slider("⚖️ Previous Convictions (Priors Count)", 0, 30, 2)
        st.markdown('<div class="slider-hint">Total prior adult criminal convictions on record</div>', unsafe_allow_html=True)

        num_arrests = st.slider("🚔 Juvenile Arrests (Age < 18)", 0, 20, 0)
        st.markdown('<div class="slider-hint">Total juvenile felony + misdemeanor + other arrests</div>', unsafe_allow_html=True)

        crime_severity = st.selectbox("🔴 Charge Severity", ["Felony", "Misdemeanor"])
        charge_felony_val = 1 if crime_severity == "Felony" else 0

        community_ties = st.select_slider(
            "🏘️ Community Ties",
            options=["None", "Weak", "Moderate", "Strong", "Very Strong"],
            value="Moderate",
        )

        flight_risk = st.selectbox(
            "✈️ Flight Risk Indicators",
            ["None known", "Failed to appear once", "Failed to appear multiple times", "Prior bail jump"],
        )
        flight_map = {"None known": 0, "Failed to appear once": 1,
                      "Failed to appear multiple times": 2, "Prior bail jump": 2}
        flight_val = flight_map[flight_risk]

        substance_abuse = st.selectbox(
            "💊 Substance Abuse History",
            ["None", "Alcohol — minor", "Cannabis — minor", "Hard drugs — treated", "Hard drugs — active"],
        )
        substance_map = {"None": 0, "Alcohol — minor": 0, "Cannabis — minor": 0,
                         "Hard drugs — treated": 1, "Hard drugs — active": 2}
        substance_val = substance_map[substance_abuse]

        # Derive COMPAS risk tier proxy from flight + substance
        compas_tier = min(2, flight_val + (1 if substance_val > 0 else 0))

        st.markdown("<br>", unsafe_allow_html=True)
        assess_btn = st.button("🔍  Run Risk Assessment", type="primary", use_container_width=True)

    with col_output:
        st.markdown('<div class="section-head">Risk Assessment Output</div>', unsafe_allow_html=True)

        if assess_btn:
            sex_enc_val = 1 if gender == "Male" else 0
            input_vec = pd.DataFrame([{
                "age": age,
                "sex_enc": sex_enc_val,
                "priors_count": prev_convictions,
                "juv_total": num_arrests,
                "charge_felony": charge_felony_val,
                "compas_risk_tier": compas_tier,
            }])

            proba = rf_model.predict_proba(input_vec)[0]
            # proba[1] = probability of violent recidivism
            high_prob = proba[1]
            score = risk_score_from_proba(high_prob)

            # Adjust score for community ties (qualitative modifier)
            community_adj = {"None": +6, "Weak": +3, "Moderate": 0, "Strong": -4, "Very Strong": -8}[community_ties]
            score = max(0, min(100, score + community_adj))

            label, color, css_cls, emoji = risk_tier(score)
            rec_text, rec_color = bail_recommendation(score, crime_severity)

            # Gauge chart
            gauge_fig = render_risk_gauge(score, label, color, emoji)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close(gauge_fig)

            # Bail recommendation
            st.markdown(
                f'<div style="background:{rec_color}22;border:1px solid {rec_color}55;border-radius:10px;'
                f'padding:.9rem 1.2rem;margin-bottom:1rem;">'
                f'<div style="font-size:.62rem;letter-spacing:.12em;text-transform:uppercase;color:{rec_color};margin-bottom:.3rem;">Bail Recommendation</div>'
                f'<div style="font-size:1rem;font-weight:700;color:#e8f4ff;">{rec_text}</div>'
                f'</div>', unsafe_allow_html=True,
            )

            # Risk breakdown table
            st.markdown('<div class="prob-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)

            factors = [
                ("Age", age, "Lower age → higher base risk" if age < 25 else "Age within moderate range"),
                ("Gender", gender, "Male defendants show higher recidivism rates in the dataset"),
                ("Prior Convictions", prev_convictions, f"{'High' if prev_convictions > 5 else 'Moderate' if prev_convictions > 2 else 'Low'} prior record"),
                ("Juvenile Arrests", num_arrests, f"{'Significant' if num_arrests > 3 else 'Minor'} juvenile history"),
                ("Charge Severity", crime_severity, "Felony charges correlate with higher risk"),
                ("Community Ties", community_ties, f"{'Stabilising' if community_ties in ['Strong','Very Strong'] else 'Limited protective'} community anchor"),
                ("Flight Risk", flight_risk, f"{'Elevated' if flight_val > 0 else 'No known'} flight history"),
                ("Substance Use", substance_abuse, f"{'Active concern' if substance_val > 1 else 'Managed/No'} substance factor"),
            ]

            for name, value, note in factors:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:flex-start;
                            padding:.4rem 0;border-bottom:1px solid #0f2337;">
                    <div style="font-size:.78rem;color:#8db8d9;font-weight:600;min-width:160px;">{name}</div>
                    <div style="font-size:.78rem;color:#4a90d9;min-width:120px;text-align:center;">{value}</div>
                    <div style="font-size:.72rem;color:#3d6a94;flex:1;text-align:right;">{note}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div class="token-note">ℹ️ Random Forest (300 trees · max_depth=8) · '
                f'Trained on {n_records:,} COMPAS records · '
                f'Violent recidivism probability: {high_prob*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="placeholder"><span class="placeholder-icon">🌲</span>'
                "Risk assessment results will appear here.<br><br>"
                "<span style='font-size:.78rem;color:#1e4060;'>"
                "Fill in defendant parameters → click <b>Run Risk Assessment</b>"
                "</span></div>", unsafe_allow_html=True,
            )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — DATASET & MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_dataset:
    st.markdown('<div class="section-head">COMPAS Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    total_records = len(df_full)
    violent_recid_rate = df_full["target"].mean() * 100
    avg_age = df_full["age"].mean()
    avg_priors = df_full["priors_count"].mean()

    c1.metric("Total Records", f"{total_records:,}", help="After cleaning")
    c2.metric("Violent Recid. Rate", f"{violent_recid_rate:.1f}%", help="is_violent_recid within 2 years")
    c3.metric("Average Age", f"{avg_age:.1f} yrs")
    c4.metric("Avg. Prior Convictions", f"{avg_priors:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)

    col_ds1, col_ds2 = st.columns([1.1, 1], gap="large")

    with col_ds1:
        st.markdown("#### 📋 Dataset Field Reference")
        field_info = [
            ("age", "Defendant age at COMPAS screening", "Numeric (16–96)"),
            ("sex", "Gender of defendant", "Male / Female"),
            ("race", "Race/ethnicity of defendant", "African-American, Caucasian, Hispanic, Other"),
            ("priors_count", "Total prior adult criminal convictions", "Numeric (0–38)"),
            ("juv_fel_count", "Juvenile felony arrests", "Numeric"),
            ("juv_misd_count", "Juvenile misdemeanor arrests", "Numeric"),
            ("juv_other_count", "Other juvenile arrests", "Numeric"),
            ("c_charge_degree", "Severity of current charge", "F (Felony) / M (Misdemeanor)"),
            ("c_charge_desc", "Textual description of current charge", "Free text"),
            ("v_decile_score", "COMPAS violent risk decile score", "1 (Low) – 10 (High)"),
            ("v_score_text", "COMPAS violent risk text label", "Low / Medium / High"),
            ("is_violent_recid", "Did the person violently reoffend? (target)", "0 = No, 1 = Yes"),
            ("two_year_recid", "Any recidivism within 2 years", "0 = No, 1 = Yes"),
            ("compas_screening_date", "Date of COMPAS screening", "Date (2013–2014)"),
        ]
        st.markdown("""
        <table class="dataset-table">
          <tr><th>Field</th><th>Description</th><th>Values / Type</th></tr>
        """ + "".join(
            f"<tr><td><code>{f}</code></td><td>{d}</td><td style='color:#4a90d9'>{v}</td></tr>"
            for f, d, v in field_info
        ) + "</table>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🌲 Random Forest — Model Performance")
        if "1" in rf_report:
            prec = rf_report["1"]["precision"]
            rec  = rf_report["1"]["recall"]
            f1   = rf_report["1"]["f1-score"]
            acc  = rf_report["accuracy"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc*100:.1f}%")
            m2.metric("Precision", f"{prec*100:.1f}%", help="Of those flagged high-risk, % who actually reoffended")
            m3.metric("Recall", f"{rec*100:.1f}%", help="Of actual violent reoffenders, % correctly flagged")
            m4.metric("F1 Score", f"{f1:.3f}")

        st.markdown("#### 🔧 Model Configuration")
        st.code("""RandomForestClassifier(
    n_estimators  = 300,      # number of decision trees
    max_depth     = 8,        # max tree depth (prevents overfitting)
    min_samples_leaf = 10,    # min samples per leaf
    class_weight  = 'balanced', # handles class imbalance
    random_state  = 42,
    n_jobs        = -1         # parallel processing
)

Features used:
  age, sex_enc, priors_count, juv_total,
  charge_felony, compas_risk_tier

Target:  is_violent_recid  (0 = no, 1 = yes)
Split:   80% train / 20% test  (stratified)
""", language="python")

    with col_ds2:
        st.markdown("#### 📊 Feature Importance")
        imp_fig = render_feature_importance(feature_imp)
        st.pyplot(imp_fig, use_container_width=True)
        plt.close(imp_fig)

        st.markdown("#### 🎯 Violent Recidivism by Age Group")
        age_bins = pd.cut(df_full["age"], bins=[15, 25, 35, 45, 55, 100],
                          labels=["16–25", "26–35", "36–45", "46–55", "55+"])
        age_rec = df_full.groupby(age_bins, observed=True)["target"].mean() * 100

        fig_age, ax_age = plt.subplots(figsize=(5, 3), facecolor="none")
        bars = ax_age.bar(age_rec.index, age_rec.values, color="#1e88e5", edgecolor="none", width=0.6)
        for bar, val in zip(bars, age_rec.values):
            ax_age.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f"{val:.1f}%", ha="center", color="#5aaad9", fontsize=8)
        ax_age.set_ylabel("Violent Recid. Rate (%)", color="#6a9bc4", fontsize=8)
        ax_age.set_xlabel("Age Group", color="#6a9bc4", fontsize=8)
        ax_age.tick_params(colors="#6a9bc4", labelsize=8)
        ax_age.spines[:].set_visible(False)
        ax_age.set_facecolor("none"); fig_age.patch.set_alpha(0.0)
        for lbl in ax_age.get_xticklabels() + ax_age.get_yticklabels():
            lbl.set_color("#8db8d9")
        st.pyplot(fig_age, use_container_width=True)
        plt.close(fig_age)

        st.markdown("#### 📈 COMPAS Score vs Actual Recidivism")
        score_rec = df_full.groupby("compas_risk_tier", observed=True)["target"].mean() * 100
        tier_labels = {0: "Low (1–3)", 1: "Medium (4–6)", 2: "High (7–10)"}
        tier_colors = ["#27ae60", "#e67e22", "#e74c3c"]

        fig_tier, ax_tier = plt.subplots(figsize=(5, 2.5), facecolor="none")
        xs = [tier_labels.get(i, str(i)) for i in score_rec.index]
        ys = score_rec.values
        colors_t = tier_colors[:len(xs)]
        bars_t = ax_tier.bar(xs, ys, color=colors_t, edgecolor="none", width=0.55)
        for bar, val in zip(bars_t, ys):
            ax_tier.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                         f"{val:.1f}%", ha="center", color="#e8f4ff", fontsize=9, fontweight="bold")
        ax_tier.set_ylabel("Actual Violent Recid. (%)", color="#6a9bc4", fontsize=8)
        ax_tier.tick_params(colors="#6a9bc4", labelsize=8)
        ax_tier.spines[:].set_visible(False)
        ax_tier.set_facecolor("none"); fig_tier.patch.set_alpha(0.0)
        for lbl in ax_tier.get_xticklabels() + ax_tier.get_yticklabels():
            lbl.set_color("#8db8d9")
        st.pyplot(fig_tier, use_container_width=True)
        plt.close(fig_tier)

        st.markdown("#### 📋 Sample Records from Dataset")
        sample_cols = ["age", "sex", "priors_count", "c_charge_degree", "v_score_text", "is_violent_recid"]
        available_sample = [c for c in sample_cols if c in df_full.columns]
        st.dataframe(
            df_full[available_sample].head(8).rename(columns={
                "age":"Age","sex":"Gender","priors_count":"Priors",
                "c_charge_degree":"Charge","v_score_text":"COMPAS Score",
                "is_violent_recid":"Violent Recid."
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="my-divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="warn-box">
    ⚠️ <b>Important Disclaimer</b> — This tool is built for academic and educational demonstration 
    of machine learning in the legal domain. The COMPAS dataset and any algorithmic risk score 
    should <b>never</b> be the sole or primary basis for bail or detention decisions. 
    ProPublica's analysis found significant racial disparities in COMPAS scores. 
    Real bail decisions must involve judicial discretion, legal counsel, and full case context.
    </div>
    """, unsafe_allow_html=True)