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

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
    DEEPFACE_ERROR = None
except Exception as e:
    HAS_DEEPFACE = False
    DEEPFACE_ERROR = str(e)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

import tempfile

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="LexAI — Legal Intelligence Suite",
    page_icon="assets/favicon.ico" if os.path.exists("assets/favicon.ico") else None,
    layout="wide",
)

st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
* { box-sizing: border-box; }

[data-testid="stAppViewContainer"] {
    background: #07090d;
    color: #c8d8e8;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #090c12 !important;
    border-right: 1px solid #12202e !important;
}

.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px;
}

/* ── Masthead ── */
.masthead {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    padding: 1.8rem 0 .6rem 0;
    border-bottom: 1px solid #12202e;
    margin-bottom: 2rem;
}
.masthead-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: .04em;
    color: #e8f0f8;
}
.masthead-wordmark span {
    color: #4a7cb5;
}
.masthead-rule {
    flex: 1;
    height: 1px;
    background: #12202e;
    margin: 0 1.5rem;
}
.masthead-descriptor {
    font-family: 'Inter', sans-serif;
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: .16em;
    text-transform: uppercase;
    color: #3a5a7a;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: .62rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #2e4f6d;
    padding-bottom: .5rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #12202e;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(160deg, #0b1928 0%, #0e2240 100%);
    border: 1px solid #1a3550;
    border-radius: 4px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
}
.result-eyebrow {
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: #2e4f6d;
    margin-bottom: .5rem;
}
.result-category {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #d8ecff;
    line-height: 1.2;
    margin-bottom: .4rem;
}
.confidence-tag {
    display: inline-block;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .08em;
    padding: .2rem .7rem;
    border-radius: 2px;
}
.override-note {
    font-size: .72rem;
    color: #b87a3a;
    background: #1e1200;
    border: 1px solid #b87a3a33;
    border-radius: 2px;
    padding: .3rem .8rem;
    margin-top: .6rem;
    display: inline-block;
    letter-spacing: .04em;
}

/* ── Desc / notice boxes ── */
.desc-box {
    background: #090f18;
    border-left: 2px solid #1e3a55;
    padding: .65rem 1rem;
    border-radius: 0 3px 3px 0;
    color: #6a9bc4;
    font-size: .82rem;
    margin-bottom: 1rem;
    line-height: 1.55;
}
.notice-info {
    background: #08111c;
    border-left: 2px solid #1e3a55;
    padding: .6rem .9rem;
    border-radius: 0 3px 3px 0;
    font-size: .8rem;
    color: #5a8ab3;
    margin-bottom: .8rem;
}
.notice-warn {
    background: #150a0a;
    border-left: 2px solid #7a2020;
    padding: .6rem .9rem;
    border-radius: 0 3px 3px 0;
    font-size: .8rem;
    color: #c06060;
    margin-bottom: .8rem;
}
.notice-ok {
    background: #081510;
    border-left: 2px solid #1e6a38;
    padding: .6rem .9rem;
    border-radius: 0 3px 3px 0;
    font-size: .8rem;
    color: #4a9a68;
    margin-bottom: .8rem;
}

/* ── Risk cards ── */
.risk-low    { background: #081510; border: 1px solid #1e6a3833; }
.risk-medium { background: #150d00; border: 1px solid #c07a2033; }
.risk-high   { background: #150808; border: 1px solid #7a202033; }

/* ── Probability bars ── */
.prob-label-title {
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #2e4f6d;
    margin-bottom: .6rem;
}
.prob-row { display: flex; align-items: center; gap: .6rem; margin-bottom: .5rem; }
.prob-name { font-size: .78rem; color: #7aaccc; min-width: 230px; flex-shrink: 0; font-family: 'Inter', sans-serif; }
.prob-track { flex: 1; background: #0b1928; border-radius: 2px; height: 5px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 2px; }
.prob-pct { font-family: 'JetBrains Mono', monospace; font-size: .73rem; color: #4a7aa0; min-width: 42px; text-align: right; }

/* ── Transcript box ── */
.transcript-box {
    background: #060a10;
    border: 1px solid #12202e;
    border-radius: 3px;
    padding: .8rem 1rem;
    font-size: .81rem;
    color: #8aaec8;
    max-height: 140px;
    overflow-y: auto;
    margin-bottom: .8rem;
    line-height: 1.6;
    font-family: 'Inter', sans-serif;
}
.engine-label {
    display: inline-block;
    background: #0b1928;
    color: #5a8ab3;
    font-size: .68rem;
    font-family: 'JetBrains Mono', monospace;
    padding: .15rem .55rem;
    border-radius: 2px;
    margin-bottom: .5rem;
    border: 1px solid #1a3550;
}

/* ── Sidebar ── */
.sidebar-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #d8ecff;
    letter-spacing: .05em;
    margin-bottom: .15rem;
}
.sidebar-sub {
    font-size: .65rem;
    font-weight: 500;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: #2e4f6d;
    margin-bottom: .8rem;
}
.sidebar-divider { border-top: 1px solid #10192a; margin: .8rem 0; }

/* ── Placeholder ── */
.placeholder {
    background: #060a10;
    border: 1px dashed #12202e;
    border-radius: 3px;
    padding: 3.5rem 2rem;
    text-align: center;
    color: #1e3a55;
    font-size: .84rem;
    font-family: 'Inter', sans-serif;
    line-height: 1.7;
}
.placeholder-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.3rem;
    color: #1e3a55;
    margin-bottom: .4rem;
}

/* ── Bail rec box ── */
.bail-rec-box {
    border-radius: 3px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
}
.bail-rec-label {
    font-size: .6rem;
    font-weight: 600;
    letter-spacing: .2em;
    text-transform: uppercase;
    margin-bottom: .3rem;
}
.bail-rec-text {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: #e8f4ff;
    line-height: 1.3;
}

/* ── Metric footer ── */
.metric-note {
    font-family: 'JetBrains Mono', monospace;
    font-size: .68rem;
    color: #1e3a55;
    margin-top: 1rem;
    padding-top: .6rem;
    border-top: 1px solid #0f1e2d;
}

/* ── Dataset table ── */
.ds-table { width: 100%; border-collapse: collapse; font-size: .79rem; color: #7aaccc; }
.ds-table th {
    background: #0b1928;
    color: #3a6a94;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-size: .62rem;
    padding: .55rem .9rem;
    border-bottom: 1px solid #12202e;
    text-align: left;
    font-weight: 600;
}
.ds-table td { padding: .45rem .9rem; border-bottom: 1px solid #0d1a28; }
.ds-table tr:hover td { background: #0b1928; }
code { background: #0b1928; color: #5aaad9; padding: .1rem .35rem; border-radius: 2px; font-family: 'JetBrains Mono', monospace; font-size: .82em; }

/* ── Slider hint ── */
.slider-hint { font-size: .68rem; color: #2e4f6d; margin-top: -.3rem; margin-bottom: .6rem; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
    color: #3a5a7a !important;
    font-size: .84rem !important;
    letter-spacing: .02em !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #c8d8e8 !important;
    border-bottom-color: #1e5090 !important;
}

/* ── Primary buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #0e2a48 !important;
    border: 1px solid #1e4a72 !important;
    border-radius: 2px !important;
    font-weight: 600 !important;
    letter-spacing: .08em !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .84rem !important;
    color: #8ab8d8 !important;
    box-shadow: none !important;
    transition: background .2s, border-color .2s;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #122f52 !important;
    border-color: #2a5a88 !important;
    color: #b8d4ee !important;
}

/* ── Facial recognition card ── */
.face-result-card {
    background: #0b1928;
    border: 1px solid #1a3550;
    border-radius: 3px;
    padding: 1.4rem 1.8rem;
    margin-bottom: 1rem;
}
.face-result-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #d8ecff;
    margin-bottom: .3rem;
}
.conf-bar-wrap {
    background: #060a10;
    border-radius: 2px;
    height: 6px;
    margin: .5rem 0 .8rem 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #1e5090, #4a7cb5);
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING DATA — NAIVE BAYES
# ══════════════════════════════════════════════════════════════════════════════

TRAINING_DATA = {
    "Civil — Contract Dispute": [
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
    "Civil — Family Law": [
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
    "Civil — Property and Land": [
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
    "Civil — Tort and Personal Injury": [
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
    "Civil — Commercial and IP": [
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
    "Criminal — Felony": [
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
    "Criminal — Misdemeanor / Bailable": [
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
    "Criminal — Traffic / Infraction": [
        "traffic violation motor vehicle act fine challan",
        "reckless driving drunk DUI alcohol breath test",
        "licence suspension speeding red light",
        "hit and run accident vehicular negligence",
        "road rage minor offence traffic police",
        "overloading truck penalty transport",
    ],
    "Specialized — Administrative Law": [
        "government agency ruling administrative tribunal",
        "licence permit cancellation government order",
        "social security benefits pension claim rejection",
        "public authority action writ mandamus certiorari",
        "service matter government employee disciplinary",
        "regulatory authority SEBI TRAI CCI order",
        "right to information RTI public authority",
        "environmental clearance pollution board order NGT",
    ],
    "Specialized — Constitutional Law": [
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
    "Specialized — Bankruptcy / Insolvency": [
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
            felony_key = "Criminal — Felony"
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
#  COMPAS DATASET & RANDOM FOREST
# ══════════════════════════════════════════════════════════════════════════════

COMPAS_PATH ="compas-scores-two-years-violent.csv"

@st.cache_data
def load_compas_data():
    df = pd.read_csv(COMPAS_PATH)
    keep_cols = [
        "age", "sex", "race",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
        "priors_count", "c_charge_degree",
        "v_decile_score",
        "is_violent_recid",
        "two_year_recid",
    ]
    try:
        df = df[keep_cols].copy()
    except KeyError:
        available = [c for c in keep_cols if c in df.columns]
        df = df[available].copy()
    df.dropna(subset=["age", "priors_count", "is_violent_recid"], inplace=True)
    df["sex_enc"] = (df["sex"].str.strip().str.lower() == "male").astype(int)
    df["charge_felony"] = (df["c_charge_degree"].str.strip().str.upper() == "F").astype(int)
    for col in ["juv_fel_count", "juv_misd_count", "juv_other_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["juv_total"] = df["juv_fel_count"] + df["juv_misd_count"] + df["juv_other_count"]
    df["v_decile_score"] = pd.to_numeric(df["v_decile_score"], errors="coerce").fillna(5)
    df["compas_risk_tier"] = pd.cut(
        df["v_decile_score"], bins=[0, 3, 6, 10],
        labels=[0, 1, 2], include_lowest=True
    ).astype(int)
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
    return min(100, int(round(proba_high * 100)))

def risk_tier(score: int):
    if score < 35:
        return "Low", "#27ae60", "risk-low"
    elif score < 65:
        return "Medium", "#c07a20", "risk-medium"
    else:
        return "High", "#c03030", "risk-high"

def bail_recommendation(score: int, severity: str) -> tuple:
    if score < 35:
        return "Bail Grantable — Low flight and recidivism risk assessed", "#1e6a38"
    elif score < 55:
        return "Conditional Bail — Surety bond or supervision conditions advised", "#a06010"
    elif score < 75:
        return "Bail with Strict Conditions — Regular court reporting required", "#a04010"
    else:
        if severity == "Felony":
            return "Bail Not Recommended — High violent recidivism risk, serious charge", "#8a1818"
        else:
            return "High-Risk Conditional Bail — Intensive supervision required", "#a04010"

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
        return "[PDF parsing requires PyMuPDF. Install: pip install pymupdf]"
    if name.endswith(".docx"):
        if HAS_DOCX:
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)
        return "[DOCX parsing requires python-docx. Install: pip install python-docx]"
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
    "Civil — Contract Dispute": "Disputes arising from breach of agreements, unpaid debts, or failure to deliver goods and services.",
    "Civil — Family Law": "Matrimonial matters including divorce, child custody, alimony, adoption, and inheritance.",
    "Civil — Property and Land": "Landlord-tenant conflicts, land title disputes, ancestral partition, and mortgage cases.",
    "Civil — Tort and Personal Injury": "Negligence claims encompassing medical malpractice, road accidents, defamation, and product liability.",
    "Civil — Commercial and IP": "Trademark and copyright infringement, antitrust matters, corporate disputes, and commercial arbitration.",
    "Criminal — Felony": "Serious offences including robbery, murder, sexual assault, kidnapping, drug trafficking, and terrorism.",
    "Criminal — Misdemeanor / Bailable": "Lesser offences such as petty theft, trespass, criminal cheating, stalking, and dishonoured cheques.",
    "Criminal — Traffic / Infraction": "Motor Vehicle Act violations, impaired driving, reckless operation, and hit-and-run cases.",
    "Specialized — Administrative Law": "Challenges to government and regulatory agency orders, RTI matters, and service disputes.",
    "Specialized — Constitutional Law": "Fundamental rights violations, writ petitions, public interest litigation, and judicial review.",
    "Specialized — Bankruptcy / Insolvency": "IBC/NCLT proceedings, corporate insolvency resolution, liquidation, and personal insolvency.",
}

GROUPS = {
    "Civil": [
        "Civil — Contract Dispute", "Civil — Family Law", "Civil — Property and Land",
        "Civil — Tort and Personal Injury", "Civil — Commercial and IP",
    ],
    "Criminal": [
        "Criminal — Felony", "Criminal — Misdemeanor / Bailable", "Criminal — Traffic / Infraction",
    ],
    "Specialized": [
        "Specialized — Administrative Law", "Specialized — Constitutional Law",
        "Specialized — Bankruptcy / Insolvency",
    ],
}

def confidence_colour(pct):
    if pct >= 70: return "#2e8a52"
    if pct >= 45: return "#9a6010"
    return "#9a2828"

# ══════════════════════════════════════════════════════════════════════════════
#  RENDERERS
# ══════════════════════════════════════════════════════════════════════════════

def render_classifier_results(clf, text, source_label=""):
    prediction, proba = clf.predict(text)
    confidence = proba[prediction] * 100
    cc = confidence_colour(confidence)
    override_note = ""
    if has_felony_override(text):
        override_note = '<div class="override-note">Keyword-matched: criminal offence indicators detected</div>'

    st.markdown(f"""
    <div class="result-card">
        <div class="result-eyebrow">Predicted Category{' · ' + source_label if source_label else ''}</div>
        <div class="result-category">{prediction}</div>
        <div>
            <span class="confidence-tag" style="background:{cc}1a;color:{cc};border:1px solid {cc}44;">
                {confidence:.1f}% confidence
            </span>
        </div>
        {override_note}
    </div>
    """, unsafe_allow_html=True)

    desc = DESCRIPTIONS.get(prediction, "")
    st.markdown(f'<div class="desc-box">{desc}</div>', unsafe_allow_html=True)

    st.markdown('<div class="prob-label-title">Probability Distribution — Top 5 Categories</div>', unsafe_allow_html=True)
    for cat, prob in sorted(proba.items(), key=lambda x: x[1], reverse=True)[:5]:
        pct = prob * 100
        w   = max(pct, 1)
        col = "#1e5090" if cat != prediction else "#2a70c0"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-name">{cat}</div>
            <div class="prob-track"><div class="prob-fill" style="width:{w}%;background:{col};"></div></div>
            <div class="prob-pct">{pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    tokens = [t for t in re.findall(r"[a-z]+", text.lower()) if t not in STOP_WORDS and len(t) > 2]
    st.markdown(
        f'<div class="metric-note">{len(tokens)} tokens analysed · Multinomial Naive Bayes (alpha = 1.0) · 11 categories</div>',
        unsafe_allow_html=True,
    )

def render_risk_gauge(score, label, color):
    fig, ax = plt.subplots(figsize=(4, 2.4), facecolor="none")
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), lw=14, color="#0b1928", solid_capstyle="round")
    fill_theta = np.linspace(np.pi, np.pi - (np.pi * score / 100), 200)
    ax.plot(np.cos(fill_theta), np.sin(fill_theta), lw=14, color=color, solid_capstyle="round")
    ax.text(0, -0.1, str(score), ha="center", va="center",
            fontsize=44, fontweight="bold", color=color, fontfamily="DejaVu Sans")
    ax.text(0, -0.58, f"{label} Risk", ha="center", va="center",
            fontsize=10, fontweight="600", color=color)
    ax.text(0, 0.45, "Composite Risk Score  /  100", ha="center", va="center",
            fontsize=7.5, color="#2e4f6d")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.9, 1.1)
    ax.axis("off")
    ax.set_facecolor("none"); fig.patch.set_alpha(0.0)
    return fig

def render_feature_importance(feature_imp):
    labels = {
        "age": "Age",
        "sex_enc": "Gender (Male = 1)",
        "priors_count": "Prior Convictions",
        "juv_total": "Juvenile Arrests",
        "charge_felony": "Charge Severity (Felony)",
        "compas_risk_tier": "COMPAS Risk Tier",
    }
    names  = [labels.get(k, k) for k in feature_imp]
    values = list(feature_imp.values())
    sorted_pairs = sorted(zip(values, names), reverse=True)
    vals, nms = zip(*sorted_pairs)
    fig, ax = plt.subplots(figsize=(5, 3), facecolor="none")
    colors = ["#2a70c0" if v == max(vals) else "#1a3a60" for v in vals]
    bars = ax.barh(nms, vals, color=colors, height=0.5, edgecolor="none")
    ax.set_xlabel("Importance", color="#3a6a94", fontsize=8)
    ax.tick_params(colors="#3a6a94", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.set_facecolor("none"); fig.patch.set_alpha(0.0)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", color="#4a8ab8", fontsize=7.5)
    ax.tick_params(axis='x', colors="#2e4f6d")
    for lbl in ax.get_yticklabels():
        lbl.set_color("#7aaccc")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  FACIAL RECOGNITION — DATASET PATH
# ══════════════════════════════════════════════════════════════════════════════

FACE_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-wordmark">LexAI Suite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-sub">Legal Intelligence &amp; Biometric Identification</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">System Status</div>', unsafe_allow_html=True)
    missing = []
    if not HAS_SR:       missing.append("SpeechRecognition")
    if not HAS_PYDUB:    missing.append("pydub")
    if not HAS_PYMUPDF:  missing.append("pymupdf")
    if not HAS_DOCX:     missing.append("python-docx")
    if not HAS_DEEPFACE: missing.append("deepface")
    if not HAS_PIL:      missing.append("Pillow")
    if missing:
        st.markdown(
            '<div class="notice-warn">Missing packages: '
            + ", ".join(f"<code>{m}</code>" for m in missing)
            + "</div>", unsafe_allow_html=True,
        )
        if not HAS_DEEPFACE:
            st.markdown(
                f'<div class="notice-warn">DeepFace Error: {DEEPFACE_ERROR}</div>',
                unsafe_allow_html=True
            )
    else:
        st.markdown('<div class="notice-ok">All packages installed and ready.</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Category Reference</div>', unsafe_allow_html=True)
    for group_label, cats in GROUPS.items():
        with st.expander(group_label):
            for cat in cats:
                st.caption(f"**{cat}**")
                st.caption(DESCRIPTIONS.get(cat, ""))

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">COMPAS Dataset</div>', unsafe_allow_html=True)
    st.caption(
        "**Source:** ProPublica — Broward County, Florida.\n\n"
        "**Records:** ~4,700 defendants screened 2013–2014.\n\n"
        "**Target:** `is_violent_recid` — violent re-offence within 2 years.\n\n"
        "*For educational and research demonstration only. Not for judicial use.*"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  MASTHEAD
# ══════════════════════════════════════════════════════════════════════════════

nb_clf   = load_nb_model()
rf_model, rf_report, rf_cm, feature_imp, n_records, X_test, y_test = train_bail_model()
df_full, feature_cols = load_compas_data()

st.markdown("""
<div class="masthead">
    <div class="masthead-wordmark">Lex<span>AI</span></div>
    <div class="masthead-rule"></div>
    <div class="masthead-descriptor">Legal Intelligence Suite — Classifier · Bail Risk · Biometric Identification</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice-info">
    A unified analytical platform combining three independent modules: a Multinomial Naive Bayes
    <strong>legal case classifier</strong> across 11 Indian legal categories; a Random Forest
    <strong>bail risk assessment tool</strong> trained on the ProPublica COMPAS violent-recidivism
    dataset (Broward County, FL, 4,700+ defendants); and a deep-learning
    <strong>facial recognition system</strong> using DeepFace for biometric identification.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_classifier, tab_bail, tab_face, tab_dataset = st.tabs([
    "Case Classifier",
    "Bail Risk Assessment",
    "Facial Recognition",
    "Dataset and Model Insights",
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — CASE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
with tab_classifier:
    col_input, col_result = st.columns([1, 1], gap="large")

    for key, default in [
        ("classified_text", ""), ("classified_label", ""),
        ("audio_transcript", ""), ("audio_engine", ""), ("error_msg", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    with col_input:
        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        tab_paste, tab_upload, tab_audio = st.tabs(["Text Input", "File Upload", "Audio Transcription"])
        active_text = ""; active_label = ""

        with tab_paste:
            pasted = st.text_area(
                "Case description",
                height=240,
                placeholder="Enter a case description, FIR summary, or legal brief…",
                label_visibility="collapsed",
            )
            if pasted.strip():
                active_text = pasted.strip(); active_label = "Pasted Text"

        with tab_upload:
            uploaded = st.file_uploader(
                "Upload case file", type=["txt","pdf","docx"],
                label_visibility="collapsed",
            )
            if uploaded:
                with st.spinner("Extracting text…"):
                    extracted = extract_text(uploaded)
                st.markdown(
                    f'<div class="notice-ok">Extracted {len(extracted):,} characters from <strong>{uploaded.name}</strong></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Preview extracted text"):
                    st.markdown(
                        f'<div class="transcript-box">{extracted[:1500]}{"…" if len(extracted) > 1500 else ""}</div>',
                        unsafe_allow_html=True,
                    )
                active_text = extracted; active_label = f"File: {uploaded.name}"

        with tab_audio:
            if not HAS_SR or not HAS_PYDUB:
                st.markdown(
                    '<div class="notice-warn">Audio transcription requires: '
                    '<code>pip install SpeechRecognition pydub</code> and <code>ffmpeg</code></div>',
                    unsafe_allow_html=True,
                )
            else:
                audio_file = st.file_uploader(
                    "Upload audio file",
                    type=["mp3","wav","m4a","mp4","ogg","flac","webm"],
                    label_visibility="collapsed",
                    key="audio_uploader",
                )
                if audio_file:
                    st.audio(audio_file)
                    if st.button("Transcribe Audio", use_container_width=True):
                        progress_bar = st.progress(0, text="Initialising…")
                        def on_progress(idx, total):
                            progress_bar.progress(int((idx/total)*100), text=f"Processing chunk {idx+1} of {total}…")
                        try:
                            t, eng = transcribe_audio(audio_file.read(), audio_file.name, on_progress)
                            progress_bar.progress(100, text="Complete.")
                            st.session_state.audio_transcript = t
                            st.session_state.audio_engine = eng
                            st.session_state.error_msg = ""
                        except Exception as e:
                            progress_bar.empty()
                            st.session_state.error_msg = str(e)
                    if st.session_state.audio_transcript:
                        st.markdown(f'<span class="engine-label">{st.session_state.audio_engine}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="transcript-box">{st.session_state.audio_transcript}</div>', unsafe_allow_html=True)
                        active_text = st.session_state.audio_transcript; active_label = "Audio Transcription"
                    if st.session_state.error_msg:
                        st.markdown(f'<div class="notice-warn">{st.session_state.error_msg}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Classify Case", type="primary", use_container_width=True):
            if active_text.strip():
                st.session_state.classified_text  = active_text
                st.session_state.classified_label = active_label
            else:
                st.session_state.classified_text  = ""
                st.warning("Please provide input before classifying.")

    with col_result:
        st.markdown('<div class="section-label">Classification Result</div>', unsafe_allow_html=True)
        if st.session_state.classified_text:
            render_classifier_results(nb_clf, st.session_state.classified_text, st.session_state.classified_label)
        else:
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-title">Awaiting Input</div>
                Provide a case description via text, file upload, or audio transcription,
                then select <strong>Classify Case</strong> to generate a prediction.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — BAIL RISK ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
with tab_bail:
    st.markdown("""
    <div class="notice-info">
        <strong>Random Forest Bail Risk Assessment</strong> — This module is trained on the
        ProPublica COMPAS violent recidivism dataset (~4,700 Broward County defendants) and
        predicts the probability of violent re-offence within two years, mapped to a 0–100
        composite risk score with a structured bail recommendation.
        <br><br>
        <em>This tool is for academic and educational demonstration only. It must not be used
        as the basis for real judicial or bail decisions.</em>
    </div>
    """, unsafe_allow_html=True)

    col_form, col_output = st.columns([1, 1.1], gap="large")

    with col_form:
        st.markdown('<div class="section-label">Defendant Parameters</div>', unsafe_allow_html=True)

        age = st.slider("Age", min_value=16, max_value=80, value=28, step=1)
        st.markdown('<div class="slider-hint">Age of defendant at time of screening</div>', unsafe_allow_html=True)

        gender = st.selectbox("Gender", ["Male", "Female"])

        employment = st.selectbox(
            "Employment Status",
            ["Employed (Full-time)", "Employed (Part-time)", "Unemployed", "Student", "Retired"],
        )

        prev_convictions = st.slider("Prior Convictions", 0, 30, 2)
        st.markdown('<div class="slider-hint">Total prior adult criminal convictions on record</div>', unsafe_allow_html=True)

        num_arrests = st.slider("Juvenile Arrests (Age < 18)", 0, 20, 0)
        st.markdown('<div class="slider-hint">Total juvenile felony, misdemeanor, and other arrests</div>', unsafe_allow_html=True)

        crime_severity = st.selectbox("Charge Severity", ["Felony", "Misdemeanor"])
        charge_felony_val = 1 if crime_severity == "Felony" else 0

        community_ties = st.select_slider(
            "Community Ties",
            options=["None", "Weak", "Moderate", "Strong", "Very Strong"],
            value="Moderate",
        )

        flight_risk = st.selectbox(
            "Flight Risk Indicators",
            ["None known", "Failed to appear once", "Failed to appear multiple times", "Prior bail jump"],
        )
        flight_map = {
            "None known": 0, "Failed to appear once": 1,
            "Failed to appear multiple times": 2, "Prior bail jump": 2,
        }
        flight_val = flight_map[flight_risk]

        substance_abuse = st.selectbox(
            "Substance Abuse History",
            ["None", "Alcohol — minor", "Cannabis — minor", "Hard drugs — treated", "Hard drugs — active"],
        )
        substance_map = {
            "None": 0, "Alcohol — minor": 0, "Cannabis — minor": 0,
            "Hard drugs — treated": 1, "Hard drugs — active": 2,
        }
        substance_val = substance_map[substance_abuse]
        compas_tier = min(2, flight_val + (1 if substance_val > 0 else 0))

        st.markdown("<br>", unsafe_allow_html=True)
        assess_btn = st.button("Run Risk Assessment", type="primary", use_container_width=True)

    with col_output:
        st.markdown('<div class="section-label">Risk Assessment Output</div>', unsafe_allow_html=True)
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
            high_prob = proba[1]
            score = risk_score_from_proba(high_prob)
            community_adj = {"None": +6, "Weak": +3, "Moderate": 0, "Strong": -4, "Very Strong": -8}[community_ties]
            score = max(0, min(100, score + community_adj))

            label, color, css_cls = risk_tier(score)
            rec_text, rec_color = bail_recommendation(score, crime_severity)

            gauge_fig = render_risk_gauge(score, label, color)
            st.pyplot(gauge_fig, use_container_width=True)
            plt.close(gauge_fig)

            st.markdown(f"""
            <div class="bail-rec-box" style="background:{rec_color}18;border:1px solid {rec_color}44;">
                <div class="bail-rec-label" style="color:{rec_color};">Bail Recommendation</div>
                <div class="bail-rec-text">{rec_text}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="prob-label-title">Risk Factor Breakdown</div>', unsafe_allow_html=True)
            factors = [
                ("Age", str(age), "Below 25 indicates elevated base risk" if age < 25 else "Within moderate range"),
                ("Gender", gender, "Male defendants show higher recidivism rates in the dataset"),
                ("Prior Convictions", str(prev_convictions), ("High" if prev_convictions > 5 else "Moderate" if prev_convictions > 2 else "Low") + " prior record"),
                ("Juvenile Arrests", str(num_arrests), ("Significant" if num_arrests > 3 else "Minimal") + " juvenile history"),
                ("Charge Severity", crime_severity, "Felony charges correlate with higher recidivism risk"),
                ("Community Ties", community_ties, ("Stabilising" if community_ties in ["Strong","Very Strong"] else "Limited protective") + " anchor"),
                ("Flight Risk", flight_risk, ("Elevated" if flight_val > 0 else "No known") + " flight history"),
                ("Substance Use", substance_abuse, ("Active concern" if substance_val > 1 else "Managed or absent") + " substance factor"),
            ]
            for name, value, note in factors:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:flex-start;
                            padding:.45rem 0;border-bottom:1px solid #0d1a28;">
                    <div style="font-size:.77rem;color:#7aaccc;font-weight:600;min-width:160px;font-family:'Inter',sans-serif;">{name}</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:.75rem;color:#4a90d9;min-width:120px;text-align:center;">{value}</div>
                    <div style="font-size:.71rem;color:#2e4f6d;flex:1;text-align:right;font-family:'Inter',sans-serif;">{note}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div class="metric-note">Random Forest (300 trees · max depth 8) · '
                f'Trained on {n_records:,} COMPAS records · '
                f'Violent recidivism probability: {high_prob*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown("""
            <div class="placeholder">
                <div class="placeholder-title">Awaiting Parameters</div>
                Configure the defendant parameters in the left panel and
                select <strong>Run Risk Assessment</strong> to generate the output.
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — FACIAL RECOGNITION
# ─────────────────────────────────────────────────────────────────────────────
with tab_face:
    if not HAS_DEEPFACE or not HAS_PIL:
        st.markdown("""
        <div class="notice-warn">
            Facial recognition requires <code>deepface</code> and <code>Pillow</code>.
            Install with: <code>pip install deepface Pillow</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="notice-info">
            <strong>Biometric Identification Module</strong> — Powered by DeepFace, this module
            compares an uploaded image against a local reference dataset to identify individuals.
            The confidence score reflects the inverse of the facial distance metric.
        </div>
        """, unsafe_allow_html=True)

        col_upload_face, col_result_face = st.columns([1, 1], gap="large")

        with col_upload_face:
            st.markdown('<div class="section-label">Subject Image</div>', unsafe_allow_html=True)

            dataset_exists = os.path.exists(FACE_DATASET_PATH)
            if dataset_exists:
                dataset_size = len([
                    f for f in os.listdir(FACE_DATASET_PATH)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ])
                st.markdown(
                    f'<div class="notice-ok">Reference dataset loaded — {dataset_size} images found.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="notice-warn">Reference dataset folder not found. '
                    'Place reference images in a <code>dataset/</code> folder '
                    'adjacent to this script.</div>',
                    unsafe_allow_html=True,
                )

            uploaded_face = st.file_uploader(
                "Upload subject image",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed",
                key="face_uploader",
            )

            if uploaded_face:
                face_image = Image.open(uploaded_face)
                st.image(face_image, caption="Uploaded Subject", use_container_width=True)
                identify_btn = st.button("Run Identification", type="primary", use_container_width=True)
            else:
                identify_btn = False

        with col_result_face:
            st.markdown('<div class="section-label">Identification Result</div>', unsafe_allow_html=True)

            if uploaded_face and identify_btn:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    face_image.save(tmp.name)
                    tmp_path = tmp.name

                with st.spinner("Searching reference dataset…"):
                    try:
                        result = DeepFace.find(
                            img_path=tmp_path,
                            db_path=FACE_DATASET_PATH,
                            enforce_detection=False,
                            silent=True,
                        )
                        if len(result[0]) > 0:
                            best_match = result[0].iloc[0]
                            matched_path = best_match["identity"]
                            filename    = os.path.basename(matched_path)
                            person_name = os.path.splitext(filename)[0].replace("_", " ").title()
                            distance    = best_match["distance"]
                            confidence  = max(0.0, (1 - distance) * 100)

                            st.markdown(f"""
                            <div class="face-result-card">
                                <div class="result-eyebrow">Identified Subject</div>
                                <div class="face-result-name">{person_name}</div>
                                <div style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:#4a7aa0;margin-bottom:.3rem;">
                                    Confidence: {confidence:.2f}%
                                </div>
                                <div class="conf-bar-wrap">
                                    <div class="conf-bar-fill" style="width:{min(int(confidence),100)}%;"></div>
                                </div>
                                <div style="font-size:.7rem;color:#2e4f6d;font-family:'Inter',sans-serif;">
                                    Facial distance: {distance:.4f}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            st.image(
                                matched_path,
                                caption=f"Best match — {person_name}",
                                use_container_width=True,
                            )
                        else:
                            st.markdown(
                                '<div class="notice-warn">No match found in the reference dataset.</div>',
                                unsafe_allow_html=True,
                            )
                    except Exception as e:
                        st.markdown(
                            f'<div class="notice-warn">Identification error: {e}</div>',
                            unsafe_allow_html=True,
                        )
                    finally:
                        os.unlink(tmp_path)

            elif not uploaded_face:
                st.markdown("""
                <div class="placeholder">
                    <div class="placeholder-title">Awaiting Subject Image</div>
                    Upload a subject photograph and select
                    <strong>Run Identification</strong> to query the reference dataset.
                </div>
                """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — DATASET AND MODEL INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
with tab_dataset:
    st.markdown('<div class="section-label">COMPAS Dataset Overview</div>', unsafe_allow_html=True)

    total_records       = len(df_full)
    violent_recid_rate  = df_full["target"].mean() * 100
    avg_age             = df_full["age"].mean()
    avg_priors          = df_full["priors_count"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{total_records:,}", help="After preprocessing and cleaning")
    c2.metric("Violent Recidivism Rate", f"{violent_recid_rate:.1f}%", help="is_violent_recid within 2 years")
    c3.metric("Average Age", f"{avg_age:.1f} yrs")
    c4.metric("Avg. Prior Convictions", f"{avg_priors:.1f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_ds1, col_ds2 = st.columns([1.1, 1], gap="large")

    with col_ds1:
        st.markdown("#### Dataset Field Reference")
        field_info = [
            ("age",                "Defendant age at COMPAS screening",              "Numeric (16–96)"),
            ("sex",                "Gender of defendant",                             "Male / Female"),
            ("race",               "Race / ethnicity of defendant",                   "African-American, Caucasian, Hispanic, Other"),
            ("priors_count",       "Total prior adult criminal convictions",          "Numeric (0–38)"),
            ("juv_fel_count",      "Juvenile felony arrests",                         "Numeric"),
            ("juv_misd_count",     "Juvenile misdemeanor arrests",                    "Numeric"),
            ("juv_other_count",    "Other juvenile arrests",                          "Numeric"),
            ("c_charge_degree",    "Severity of current charge",                      "F (Felony) / M (Misdemeanor)"),
            ("c_charge_desc",      "Textual description of current charge",           "Free text"),
            ("v_decile_score",     "COMPAS violent risk decile score",                "1 (Low) — 10 (High)"),
            ("v_score_text",       "COMPAS violent risk text label",                  "Low / Medium / High"),
            ("is_violent_recid",   "Violent re-offence within 2 years (target)",      "0 = No, 1 = Yes"),
            ("two_year_recid",     "Any recidivism within 2 years",                   "0 = No, 1 = Yes"),
            ("compas_screening_date", "Date of COMPAS screening",                     "Date (2013–2014)"),
        ]
        rows = "".join(
            f"<tr><td><code>{f}</code></td><td>{d}</td><td style='color:#3a6a94'>{v}</td></tr>"
            for f, d, v in field_info
        )
        st.markdown(
            f'<table class="ds-table"><tr><th>Field</th><th>Description</th><th>Values / Type</th></tr>{rows}</table>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Random Forest — Model Performance")
        if "1" in rf_report:
            prec = rf_report["1"]["precision"]
            rec  = rf_report["1"]["recall"]
            f1   = rf_report["1"]["f1-score"]
            acc  = rf_report["accuracy"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc*100:.1f}%")
            m2.metric("Precision", f"{prec*100:.1f}%")
            m3.metric("Recall", f"{rec*100:.1f}%")
            m4.metric("F1 Score", f"{f1:.3f}")

        st.markdown("#### Model Configuration")
        st.code("""RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 8,
    min_samples_leaf = 10,
    class_weight     = 'balanced',
    random_state     = 42,
    n_jobs           = -1
)

Features:  age, sex_enc, priors_count, juv_total,
           charge_felony, compas_risk_tier

Target:    is_violent_recid  (0 = No, 1 = Yes)
Split:     80% train / 20% test  (stratified)
""", language="python")

    with col_ds2:
        st.markdown("#### Feature Importance")
        imp_fig = render_feature_importance(feature_imp)
        st.pyplot(imp_fig, use_container_width=True)
        plt.close(imp_fig)

        st.markdown("#### Violent Recidivism by Age Group")
        age_bins = pd.cut(df_full["age"], bins=[15, 25, 35, 45, 55, 100],
                          labels=["16–25", "26–35", "36–45", "46–55", "55+"])
        age_rec = df_full.groupby(age_bins, observed=True)["target"].mean() * 100

        fig_age, ax_age = plt.subplots(figsize=(5, 3), facecolor="none")
        bars = ax_age.bar(age_rec.index, age_rec.values, color="#1e5090", edgecolor="none", width=0.6)
        for bar, val in zip(bars, age_rec.values):
            ax_age.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                        f"{val:.1f}%", ha="center", color="#4a8ab8", fontsize=8)
        ax_age.set_ylabel("Violent Recidivism Rate (%)", color="#3a6a94", fontsize=8)
        ax_age.set_xlabel("Age Group", color="#3a6a94", fontsize=8)
        ax_age.tick_params(colors="#3a6a94", labelsize=8)
        ax_age.spines[:].set_visible(False)
        ax_age.set_facecolor("none"); fig_age.patch.set_alpha(0.0)
        for lbl in ax_age.get_xticklabels() + ax_age.get_yticklabels():
            lbl.set_color("#7aaccc")
        st.pyplot(fig_age, use_container_width=True)
        plt.close(fig_age)

        st.markdown("#### COMPAS Score vs. Actual Recidivism")
        score_rec = df_full.groupby("compas_risk_tier", observed=True)["target"].mean() * 100
        tier_labels = {0: "Low (1–3)", 1: "Medium (4–6)", 2: "High (7–10)"}
        tier_colors = ["#1e6a38", "#9a6010", "#8a1818"]

        fig_tier, ax_tier = plt.subplots(figsize=(5, 2.5), facecolor="none")
        xs = [tier_labels.get(i, str(i)) for i in score_rec.index]
        ys = score_rec.values
        colors_t = tier_colors[:len(xs)]
        bars_t = ax_tier.bar(xs, ys, color=colors_t, edgecolor="none", width=0.5)
        for bar, val in zip(bars_t, ys):
            ax_tier.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                         f"{val:.1f}%", ha="center", color="#c8d8e8", fontsize=9, fontweight="bold")
        ax_tier.set_ylabel("Actual Violent Recidivism (%)", color="#3a6a94", fontsize=8)
        ax_tier.tick_params(colors="#3a6a94", labelsize=8)
        ax_tier.spines[:].set_visible(False)
        ax_tier.set_facecolor("none"); fig_tier.patch.set_alpha(0.0)
        for lbl in ax_tier.get_xticklabels() + ax_tier.get_yticklabels():
            lbl.set_color("#7aaccc")
        st.pyplot(fig_tier, use_container_width=True)
        plt.close(fig_tier)

        st.markdown("#### Sample Records")
        sample_cols = ["age", "sex", "priors_count", "c_charge_degree", "v_score_text", "is_violent_recid"]
        available_sample = [c for c in sample_cols if c in df_full.columns]
        st.dataframe(
            df_full[available_sample].head(8).rename(columns={
                "age": "Age", "sex": "Gender", "priors_count": "Priors",
                "c_charge_degree": "Charge", "v_score_text": "COMPAS Score",
                "is_violent_recid": "Violent Recidivism",
            }),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="notice-warn">
        <strong>Disclaimer</strong> — This platform is constructed for academic and educational
        demonstration of machine learning applied to legal contexts. The COMPAS dataset and any
        algorithmic risk score produced by this tool must not serve as the sole or primary basis
        for bail, detention, or any judicial decision. ProPublica's analysis documented significant
        racial disparities in COMPAS scores. All real bail determinations require judicial
        discretion, full case context, and the participation of legal counsel.
    </div>
    """, unsafe_allow_html=True)
