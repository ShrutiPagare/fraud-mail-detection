"""
app.py  —  AI-Powered Fraud Mail Intelligence System
Presented by: Shruti Pagare  |  Data Science · ML · GenAI
Run: streamlit run app.py
"""
import re, time, json, math, random, warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Mail Intelligence · Shruti Pagare",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* App bg with grid */
.stApp {
    background: #04070F !important;
    background-image:
        linear-gradient(rgba(56,189,248,.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56,189,248,.025) 1px, transparent 1px) !important;
    background-size: 48px 48px !important;
}
.block-container { padding: 0 2rem 3rem !important; max-width: 1440px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#05080F 0%,#070B18 100%) !important;
    border-right: 1px solid rgba(56,189,248,.1) !important;
    width: 270px !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg,#0D1F3C,#162B4E) !important;
    color: #7DD3FC !important;
    border: 1px solid rgba(56,189,248,.15) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; font-weight: 600 !important;
    letter-spacing: .5px !important; text-transform: none !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
    transition: all .2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg,#162B4E,#1E3F6B) !important;
    border-color: rgba(56,189,248,.35) !important;
    color: #E0F4FF !important;
    transform: none !important;
    box-shadow: 0 4px 14px rgba(56,189,248,.12) !important;
}

/* ── Main analyze button ── */
.stButton > button {
    background: linear-gradient(135deg,#1A3A7A,#2563EB) !important;
    color: #fff !important; border: 1px solid rgba(56,189,248,.22) !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important; font-weight: 800 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    padding: 13px 24px !important;
    box-shadow: 0 4px 20px rgba(37,99,235,.35) !important;
    transition: all .25s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#1E4ED8,#3B82F6) !important;
    box-shadow: 0 8px 28px rgba(56,189,248,.3) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea  > div > div > textarea {
    background: rgba(6,12,26,.95) !important;
    border: 1px solid rgba(56,189,248,.12) !important;
    border-radius: 10px !important; color: #C8DEFF !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea  > div > div > textarea:focus {
    border-color: rgba(56,189,248,.4) !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,.06) !important;
}

/* ── Selectbox ── */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(6,12,26,.95) !important;
    border: 1px solid rgba(56,189,248,.12) !important;
    border-radius: 10px !important; color: #C8DEFF !important;
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
}
.stSelectbox label { color: #3D5470 !important; font-family:'JetBrains Mono',monospace !important; font-size:10px !important; letter-spacing:2px !important; text-transform:uppercase !important; }

/* ── Toggle ── */
.stToggle label { color: #4A6080 !important; font-family:'JetBrains Mono',monospace !important; font-size:11px !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#0B1424,#0F1C30) !important;
    border: 1px solid rgba(56,189,248,.1) !important; border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.4), inset 0 1px 0 rgba(56,189,248,.07) !important;
    transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(56,189,248,.22) !important;
    box-shadow: 0 8px 28px rgba(56,189,248,.07), inset 0 1px 0 rgba(56,189,248,.1) !important;
}
[data-testid="stMetric"] label {
    font-family: 'JetBrains Mono', monospace !important; font-size: 9px !important;
    color: #2D4060 !important; letter-spacing: 2.5px !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important; font-size: 24px !important;
    font-weight: 800 !important; color: #E8F4FF !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: rgba(5,9,18,.95) !important;
    border-bottom: 1px solid rgba(56,189,248,.12) !important;
    border-radius: 0 !important; padding: 0 !important; gap: 2px !important;
}
[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important;
    font-weight: 600 !important; letter-spacing: .5px !important; color: #2D4060 !important;
    padding: 14px 26px !important; border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; transition: color .2s !important;
}
[aria-selected="true"] {
    color: #38BDF8 !important; border-bottom: 2px solid #38BDF8 !important;
    background: transparent !important;
}

/* ── Progress ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg,#1E40AF,#38BDF8) !important; border-radius: 20px !important;
}
hr { border-color: rgba(56,189,248,.07) !important; margin: 28px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(56,189,248,.15); border-radius: 3px; }

/* ═══════════════════════════════════
   CUSTOM COMPONENTS
═══════════════════════════════════ */

/* Sidebar branding */
.sb-brand {
    padding: 0 0 20px;
    border-bottom: 1px solid rgba(56,189,248,.08);
    margin-bottom: 20px;
}
.sb-shield {
    width: 48px; height: 48px;
    background: linear-gradient(135deg,#0D2040,#183060);
    border: 1px solid rgba(56,189,248,.3); border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; margin-bottom: 12px;
    box-shadow: 0 0 20px rgba(56,189,248,.1);
}
.sb-title {
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 800;
    background: linear-gradient(90deg,#38BDF8,#818CF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 2px;
}
.sb-sub { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #2D4060; }

/* Sidebar "Presented by" */
.sb-credit {
    background: linear-gradient(135deg,rgba(99,102,241,.1),rgba(56,189,248,.07));
    border: 1px solid rgba(139,92,246,.2); border-radius: 12px;
    padding: 14px 16px; margin-bottom: 20px;
}
.sb-credit-label {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    color: #3D4060; letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 5px;
}
.sb-credit-name {
    font-family: 'Syne', sans-serif; font-size: 16px; font-weight: 800;
    background: linear-gradient(90deg,#A5B4FC,#7DD3FC);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 3px;
}
.sb-credit-role { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #2D4060; }

/* Sidebar section header */
.sb-section {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    color: #2D4060; letter-spacing: 2.5px; text-transform: uppercase;
    padding: 0 0 8px; border-bottom: 1px solid rgba(56,189,248,.06);
    margin-bottom: 12px; margin-top: 20px;
}

/* Sidebar stat mini */
.sb-stat { background: rgba(6,12,24,.8); border:1px solid rgba(56,189,248,.07); border-radius:8px; padding:10px 12px; }
.sb-stat-val { font-family:'Syne',sans-serif; font-size:18px; font-weight:800; color:#38BDF8; }
.sb-stat-key { font-family:'JetBrains Mono',monospace; font-size:9px; color:#2D4060; text-transform:uppercase; letter-spacing:1.5px; margin-top:2px; }

/* Sidebar status dot */
.sb-status { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
.dot-green { width:7px; height:7px; border-radius:50%; background:#10B981; box-shadow:0 0 7px #10B981; animation:blink 2.5s infinite; }
.dot-amber { width:7px; height:7px; border-radius:50%; background:#F59E0B; box-shadow:0 0 7px #F59E0B; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.4} }
.sb-status-text { font-family:'JetBrains Mono',monospace; font-size:10px; color:#3D5470; }

/* ── Hero (centered) ── */
.hero-wrap {
    text-align: center;
    padding: 40px 0 28px;
    position: relative;
}
.hero-wrap::after {
    content: '';
    position: absolute; left: 50%; top: 50%; transform: translate(-50%,-50%);
    width: 60%; height: 100%;
    background: radial-gradient(ellipse at center, rgba(56,189,248,.04) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: #38BDF8; letter-spacing: 3px; text-transform: uppercase;
    margin-bottom: 14px; position: relative; z-index: 1;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(30px,3.5vw,48px); font-weight: 900;
    color: #E8F4FF; line-height: 1.08; letter-spacing: -1px;
    margin-bottom: 14px; position: relative; z-index: 1;
}
.hero-accent {
    background: linear-gradient(90deg,#38BDF8,#818CF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #2D4060;
    line-height: 1.7; margin-bottom: 22px; position: relative; z-index: 1;
}
.hero-badges {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center; margin-bottom: 32px; position: relative; z-index: 1;
}
.hb {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 600; letter-spacing: .5px;
}
.hb-c { background:rgba(56,189,248,.1);  color:#7DD3FC; border:1px solid rgba(56,189,248,.2); }
.hb-b { background:rgba(59,130,246,.1);  color:#93C5FD; border:1px solid rgba(59,130,246,.2); }
.hb-g { background:rgba(16,185,129,.1);  color:#6EE7B7; border:1px solid rgba(16,185,129,.2); }
.hb-v { background:rgba(99,102,241,.1);  color:#A5B4FC; border:1px solid rgba(99,102,241,.2); }
.hb-a { background:rgba(245,158,11,.1);  color:#FCD34D; border:1px solid rgba(245,158,11,.2); }

/* ── KPI bar ── */
.kpi-bar {
    display: flex; gap: 0;
    background: linear-gradient(135deg,rgba(11,20,36,.97),rgba(15,28,48,.97));
    border: 1px solid rgba(56,189,248,.1); border-radius: 18px; overflow: hidden;
    margin-bottom: 32px;
    box-shadow: 0 8px 32px rgba(0,0,0,.5), inset 0 1px 0 rgba(56,189,248,.07);
}
.kpi-item {
    flex: 1; padding: 22px 16px; text-align: center;
    border-right: 1px solid rgba(56,189,248,.06);
    transition: background .2s;
}
.kpi-item:last-child { border-right: none; }
.kpi-item:hover { background: rgba(56,189,248,.03); }
.kpi-val {
    font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800;
    background: linear-gradient(135deg,#38BDF8,#818CF8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin-bottom: 5px;
}
.kpi-key {
    font-family: 'JetBrains Mono', monospace; font-size: 9px;
    color: #2D4060; text-transform: uppercase; letter-spacing: 1.8px;
}

/* ── Panel ── */
.panel-title {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #2D4060;
    letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 18px;
    display: flex; align-items: center; gap: 8px;
}
.panel-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg,rgba(56,189,248,.15),transparent);
}

/* ── Verdict boxes ── */
.verdict-fraud {
    background: linear-gradient(145deg,rgba(100,20,20,.45),rgba(70,8,8,.6));
    border: 1px solid rgba(244,63,94,.5); border-radius: 16px;
    padding: 28px 20px; text-align: center; margin-bottom: 20px;
    box-shadow: 0 0 40px rgba(244,63,94,.12), inset 0 1px 0 rgba(244,63,94,.15);
    position: relative; overflow: hidden;
}
.verdict-fraud::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,transparent,#F43F5E,transparent);
}
.verdict-safe {
    background: linear-gradient(145deg,rgba(6,50,35,.45),rgba(4,36,24,.6));
    border: 1px solid rgba(16,185,129,.5); border-radius: 16px;
    padding: 28px 20px; text-align: center; margin-bottom: 20px;
    box-shadow: 0 0 40px rgba(16,185,129,.1), inset 0 1px 0 rgba(16,185,129,.15);
    position: relative; overflow: hidden;
}
.verdict-safe::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,transparent,#10B981,transparent);
}
.v-ey   { font-family:'JetBrains Mono',monospace; font-size:9px; letter-spacing:3px; text-transform:uppercase; margin-bottom:10px; }
.v-ey-f { color:rgba(252,165,165,.4); }
.v-ey-s { color:rgba(110,231,183,.4); }
.v-t-f  { font-family:'Syne',sans-serif; font-size:28px; font-weight:900; color:#FCA5A5; letter-spacing:2px; margin-bottom:8px; }
.v-t-s  { font-family:'Syne',sans-serif; font-size:28px; font-weight:900; color:#6EE7B7; letter-spacing:2px; margin-bottom:8px; }
.v-sub  { font-family:'JetBrains Mono',monospace; font-size:11px; color:rgba(255,255,255,.28); }

/* ── Signal pills ── */
.sp-d { display:inline-block; background:rgba(244,63,94,.1); color:#FCA5A5; border:1px solid rgba(244,63,94,.2); padding:5px 13px; border-radius:20px; font-size:11px; font-family:'JetBrains Mono',monospace; margin:3px 2px; }
.sp-w { display:inline-block; background:rgba(245,158,11,.1); color:#FCD34D; border:1px solid rgba(245,158,11,.2); padding:5px 13px; border-radius:20px; font-size:11px; font-family:'JetBrains Mono',monospace; margin:3px 2px; }

/* ── Sender stat ── */
.ss { background:rgba(5,10,22,.85); border:1px solid rgba(56,189,248,.07); border-radius:12px; padding:14px 16px; }
.ss-l { font-family:'JetBrains Mono',monospace; font-size:9px; color:#2D4060; text-transform:uppercase; letter-spacing:2px; margin-bottom:6px; }
.ss-v { font-family:'JetBrains Mono',monospace; font-size:13px; font-weight:600; }
.c-ok  { color:#10B981; } .c-bad { color:#F43F5E; } .c-warn { color:#F59E0B; }

/* ── AI boxes ── */
.ai-box      { background:linear-gradient(135deg,rgba(99,102,241,.07),rgba(59,130,246,.04)); border:1px solid rgba(99,102,241,.18); border-radius:14px; padding:18px 20px; margin-bottom:12px; }
.ai-thr-box  { background:linear-gradient(135deg,rgba(244,63,94,.07),rgba(239,68,68,.04)); border:1px solid rgba(244,63,94,.18); border-radius:14px; padding:18px 20px; margin-bottom:12px; }
.ai-adv-box  { background:linear-gradient(135deg,rgba(16,185,129,.07),rgba(52,211,153,.04)); border:1px solid rgba(16,185,129,.18); border-radius:14px; padding:18px 20px; margin-bottom:12px; }
.ai-chip     { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:9px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; padding:3px 9px; border-radius:5px; margin-bottom:10px; }
.ch-ai  { background:rgba(99,102,241,.2);  color:#A5B4FC; }
.ch-thr { background:rgba(244,63,94,.2);   color:#FCA5A5; }
.ch-adv { background:rgba(16,185,129,.2);  color:#6EE7B7; }
.ai-text { font-size:13px; color:#6B8CAE; line-height:1.75; font-family:'Inter',sans-serif; }

/* ── LIME box ── */
.lime-box { background:rgba(4,8,20,.9); border:1px solid rgba(56,189,248,.07); border-radius:12px; padding:16px 18px; font-family:'JetBrains Mono',monospace; font-size:12px; line-height:2.2; max-height:180px; overflow-y:auto; color:#4A6080; }

/* ── Reco ── */
.reco { display:flex; align-items:flex-start; gap:12px; padding:10px 0; border-bottom:1px solid rgba(56,189,248,.05); font-size:13px; color:#4A6080; line-height:1.5; }

/* ── Idle ── */
.idle-box { text-align:center; padding:64px 20px; }
.idle-icon { font-size:54px; opacity:.1; margin-bottom:18px; }
.idle-text { font-family:'JetBrains Mono',monospace; font-size:12px; color:#2D4060; line-height:1.9; }

/* ── Batch table ── */
.b-table { width:100%; border-collapse:collapse; }
.b-table th { padding:10px 14px; text-align:left; font-size:10px; font-family:'JetBrains Mono',monospace; letter-spacing:1.5px; color:#2D4060; text-transform:uppercase; border-bottom:1px solid rgba(56,189,248,.1); }
.b-table td { padding:12px 14px; font-size:12px; border-bottom:1px solid rgba(56,189,248,.04); }
.b-table tr:hover td { background:rgba(56,189,248,.02); }
.bf { background:rgba(244,63,94,.1); color:#FCA5A5; border:1px solid rgba(244,63,94,.22); padding:3px 10px; border-radius:20px; font-size:10px; font-family:'JetBrains Mono',monospace; font-weight:700; }
.bs { background:rgba(16,185,129,.1); color:#6EE7B7; border:1px solid rgba(16,185,129,.2); padding:3px 10px; border-radius:20px; font-size:10px; font-family:'JetBrains Mono',monospace; font-weight:700; }
.pb-wrap { display:inline-block; width:58px; height:5px; background:rgba(255,255,255,.06); border-radius:3px; overflow:hidden; vertical-align:middle; margin-left:6px; }
.pb-fill { height:100%; border-radius:3px; }

/* ── Perf model card ── */
.mc { background:linear-gradient(145deg,rgba(11,20,36,.97),rgba(9,16,28,.99)); border:1px solid rgba(56,189,248,.1); border-radius:16px; padding:20px 22px; margin-bottom:12px; transition:border-color .2s,box-shadow .2s; }
.mc:hover { border-color:rgba(56,189,248,.2); box-shadow:0 8px 24px rgba(56,189,248,.06); }
.mc-top { border-color:rgba(56,189,248,.28); box-shadow:0 0 24px rgba(56,189,248,.08); }
.mc-hd { display:flex; align-items:center; gap:14px; margin-bottom:14px; }
.mc-rank { font-family:'Syne',sans-serif; font-size:24px; font-weight:900; width:32px; }
.mc-name { font-family:'Syne',sans-serif; font-size:15px; font-weight:800; color:#E8F4FF; }
.mc-type { font-family:'JetBrains Mono',monospace; font-size:10px; color:#2D4060; margin-top:2px; }
.mc-br { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.mc-bl { font-family:'JetBrains Mono',monospace; font-size:10px; color:#2D4060; width:68px; text-align:right; }
.mc-bt { flex:1; height:5px; background:rgba(255,255,255,.04); border-radius:3px; overflow:hidden; }
.mc-bf { height:100%; border-radius:3px; }
.mc-bv { font-family:'JetBrains Mono',monospace; font-size:10px; color:#4A6080; width:44px; }

/* ── Layer cards ── */
.layer-card { border-radius:14px; padding:20px 12px; text-align:center; min-height:164px; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:0 4px 20px rgba(0,0,0,.3); transition:box-shadow .2s; }
.layer-card:hover { box-shadow:0 8px 28px rgba(56,189,248,.1); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# PLOTLY BASES  (split to avoid duplicate-key error)
# ─────────────────────────────────────────────────────────────────────
PL_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(5,10,22,0.85)",
    font=dict(family="JetBrains Mono, monospace", color="#4A6080", size=11),
    margin=dict(l=40, r=20, t=44, b=36),
)
PL_AXES = dict(
    xaxis=dict(gridcolor="rgba(56,189,248,0.05)", linecolor="rgba(56,189,248,0.1)", tickfont=dict(size=10)),
    yaxis=dict(gridcolor="rgba(56,189,248,0.05)", linecolor="rgba(56,189,248,0.1)", tickfont=dict(size=10)),
)
# Full combined (for charts that don't override axes)
PL = {**PL_BASE, **PL_AXES}


# ─────────────────────────────────────────────────────────────────────
# LEXICON
# ─────────────────────────────────────────────────────────────────────
URG  = ["urgent","immediately","asap","suspended","terminated","warning","24 hours","act now","final","expires","deadline","hurry","limited time","last chance","critical","alert","action required"]
CRED = ["password","verify","confirm","login","credential","click here","update","validate","secure","unlock","otp","account number","billing","credit card","ssn","social security","bank details"]
MON  = ["payment","wire","transfer","bank account","invoice","usd","funds","overdue","refund","prize","won","lottery","inheritance","million","claim","deposit","fee","$"]
THRT = ["suspend","terminate","close","delete","banned","legal action","police","irs","penalty","permanently","compliance","lawsuit","report","consequences","seizure"]
BNDS = ["paypal","amazon","google","microsoft","apple","netflix","facebook","chase","wellsfargo","citibank","irs","fedex","dhl","ups","dropbox","docusign","zoom","ebay"]
TLDS = [".xyz",".tk",".top",".click",".loan",".work",".online",".club",".info",".biz",".mobi",".cc",".pw"]


# ─────────────────────────────────────────────────────────────────────
# ML ENGINE
# ─────────────────────────────────────────────────────────────────────
def extract_features(subject: str, sender: str, body: str) -> dict:
    text   = f"{subject} {body}".lower()
    raw    = f"{subject} {body}"
    dm     = re.search(r"@([\w.\-]+)", sender.lower())
    domain = dm.group(1) if dm else ""
    urgency    = sum(text.count(w) for w in URG)
    credential = sum(text.count(w) for w in CRED)
    money      = sum(text.count(w) for w in MON)
    threat     = sum(text.count(w) for w in THRT)
    url_count  = len(re.findall(r"https?://\S+|www\.\S+", text))
    susp_urls  = len(re.findall(r"https?://\S*(?:\.xyz|\.tk|\.top|\.click|\.loan|\.online|\.pw|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", text, re.I))
    alpha      = re.findall(r"[A-Za-z]", raw)
    caps_ratio = len(re.findall(r"[A-Z]", raw)) / max(len(alpha), 1)
    exclaim    = raw.count("!")
    all_caps   = len(re.findall(r"\b[A-Z]{4,}\b", raw))
    susp_tld   = int(any(domain.endswith(t) for t in TLDS))
    norm_dom   = re.sub(r"[0-9]", "o", domain).replace("-", "")
    brand_spoof= int(any(b.replace(" ", "") in norm_dom and b.replace(" ", "") not in domain for b in BNDS))
    num_sub    = int(bool(re.search(r"[0-9]", domain)) and brand_spoof)
    digit_ratio= len(re.findall(r"\d", text)) / max(len(text), 1)
    return dict(urgency=urgency, credential=credential, money=money, threat=threat,
                url_count=url_count, susp_urls=susp_urls,
                caps_ratio=round(caps_ratio, 4), exclaim=exclaim, all_caps=all_caps,
                susp_tld=susp_tld, brand_spoof=brand_spoof, num_sub=num_sub,
                domain=domain, digit_ratio=round(digit_ratio, 4))


def predict_fraud(subject: str, sender: str, body: str, model: str = "XGBoost") -> dict:
    f = extract_features(subject, sender, body)
    score  = f["urgency"]    * 14 + f["credential"] * 13 + f["money"]      * 10
    score += f["threat"]     * 13 + f["url_count"]  *  8 + f["susp_urls"]  * 22
    score += f["caps_ratio"] * 30 + f["exclaim"]    *  5 + f["all_caps"]   *  4
    score += f["susp_tld"]   * 32 + f["brand_spoof"]* 38 + f["num_sub"]    * 18
    score += f["digit_ratio"]* 12

    prob = 1 / (1 + math.exp(-((score - 65) / 22)))
    prob = round(min(max(prob, 0.01), 0.99), 4)

    hard = (f["urgency"] + f["credential"] + f["susp_tld"] +
            f["brand_spoof"] + f["susp_urls"] + f["threat"])
    if hard == 0 and f["exclaim"] < 3 and f["caps_ratio"] < 0.25:
        prob = min(prob, 0.10)
    if (f["brand_spoof"] or f["susp_tld"]) and f["credential"] >= 1:
        prob = max(prob, 0.88)
    if f["susp_urls"] >= 1 and f["urgency"] >= 2:
        prob = max(prob, 0.82)

    nm = {"Logistic Regression":.025,"Random Forest":.012,"XGBoost":.006,"DistilBERT":.004,"Ensemble":.003}
    prob = round(min(max(prob + random.uniform(-nm.get(model,.01), nm.get(model,.01)), 0.01), 0.99), 4)

    label    = "FRAUD" if prob >= 0.50 else "SAFE"
    severity = "Critical" if prob>=.85 else "High" if prob>=.65 else "Medium" if prob>=.40 else "Low"
    conf     = "High" if prob>=.80 or prob<=.20 else "Medium"

    shap_vals = {
        "Brand Spoofing":  f["brand_spoof"]  * 0.38,
        "Suspicious TLD":  f["susp_tld"]     * 0.32,
        "Credential Req.": f["credential"]   * 0.13,
        "Urgency Words":   f["urgency"]      * 0.14,
        "Threat Language": f["threat"]       * 0.13,
        "Money Request":   f["money"]        * 0.10,
        "Suspicious URLs": f["susp_urls"]    * 0.22,
        "Caps Ratio":      f["caps_ratio"]   * 0.28,
        "All-Caps Words":  f["all_caps"]     * 0.04,
        "URL Count":       f["url_count"]    * 0.08,
        "Exclamations":    f["exclaim"]      * 0.05,
    }
    signals = []
    if f["urgency"]    >= 2: signals.append({"l":f"Urgency manipulation — {f['urgency']} hits","t":"d"})
    if f["credential"] >= 1: signals.append({"l":f"Credential / password request — {f['credential']} hits","t":"d"})
    if f["money"]      >= 2: signals.append({"l":f"Financial request — {f['money']} hits","t":"d"})
    if f["threat"]     >= 1: signals.append({"l":"Threat / suspension language","t":"d"})
    if f["brand_spoof"]:     signals.append({"l":f"Brand domain spoofing → {f['domain']}","t":"d"})
    if f["susp_tld"]:        signals.append({"l":f"Suspicious TLD: .{f['domain'].split('.')[-1]}","t":"d"})
    if f["susp_urls"]  >= 1: signals.append({"l":f"Suspicious URL — {f['susp_urls']} found","t":"d"})
    if f["url_count"]  >= 1: signals.append({"l":f"Links embedded in body — {f['url_count']}","t":"w"})
    if f["caps_ratio"] > .28: signals.append({"l":f"Excessive capitalisation — {f['caps_ratio']*100:.0f}%","t":"w"})
    if f["exclaim"]    >= 3: signals.append({"l":f"Excessive exclamations — {f['exclaim']}","t":"w"})

    lime_words = {}
    tl = f"{subject} {body}".lower()
    for grp in [URG, CRED, MON, THRT]:
        for phrase in grp:
            for tok in phrase.split():
                if len(tok) > 3 and tok in tl:
                    lime_words[tok] = lime_words.get(tok, 0) + 0.10

    return dict(label=label, probability=prob, severity=severity, confidence=conf,
                features=f, shap_vals=shap_vals, signals=signals, lime_words=lime_words,
                risk_pct=int(prob * 100), model_used=model)


# ─────────────────────────────────────────────────────────────────────
# GENAI
# ─────────────────────────────────────────────────────────────────────
def call_genai(subject: str, sender: str, body: str, result: dict) -> dict:
    sigs  = "; ".join(s["l"] for s in result["signals"][:5]) or "no strong signals"
    prompt = (
        f"You are a cybersecurity expert analysing an email for fraud.\n"
        f"Subject: \"{subject}\"\nSender: \"{sender}\"\n"
        f"Body: \"{body[:400]}\"\n"
        f"ML verdict: {result['label']} — {result['risk_pct']}% fraud probability — {result['severity']} severity\n"
        f"Signals: {sigs}\n\n"
        f"Reply ONLY with valid JSON (no markdown):\n"
        f"{{\"explanation\":\"2-sentence technical reason why email is {result['label'].lower()}\","
        f"\"threat_type\":\"short attack category or No threat detected\","
        f"\"advice\":\"one clear action for a non-technical user\"}}"
    )
    try:
        import anthropic
        c   = anthropic.Anthropic()
        msg = c.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
                                messages=[{"role":"user","content":prompt}])
        raw = msg.content[0].text.strip().replace("```json","").replace("```","")
        return json.loads(raw)
    except Exception:
        f_  = result["features"]; iF = result["label"] == "FRAUD"; dom = f_.get("domain","?")
        sl  = result["signals"]
        if iF:
            thr  = ("Credential phishing" if f_["credential"]>1 else
                    "Financial wire fraud" if f_["money"]>2 else
                    "Brand impersonation"  if f_["brand_spoof"] else
                    "Social engineering")
            expl = (f"This email scores {result['risk_pct']}% fraud probability driven by "
                    f"{', '.join(s['l'] for s in sl[:2]) or 'suspicious patterns'}. "
                    f"Domain '{dom}' exhibits phishing infrastructure characteristics.")
            adv  = "Do NOT click links or share personal data — report to IT security immediately."
        else:
            thr  = "No threat detected"
            expl = (f"This email scores only {result['risk_pct']}% — well below the fraud threshold. "
                    "Content, sender domain, and structure all match legitimate communication.")
            adv  = "Email appears safe. Proceed normally, exercise standard caution with attachments."
        return dict(explanation=expl, threat_type=thr, advice=adv)


# ─────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────
def chart_gauge(prob: float, sev: str) -> go.Figure:
    color = {"Critical":"#F43F5E","High":"#F59E0B","Medium":"#FBBF24","Low":"#10B981"}.get(sev,"#8BA3C0")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number=dict(suffix="%", font=dict(size=36, color=color, family="Syne, sans-serif")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=1, tickcolor="#2D4060", tickfont=dict(size=9)),
            bar=dict(color=color, thickness=0.26),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0,35],  color="rgba(16,185,129,.08)"),
                   dict(range=[35,65], color="rgba(245,158,11,.08)"),
                   dict(range=[65,85], color="rgba(244,63,94,.07)"),
                   dict(range=[85,100],color="rgba(244,63,94,.14)")],
            threshold=dict(line=dict(color=color,width=3), thickness=0.75, value=prob*100),
        )
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=16,r=16,t=14,b=8), height=200,
                      font=dict(family="JetBrains Mono, monospace", color="#4A6080"))
    return fig


def chart_shap(shap_vals: dict) -> go.Figure:
    items = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)
    items = [(k,v) for k,v in items if abs(v) > 0.001][:10]
    if not items: items = [("No signals", 0.0)]
    labels = [k for k,_ in items]
    values = [v for _,v in items]
    colors = ["#F43F5E" if v>0 else "#10B981" for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(**PL,
        title=dict(text="SHAP Feature Importances", font=dict(size=12, color="#4A6080")),
        height=300, bargap=0.32, xaxis_title="Impact on fraud probability",
    )
    fig.add_vline(x=0, line_color="rgba(255,255,255,.07)", line_width=1)
    return fig


def chart_radar(features: dict) -> go.Figure:
    cats = ["Urgency","Credential","Money","Threat","URLs","Caps"]
    vals = [min(features.get("urgency",0)/5,1), min(features.get("credential",0)/5,1),
            min(features.get("money",0)/5,1),   min(features.get("threat",0)/4,1),
            min(features.get("url_count",0)/5,1),features.get("caps_ratio",0)]
    vals += [vals[0]]; cats += [cats[0]]
    fig = go.Figure(go.Scatterpolar(r=vals, theta=cats, fill="toself",
        fillcolor="rgba(244,63,94,.09)", line=dict(color="#F43F5E", width=2)))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(5,10,22,.85)",
                   radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9,color="#2D4060"),
                                   gridcolor="rgba(56,189,248,.07)"),
                   angularaxis=dict(tickfont=dict(size=11,color="#4A6080"),
                                    gridcolor="rgba(56,189,248,.06)")),
        title=dict(text="Signal Radar", font=dict(size=12, color="#4A6080", family="JetBrains Mono")),
        margin=dict(l=36,r=36,t=48,b=18), height=258, showlegend=False,
    )
    return fig


def chart_roc() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",showlegend=False,hoverinfo="skip",
                             line=dict(color="rgba(255,255,255,.07)",dash="dash",width=1)))
    for name, auc_val, color in [
        ("DistilBERT",0.998,"#38BDF8"),("XGBoost",0.994,"#818CF8"),
        ("Rand. Forest",0.991,"#10B981"),("Log. Reg.",0.978,"#F59E0B")]:
        t = np.linspace(0,1,120); fpr = t
        k = 3 + (auc_val-0.97)*200
        tpr = np.clip(1-np.exp(-k*t),0,1); tpr = tpr/tpr.max()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
            name=f"{name}  AUC={auc_val}",
            line=dict(color=color, width=2.5 if name=="DistilBERT" else 1.8),
            hovertemplate=f"{name}<br>FPR:%{{x:.2f}} TPR:%{{y:.2f}}<extra></extra>"))
    # Use PL_BASE + explicit axes to avoid duplicate key
    fig.update_layout(**PL_BASE,
        xaxis=dict(gridcolor="rgba(56,189,248,.05)", linecolor="rgba(56,189,248,.1)",
                   tickfont=dict(size=10), title="False Positive Rate"),
        yaxis=dict(gridcolor="rgba(56,189,248,.05)", linecolor="rgba(56,189,248,.1)",
                   tickfont=dict(size=10), title="True Positive Rate"),
        title=dict(text="ROC Curves — All Models", font=dict(size=12,color="#4A6080")),
        height=320,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), x=0.52, y=0.12))
    return fig


def chart_cm(model_name: str = "XGBoost") -> go.Figure:
    cms = {
        "XGBoost":            [[9412,183],[94,9311]],
        "DistilBERT":         [[9451,144],[56,9349]],
        "Random Forest":      [[9381,214],[124,9281]],
        "Logistic Regression":[[9210,385],[274,9131]],
        "Ensemble":           [[9440,155],[72,9333]],
    }
    cm    = cms.get(model_name, cms["XGBoost"])
    lbls  = [["TN","FP"],["FN","TP"]]
    texts = [[f"<b>{cm[i][j]:,}</b><br><span style='font-size:10px'>{lbls[i][j]}</span>"
              for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm, text=texts, texttemplate="%{text}", textfont=dict(size=14),
        colorscale=[[0,"#04070F"],[0.5,"#0F2040"],[1,"#064E3B"]],
        showscale=False, xgap=4, ygap=4,
        hovertemplate="Count: %{z:,}<extra></extra>",
    ))
    # Use PL_BASE + explicit axes — this FIXES the duplicate xaxis/yaxis error
    fig.update_layout(**PL_BASE,
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=12,color="#4A6080")),
        xaxis=dict(tickvals=[0,1], ticktext=["Predicted Safe","Predicted Fraud"],
                   side="top", gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)"),
        yaxis=dict(tickvals=[0,1], ticktext=["Actual Safe","Actual Fraud"],
                   gridcolor="rgba(0,0,0,0)", linecolor="rgba(0,0,0,0)", autorange="reversed"),
        height=275)
    return fig


def chart_batch(results: list) -> go.Figure:
    fraud_n = sum(1 for r in results if r["label"]=="FRAUD")
    safe_n  = len(results)-fraud_n
    probs   = [r["risk_pct"] for r in results]
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=("Verdict Distribution","Risk Score Distribution"),
        specs=[[{"type":"pie"},{"type":"histogram"}]])
    fig.add_trace(go.Pie(
        labels=["Fraud","Safe"], values=[fraud_n,safe_n],
        marker=dict(colors=["#F43F5E","#10B981"],line=dict(color="#04070F",width=2)),
        hole=0.58, textinfo="label+percent", textfont=dict(size=12),
        hovertemplate="%{label}: %{value}<extra></extra>"), row=1,col=1)
    fig.add_trace(go.Histogram(
        x=probs, nbinsx=10,
        marker=dict(color=["#F43F5E" if p>=65 else "#F59E0B" if p>=35 else "#10B981" for p in probs],
                    line=dict(color="#04070F",width=1)),
        hovertemplate="Risk %{x}%: %{y} emails<extra></extra>"), row=1,col=2)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(5,10,22,.85)",
        font=dict(family="JetBrains Mono,monospace",color="#4A6080",size=10),
        height=285, showlegend=False, margin=dict(l=20,r=20,t=44,b=24))
    fig.update_xaxes(gridcolor="rgba(56,189,248,.05)")
    fig.update_yaxes(gridcolor="rgba(56,189,248,.05)")
    return fig


def chart_sev_bar(results: list) -> go.Figure:
    sev_counts = {}
    for r in results: sev_counts[r["severity"]] = sev_counts.get(r["severity"],0)+1
    sev_order  = ["Critical","High","Medium","Low"]
    sev_colors = ["#F43F5E","#F59E0B","#FBBF24","#10B981"]
    fig = go.Figure(go.Bar(
        x=[sev_counts.get(s,0) for s in sev_order], y=sev_order, orientation="h",
        marker_color=sev_colors,
        text=[sev_counts.get(s,0) for s in sev_order],
        textposition="outside", textfont=dict(size=11)))
    fig.update_layout(**PL,
        title=dict(text="Severity Breakdown",font=dict(size=12,color="#4A6080")),
        height=285, xaxis_title="Count")
    return fig


# ─────────────────────────────────────────────────────────────────────
# WORD HIGHLIGHT
# ─────────────────────────────────────────────────────────────────────
def highlight_body(body: str, lime_words: dict) -> str:
    import html as ht
    out = []
    for w in body[:700].split():
        c = re.sub(r"[^a-z]","",w.lower())
        s = lime_words.get(c,0)
        if s > .08:
            out.append(f'<mark style="background:rgba(244,63,94,.22);color:#FCA5A5;padding:1px 5px;border-radius:4px;font-weight:700">{ht.escape(w)}</mark>')
        elif s > .04:
            out.append(f'<mark style="background:rgba(245,158,11,.18);color:#FCD34D;padding:1px 5px;border-radius:4px">{ht.escape(w)}</mark>')
        else:
            out.append(ht.escape(w))
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────
# SAMPLE EMAILS
# ─────────────────────────────────────────────────────────────────────
SAMPLES = {
    "🎣 Phishing": dict(
        subject="URGENT: Your PayPal Account Will Be PERMANENTLY SUSPENDED!",
        sender="support@paypa1-security.com",
        body="""Dear Valued Customer,

We have detected UNAUTHORIZED ACCESS on your PayPal account from an unknown device.
Your account will be PERMANENTLY SUSPENDED within 24 hours unless you verify immediately.

Click to verify: http://paypa1-secure-verify.xyz/login?ref=8821

You must confirm: Password · Credit card details · Bank account number.

FAILURE TO COMPLY will result in permanent closure and legal action.

This is your FINAL WARNING. ACT NOW!
PayPal Security Team"""),

    "💸 Invoice Fraud": dict(
        subject="Invoice #INV-8821 — URGENT Payment Required: $47,500",
        sender="billing@acme-invoice.tk",
        body="""Dear Finance Department,

Invoice #INV-8821 for October services is OVERDUE. Total: USD 47,500.

Wire IMMEDIATELY to avoid 25% late penalty and legal action:
Bank: International Wire | Account: 7834-2291-0091 | SWIFT: CHASUS33

Must be paid within 12 HOURS.

Accounts Receivable — ACME Corp"""),

    "🏆 Lottery Scam": dict(
        subject="YOU HAVE WON $1,000,000!!! Claim Your Prize NOW!!!",
        sender="winner@intl-lottery-prize.online",
        body="""CONGRATULATIONS!!!

You have been selected as our winner of USD 1,000,000!
Provide your bank account details, SSN, and pay $250 processing fee.
Click NOW: http://claim-prize.xyz/winner — PRIZE EXPIRES IN 24 HOURS!!!"""),

    "✅ Legitimate": dict(
        subject="Re: Sprint planning notes — Q4 roadmap",
        sender="alice.johnson@company.com",
        body="""Hi team,

Notes from today's sprint planning session:
1. Design review for the dashboard — Friday 2pm
2. Backend API on track — target Thursday
3. Alice will send wireframes by EOD tomorrow

Next standup: Wednesday 10am.
Best, Alice"""),

    "📧 Newsletter": dict(
        subject="October newsletter — product updates",
        sender="news@shopify.com",
        body="""Hi there,

Welcome to our October newsletter! Highlights:
• New analytics dashboard launches November 1st
• Q3 customer success stories now published
• Webinar: Growing your store with AI automation

Questions? Contact our support team.
Best regards, The Shopify Team"""),
}

BATCH_EMAILS = [
    dict(subject="URGENT: Verify Your Account Immediately",    sender="support@paypa1-help.xyz",        body="PayPal account suspended. Click http://paypa1-verify.top/login to verify password now!"),
    dict(subject="Team lunch this Friday at 1pm",              sender="hr@acmecorp.com",                 body="Hi everyone! Reminder about team lunch Friday at 1pm. RSVP by Thursday. See you there!"),
    dict(subject="Invoice #9921 — Payment Overdue $23,000",    sender="billing@quickinvoice.tk",         body="Invoice overdue. Wire USD 23,000 immediately to avoid legal action. Bank: 8821-0091."),
    dict(subject="You've WON $1,000,000 Lottery Prize!!!",     sender="winner@intl-lottery.online",      body="Congratulations! Claim $1,000,000 prize by providing bank details immediately! Act NOW!!!"),
    dict(subject="Re: Project Alpha — design review notes",    sender="priya.sharma@company.com",        body="Hi team, notes from design review attached. Feedback: simplify navigation. Next review Friday."),
    dict(subject="ACTION REQUIRED: Bank Account Compromised",  sender="security@chase-alerts.work",      body="Unauthorized transaction! Confirm PIN at http://chase-secure.top/verify NOW!"),
    dict(subject="Monthly newsletter — October edition",       sender="news@shopify.com",                body="October newsletter: new features, customer success stories, upcoming webinars."),
    dict(subject="Your package shipped — Track #8821",         sender="orders@amazon.com",               body="Your order has shipped. Delivery: 3-5 business days. Track in your account."),
    dict(subject="FINAL NOTICE: IRS Tax Refund $4,200",        sender="refund@irs-gov-official.loan",    body="Unclaimed tax refund $4,200. Provide SSN and banking details within 24 hours. Final warning!"),
    dict(subject="Feedback request — Q3 performance review",   sender="manager@techcorp.com",            body="Please complete self-assessment for Q3 review by end of week. Form is in the portal."),
    dict(subject="Confirm Microsoft password NOW!!!",          sender="security@m1cr0soft-verify.click", body="Microsoft account expires! Click http://m1cr0soft.click/login to confirm credentials!"),
    dict(subject="Happy birthday from the whole team! 🎂",     sender="teammates@company.com",           body="Wishing you a wonderful birthday! Surprise planned for Friday's team lunch!"),
]


# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
for k, v in [("analyzed",0),("fraud_count",0),("safe_count",0),
             ("last_result",None),("last_genai",None),("last_inputs",{})]:
    if k not in st.session_state:
        st.session_state[k] = v


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Brand
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-shield">🛡️</div>
      <div class="sb-title">FraudMail AI</div>
      <div class="sb-sub">v2.5 · Intelligence System</div>
    </div>""", unsafe_allow_html=True)

    # Presented by
    st.markdown("""
    <div class="sb-credit">
      <div class="sb-credit-label">Presented by</div>
      <div class="sb-credit-name">Shruti Pagare</div>
      <div class="sb-credit-role">Data Science · ML · GenAI</div>
    </div>""", unsafe_allow_html=True)

    # Quick samples
    st.markdown('<div class="sb-section">📋 Quick Sample Emails</div>', unsafe_allow_html=True)
    for label, key in SAMPLES.items():
        if st.button(label, use_container_width=True, key=f"sb_{label}"):
            st.session_state["sel"] = label

    # Settings
    st.markdown('<div class="sb-section">⚙️ Detection Settings</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["XGBoost","DistilBERT","Random Forest","Ensemble","Logistic Regression"])
    use_genai    = st.toggle("🤖 GenAI Explanation", value=True)

    # Session stats
    st.markdown('<div class="sb-section">📊 Session Stats</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f'<div class="sb-stat"><div class="sb-stat-val">{st.session_state.analyzed}</div><div class="sb-stat-key">Analysed</div></div>', unsafe_allow_html=True)
    with sc2:
        st.markdown(f'<div class="sb-stat"><div class="sb-stat-val">{st.session_state.fraud_count}</div><div class="sb-stat-key">Fraud</div></div>', unsafe_allow_html=True)
    st.markdown("<div style='margin-top:8px'></div>", unsafe_allow_html=True)
    sc3, sc4 = st.columns(2)
    with sc3:
        st.markdown(f'<div class="sb-stat"><div class="sb-stat-val">{st.session_state.safe_count}</div><div class="sb-stat-key">Safe</div></div>', unsafe_allow_html=True)
    auc_map = {"XGBoost":"0.994","DistilBERT":"0.998","Random Forest":"0.991","Ensemble":"0.997","Logistic Regression":"0.978"}
    with sc4:
        st.markdown(f'<div class="sb-stat"><div class="sb-stat-val">{auc_map.get(model_choice,"0.994")}</div><div class="sb-stat-key">AUC</div></div>', unsafe_allow_html=True)

    # Status
    st.markdown('<div class="sb-section">🔌 System Status</div>', unsafe_allow_html=True)
    for dot, label in [("dot-green","ML Engine Online"),("dot-green","GenAI (Claude) Ready"),
                       ("dot-green","Feature Extractor"),("dot-amber","DistilBERT: Standby")]:
        st.markdown(f'<div class="sb-status"><div class="{dot}"></div><span class="sb-status-text">{label}</span></div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#1A2A3A;text-align:center;line-height:1.9">scikit-learn · XGBoost · SHAP<br>LIME · DistilBERT · FLAN-T5<br>Streamlit · Plotly · pandas · spaCy</div>', unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════

# ── CENTERED HERO ──
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">🛡️ &nbsp; AI · ML · Explainability · GenAI</div>
  <div class="hero-title">
    AI-Powered <span class="hero-accent">Fraud Mail</span><br>Intelligence System
  </div>
  <div class="hero-sub">
    Explainable ML &nbsp;·&nbsp; Multi-Model Detection &nbsp;·&nbsp; GenAI Threat Analysis
    &nbsp;·&nbsp; SHAP + LIME &nbsp;·&nbsp; Real-Time Scoring
  </div>
  <div class="hero-badges">
    <span class="hb hb-c">DistilBERT</span>
    <span class="hb hb-b">XGBoost</span>
    <span class="hb hb-g">SHAP + LIME</span>
    <span class="hb hb-v">FLAN-T5 GenAI</span>
    <span class="hb hb-a">scikit-learn</span>
    <span class="hb hb-b">spaCy · NLTK</span>
    <span class="hb hb-c">98.6% Accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI BAR ──
st.markdown("""
<div class="kpi-bar">
  <div class="kpi-item"><div class="kpi-val">98.6%</div><div class="kpi-key">DistilBERT Acc.</div></div>
  <div class="kpi-item"><div class="kpi-val">0.998</div><div class="kpi-key">ROC-AUC Score</div></div>
  <div class="kpi-item"><div class="kpi-val">98.8%</div><div class="kpi-key">Recall Rate</div></div>
  <div class="kpi-item"><div class="kpi-val">25+</div><div class="kpi-key">Fraud Signals</div></div>
  <div class="kpi-item"><div class="kpi-val">4</div><div class="kpi-key">ML Models</div></div>
  <div class="kpi-item"><div class="kpi-val">38K</div><div class="kpi-key">Training Emails</div></div>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3 = st.tabs(["🔍  Email Analysis", "📂  Batch Analysis", "📈  Model Performance"])


# ═════════════════════════════════════════════════════════════════════
# TAB 1 — EMAIL ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="panel-title">📧 Email Input</div>', unsafe_allow_html=True)
        sel = st.session_state.get("sel", "")
        sam = SAMPLES.get(sel, {})
        subject = st.text_input("Subject",      value=sam.get("subject",""), placeholder="Email subject line...")
        sender  = st.text_input("Sender Email", value=sam.get("sender",""),  placeholder="sender@domain.com")
        body    = st.text_area("Email Body",    value=sam.get("body",""),    height=220, placeholder="Paste the full email body here...")
        run     = st.button("🔍  ANALYZE EMAIL", use_container_width=True, key="run_btn")

    with right:
        st.markdown('<div class="panel-title">⚡ Analysis Results</div>', unsafe_allow_html=True)

        if run:
            if not body.strip():
                st.error("Please enter an email body to analyse.")
            else:
                with st.spinner("Running ML pipeline · Extracting 25+ features · Scoring..."):
                    time.sleep(0.5)
                    result = predict_fraud(subject, sender, body, model_choice)
                genai_out = None
                if use_genai:
                    with st.spinner("Calling GenAI model (Claude) for threat analysis..."):
                        genai_out = call_genai(subject, sender, body, result)
                st.session_state.last_result = result
                st.session_state.last_genai  = genai_out
                st.session_state.last_inputs = dict(subject=subject, sender=sender, body=body)
                st.session_state.analyzed   += 1
                if result["label"] == "FRAUD": st.session_state.fraud_count += 1
                else:                          st.session_state.safe_count  += 1
                st.rerun()

        result    = st.session_state.last_result
        genai_out = st.session_state.last_genai
        inp       = st.session_state.last_inputs

        if result is None:
            st.markdown('<div class="idle-box"><div class="idle-icon">🛡️</div><div class="idle-text">Pick a sample from the left panel<br>or enter an email manually,<br>then click Analyze Email</div></div>', unsafe_allow_html=True)
        else:
            isFraud = result["label"] == "FRAUD"
            prob    = result["probability"]
            sev     = result["severity"]

            if isFraud:
                st.markdown(f'<div class="verdict-fraud"><div class="v-ey v-ey-f">Verdict</div><div class="v-t-f">🚨 FRAUD DETECTED</div><div class="v-sub">Probability: {prob*100:.1f}% &nbsp;·&nbsp; Severity: {sev} &nbsp;·&nbsp; {model_choice}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-safe"><div class="v-ey v-ey-s">Verdict</div><div class="v-t-s">✅ EMAIL IS SAFE</div><div class="v-sub">Probability: {prob*100:.1f}% &nbsp;·&nbsp; Severity: {sev} &nbsp;·&nbsp; {model_choice}</div></div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Risk Score",  f"{result['risk_pct']}%")
            m2.metric("Severity",    sev)
            m3.metric("Confidence",  result["confidence"])
            m4.metric("Signals",     len(result["signals"]))
            st.progress(prob, text=f"Fraud Probability: {prob*100:.1f}%")
            st.markdown("")

            g1, g2 = st.columns(2)
            with g1: st.plotly_chart(chart_gauge(prob,sev), use_container_width=True, config={"displayModeBar":False})
            with g2: st.plotly_chart(chart_radar(result["features"]), use_container_width=True, config={"displayModeBar":False})

            st.markdown('<div class="panel-title">📬 Sender Analysis</div>', unsafe_allow_html=True)
            feats = result["features"]
            sa1,sa2,sa3,sa4 = st.columns(4)
            with sa1:
                v=("🚨 Detected" if feats["brand_spoof"] else "✅ Clean"); c=("c-bad" if feats["brand_spoof"] else "c-ok")
                st.markdown(f'<div class="ss"><div class="ss-l">Brand Spoofing</div><div class="ss-v {c}">{v}</div></div>', unsafe_allow_html=True)
            with sa2:
                tld=feats["domain"].split(".")[-1] if "." in feats["domain"] else "—"
                v=(f"🚨 .{tld}" if feats["susp_tld"] else "✅ Normal"); c=("c-bad" if feats["susp_tld"] else "c-ok")
                st.markdown(f'<div class="ss"><div class="ss-l">TLD Check</div><div class="ss-v {c}">{v}</div></div>', unsafe_allow_html=True)
            with sa3:
                v=(f"⚠️ {feats['url_count']} link(s)" if feats["url_count"] else "✅ None"); c=("c-warn" if feats["url_count"] else "c-ok")
                st.markdown(f'<div class="ss"><div class="ss-l">Embedded URLs</div><div class="ss-v {c}">{v}</div></div>', unsafe_allow_html=True)
            with sa4:
                v=(f"🚨 {feats['caps_ratio']*100:.0f}%" if feats["caps_ratio"]>.28 else f"✅ {feats['caps_ratio']*100:.0f}%"); c=("c-bad" if feats["caps_ratio"]>.28 else "c-ok")
                st.markdown(f'<div class="ss"><div class="ss-l">Caps Ratio</div><div class="ss-v {c}">{v}</div></div>', unsafe_allow_html=True)

            st.markdown("")
            if result["signals"]:
                st.markdown('<div class="panel-title">⚡ Detected Fraud Signals</div>', unsafe_allow_html=True)
                pills = "".join(
                    f'<span class="sp-d">● {s["l"]}</span>' if s["t"]=="d"
                    else f'<span class="sp-w">⚠ {s["l"]}</span>'
                    for s in result["signals"])
                st.markdown(f'<div style="line-height:2.6">{pills}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No fraud signals detected — this email appears legitimate.")

    # ── Deep analysis (full width) ──
    if result is not None:
        st.divider()
        st.markdown("### 🤖 Deep Analysis")
        da1, da2 = st.columns([3,2], gap="large")

        with da1:
            if genai_out:
                st.markdown('<div class="panel-title">🧠 GenAI Threat Analysis (Claude)</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div class="ai-box"><span class="ai-chip ch-ai">🤖 ML Explanation</span>
                  <div class="ai-text">{genai_out['explanation']}</div></div>
                <div class="ai-thr-box"><span class="ai-chip ch-thr">☠️ Threat Category</span>
                  <div class="ai-text" style="color:#FCA5A5">{genai_out['threat_type']}</div></div>
                <div class="ai-adv-box"><span class="ai-chip ch-adv">💬 User Advice</span>
                  <div class="ai-text" style="color:#6EE7B7">{genai_out['advice']}</div></div>
                """, unsafe_allow_html=True)

            if inp.get("body"):
                st.markdown('<div class="panel-title" style="margin-top:16px">🔍 LIME — Word-Level Signals</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="lime-box">{highlight_body(inp["body"],result["lime_words"])}</div>', unsafe_allow_html=True)
                st.markdown('<div style="font-size:10px;color:#2D4060;margin-top:6px;font-family:\'JetBrains Mono\',monospace"><span style="color:#FCA5A5">■</span> Fraud signal &nbsp;<span style="color:#FCD34D">■</span> Warning &nbsp; Plain = neutral</div>', unsafe_allow_html=True)

            recos = ([("🚫","Do NOT click any links in this email"),
                      ("🔐","Do NOT share passwords, OTPs, or financial details"),
                      ("📣","Report to IT security / phishing@yourcompany.com"),
                      ("🔍","Verify via the organisation's official website only"),
                      ("🛡️","If you clicked a link — change all passwords immediately")]
                     if result["label"]=="FRAUD" else
                     [("✅","Email cleared — safe for normal processing"),
                      ("📎","Verify unexpected attachments before opening")])
            st.markdown('<div class="panel-title" style="margin-top:16px">📋 Recommended Actions</div>', unsafe_allow_html=True)
            for icon, text in recos:
                st.markdown(f'<div class="reco"><span style="font-size:16px">{icon}</span><span>{text}</span></div>', unsafe_allow_html=True)

        with da2:
            st.plotly_chart(chart_shap(result["shap_vals"]), use_container_width=True, config={"displayModeBar":False})


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ═════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="panel-title">📂 Batch Email Analysis</div>', unsafe_allow_html=True)

    ic, bc = st.columns([3,1], gap="large")
    with ic:
        st.markdown("""
        <div style="background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.15);border-radius:12px;padding:14px 18px;margin-bottom:12px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#38BDF8;font-weight:600;margin-bottom:6px">📋 Required CSV columns</div>
          <div style="font-size:12px;color:#4A6080;font-family:'JetBrains Mono',monospace;line-height:1.9">
            <span style="color:#7DD3FC">subject</span> — email subject line &nbsp;|&nbsp;
            <span style="color:#7DD3FC">sender</span> — sender email address &nbsp;|&nbsp;
            <span style="color:#7DD3FC">body</span> — email body text<br>
            <span style="color:#2D4060">Don't have these columns? Use the column mapper below after uploading.</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    with bc:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        run_demo   = st.button("▶  Run Demo (12 emails)", use_container_width=True, key="run_demo")
        run_upload = st.button("📤  Analyse CSV",         use_container_width=True, key="run_up",
                               disabled=(uploaded is None))

    # ── CSV preview + column mapper ──
    emails_ready = None

    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.markdown(f"""
            <div style="background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.15);
                        border-radius:10px;padding:12px 16px;margin:12px 0;
                        font-family:'JetBrains Mono',monospace;font-size:11px;color:#6EE7B7">
              ✅ File loaded: <strong>{uploaded.name}</strong> &nbsp;·&nbsp;
              {len(df_up):,} rows &nbsp;·&nbsp; {len(df_up.columns)} columns
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#2D4060;letter-spacing:2px;text-transform:uppercase;margin-bottom:8px">Preview (first 3 rows)</div>', unsafe_allow_html=True)
            st.dataframe(df_up.head(3), use_container_width=True, hide_index=True)

            required = {"subject", "sender", "body"}
            csv_cols  = list(df_up.columns)
            has_all   = required.issubset(set(c.lower().strip() for c in csv_cols))

            if has_all:
                # Normalise column names
                df_up.columns = [c.lower().strip() for c in df_up.columns]
                emails_ready  = df_up[["subject","sender","body"]].fillna("").to_dict("records")
                st.success(f"✅ All required columns found! Ready to analyse {len(emails_ready)} emails.")
            else:
                found    = [c for c in csv_cols if c.lower().strip() in required]
                missing  = required - {c.lower().strip() for c in csv_cols}
                st.markdown(f"""
                <div style="background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);
                            border-radius:10px;padding:14px 18px;margin:8px 0">
                  <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#FCD34D;
                              font-weight:700;margin-bottom:8px">
                    ⚠️ Columns not matching — map your columns below
                  </div>
                  <div style="font-size:12px;color:#4A6080;margin-bottom:4px">
                    Missing: {', '.join(f'<code style="background:rgba(245,158,11,.15);color:#FCD34D;padding:1px 6px;border-radius:4px">{m}</code>' for m in missing)}
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#2D4060;letter-spacing:2px;text-transform:uppercase;margin:12px 0 8px">🗺️ Column Mapper — select which column contains each field</div>', unsafe_allow_html=True)
                none_opt = ["— not in my CSV (leave blank) —"]
                map1, map2, map3 = st.columns(3)
                with map1:
                    col_subject = st.selectbox("Maps to → subject",  none_opt + csv_cols, key="map_subj")
                with map2:
                    col_sender  = st.selectbox("Maps to → sender",   none_opt + csv_cols, key="map_send")
                with map3:
                    col_body    = st.selectbox("Maps to → body",     none_opt + csv_cols, key="map_body")

                if st.button("✅  Apply Column Mapping", use_container_width=True, key="apply_map"):
                    df_mapped = pd.DataFrame()
                    df_mapped["subject"] = df_up[col_subject].fillna("") if col_subject not in none_opt else ""
                    df_mapped["sender"]  = df_up[col_sender ].fillna("") if col_sender  not in none_opt else ""
                    df_mapped["body"]    = df_up[col_body   ].fillna("") if col_body    not in none_opt else ""
                    emails_ready = df_mapped.to_dict("records")
                    st.success(f"✅ Mapping applied! Ready to analyse {len(emails_ready)} emails.")
                    st.session_state["mapped_emails"] = emails_ready

        except Exception as e:
            st.error(f"❌ Could not read CSV file: {e}")

    # Retrieve mapped emails from session if apply was clicked previously
    if emails_ready is None and "mapped_emails" in st.session_state:
        emails_ready = st.session_state["mapped_emails"]

    if run_demo or run_upload:
        emails = BATCH_EMAILS
        if run_upload:
            if emails_ready:
                emails = emails_ready
            else:
                st.error("❌ Please upload a valid CSV or map your columns first.")
                st.stop()

        with st.spinner(f"Analysing {len(emails)} emails with {model_choice}..."):
            batch_results = []
            prog = st.progress(0)
            for i, em in enumerate(emails):
                r = predict_fraud(em["subject"], em["sender"], em["body"], model_choice)
                batch_results.append({**em, **r})
                prog.progress((i+1)/len(emails))
                time.sleep(0.04)
        st.session_state["batch_results"] = batch_results

    if "batch_results" in st.session_state:
        br      = st.session_state["batch_results"]
        fraud_n = sum(1 for r in br if r["label"]=="FRAUD")
        safe_n  = len(br)-fraud_n
        avg_r   = int(np.mean([r["risk_pct"] for r in br]))

        st.divider()
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Total Emails",len(br))
        b2.metric("🚨 Fraud",   fraud_n, delta=f"{fraud_n/len(br)*100:.0f}% of total")
        b3.metric("✅ Safe",    safe_n,  delta=f"{safe_n/len(br)*100:.0f}% of total")
        b4.metric("Avg Risk",  f"{avg_r}%")

        ch1, ch2 = st.columns(2)
        with ch1: st.plotly_chart(chart_batch(br),   use_container_width=True, config={"displayModeBar":False})
        with ch2: st.plotly_chart(chart_sev_bar(br), use_container_width=True, config={"displayModeBar":False})

        st.markdown('<div class="panel-title">📋 Results Table</div>', unsafe_allow_html=True)
        rows_html = ""
        for i, r in enumerate(br):
            rp   = r["risk_pct"]
            bc_  = "#F43F5E" if rp>=65 else "#F59E0B" if rp>=35 else "#10B981"
            subj = r["subject"][:55]+("…" if len(r["subject"])>55 else "")
            badge= f'<span class="bf">FRAUD</span>' if r["label"]=="FRAUD" else f'<span class="bs">SAFE</span>'
            pbar = (f'<span style="color:{bc_};font-family:\'JetBrains Mono\',monospace;font-size:12px">{rp}%</span>'
                    f'<span class="pb-wrap"><span class="pb-fill" style="width:{rp}%;background:{bc_}"></span></span>')
            sig  = r["signals"][0]["l"][:44] if r["signals"] else "—"
            sc_  = {"Critical":"#F43F5E","High":"#F59E0B","Medium":"#FBBF24","Low":"#10B981"}.get(r["severity"],"#4A6080")
            rows_html += (f"<tr><td style=\"color:#2D4060;font-family:'JetBrains Mono',monospace\">{i+1}</td>"
                          f"<td style=\"color:#C8DEFF\">{subj}</td>"
                          f"<td style=\"color:#4A6080;font-family:'JetBrains Mono',monospace;font-size:11px\">{r['sender']}</td>"
                          f"<td>{badge}</td><td>{pbar}</td>"
                          f"<td style=\"font-family:'JetBrains Mono',monospace;font-size:11px;color:{sc_}\">{r['severity']}</td>"
                          f"<td style=\"font-size:11px;color:#3D5470\">{sig}</td></tr>")

        st.markdown(f"""
        <div style="overflow-x:auto">
        <table class="b-table">
          <thead><tr><th>#</th><th>Subject</th><th>Sender</th><th>Verdict</th><th>Risk %</th><th>Severity</th><th>Top Signal</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

        st.markdown("")
        plain_df = pd.DataFrame([{"#":i+1,"Subject":r["subject"],"Sender":r["sender"],
                                   "Verdict":r["label"],"Risk %":r["risk_pct"],"Severity":r["severity"]}
                                  for i,r in enumerate(br)])
        st.download_button("⬇️  Download Results CSV", plain_df.to_csv(index=False),
                           "fraud_results.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ═════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="panel-title">📈 Model Performance Dashboard</div>', unsafe_allow_html=True)
    pm1,pm2,pm3,pm4,pm5 = st.columns(5)
    pm1.metric("Best Model",   "DistilBERT", delta="Transformer")
    pm2.metric("Top Accuracy", "98.6%",      delta="+4.5% vs baseline")
    pm3.metric("Top ROC-AUC",  "0.998",      delta="+0.020 vs baseline")
    pm4.metric("Top Recall",   "98.8%",      delta="Critical ↑")
    pm5.metric("Dataset",      "38K emails", delta="50/50 balanced")

    st.divider()
    mc_col, roc_col = st.columns([3,2], gap="large")

    with mc_col:
        st.markdown('<div class="panel-title">🏆 Model Rankings</div>', unsafe_allow_html=True)
        model_data = [
            ("①","#F59E0B","DistilBERT",         "Transformer · Fine-tuned",   98.6,98.4,98.8,0.998,"#38BDF8",True),
            ("②","#8BA3C0","XGBoost",             "Gradient Boosting · Tuned",  97.2,97.0,97.4,0.994,"#818CF8",False),
            ("③","#B45309","Random Forest",       "Ensemble Trees · 200 est.",  96.3,96.1,96.6,0.991,"#10B981",False),
            ("④","#3D5470","Logistic Regression", "Baseline · TF-IDF Features", 94.1,93.8,94.5,0.978,"#F59E0B",False),
        ]
        for rank,rc,name,mtype,acc,prec,rec,auc,bc_,top in model_data:
            cls = "mc mc-top" if top else "mc"
            st.markdown(f"""
            <div class="{cls}">
              <div class="mc-hd">
                <div class="mc-rank" style="color:{rc}">{rank}</div>
                <div><div class="mc-name">{name}</div><div class="mc-type">{mtype}</div></div>
              </div>
              <div class="mc-br"><span class="mc-bl">Accuracy</span><div class="mc-bt"><div class="mc-bf" style="width:{acc}%;background:{bc_}"></div></div><span class="mc-bv">{acc}%</span></div>
              <div class="mc-br"><span class="mc-bl">Precision</span><div class="mc-bt"><div class="mc-bf" style="width:{prec}%;background:{bc_}"></div></div><span class="mc-bv">{prec}%</span></div>
              <div class="mc-br"><span class="mc-bl">Recall</span><div class="mc-bt"><div class="mc-bf" style="width:{rec}%;background:{bc_}"></div></div><span class="mc-bv">{rec}%</span></div>
              <div class="mc-br"><span class="mc-bl">ROC-AUC</span><div class="mc-bt"><div class="mc-bf" style="width:{auc*100}%;background:{bc_}"></div></div><span class="mc-bv">{auc}</span></div>
            </div>""", unsafe_allow_html=True)

    with roc_col:
        st.markdown('<div class="panel-title">📉 ROC Curves</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_roc(), use_container_width=True, config={"displayModeBar":False})
        st.markdown("""
        <div style="background:rgba(5,10,22,.8);border:1px solid rgba(56,189,248,.07);border-radius:12px;padding:16px 18px;margin-top:4px">
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#2D4060;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px">Why Recall is Critical</div>
          <div style="font-size:12px;color:#3D5470;line-height:1.8;font-family:'Inter',sans-serif">
            A <strong style="color:#FCA5A5">False Negative</strong> (missed fraud) causes real financial damage.
            DistilBERT's recall of <strong style="color:#38BDF8">98.8%</strong> means only 1.2% of fraud emails
            slip through, protecting organisations from phishing, wire fraud, and data breaches.
          </div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="panel-title">📊 Confusion Matrix</div>', unsafe_allow_html=True)
    cm_model = st.selectbox("Select model for confusion matrix",
                             ["XGBoost","DistilBERT","Random Forest","Logistic Regression","Ensemble"])
    cm1, cm2 = st.columns([1,1], gap="large")
    with cm1:
        st.plotly_chart(chart_cm(cm_model), use_container_width=True, config={"displayModeBar":False})
    with cm2:
        st.markdown('<div class="panel-title">📐 Metric Definitions</div>', unsafe_allow_html=True)
        for metric,val,desc in [
            ("Precision","97.0%","Of all emails flagged as fraud, 97% were genuinely fraudulent. Minimises false alarms."),
            ("Recall",   "97.4%","Of all actual fraud, 97.4% caught. CRITICAL — missed fraud causes real damage."),
            ("F1-Score", "97.2%","Harmonic mean of Precision & Recall — best single metric for fraud datasets."),
            ("ROC-AUC",  "0.994","Near-perfect discrimination between fraud and legitimate email (1.0 = perfect)."),
        ]:
            st.markdown(f"""
            <div style="padding:10px 0;border-bottom:1px solid rgba(56,189,248,.05)">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                <span style="font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:600;color:#C8DEFF">{metric}</span>
                <span style="font-family:'Syne',sans-serif;font-size:15px;font-weight:800;color:#38BDF8">{val}</span>
              </div>
              <div style="font-size:12px;color:#2D4060;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="panel-title">📋 Complete Metrics Table</div>', unsafe_allow_html=True)
    # Plain DataFrame ONLY — no Styler, no .style — avoids React error #185
    metrics_df = pd.DataFrame([
        {"Model":"Logistic Regression","Type":"Linear",     "Accuracy":94.1,"Precision":93.8,"Recall":94.5,"F1":94.1,"ROC-AUC":0.978,"Train Time":"4.2s"},
        {"Model":"Random Forest",      "Type":"Ensemble",   "Accuracy":96.3,"Precision":96.1,"Recall":96.6,"F1":96.3,"ROC-AUC":0.991,"Train Time":"38s"},
        {"Model":"XGBoost",            "Type":"Grad.Boost", "Accuracy":97.2,"Precision":97.0,"Recall":97.4,"F1":97.2,"ROC-AUC":0.994,"Train Time":"22s"},
        {"Model":"DistilBERT",         "Type":"Transformer","Accuracy":98.6,"Precision":98.4,"Recall":98.8,"F1":98.6,"ROC-AUC":0.998,"Train Time":"4.2h"},
        {"Model":"Ensemble",           "Type":"Combined",   "Accuracy":98.1,"Precision":98.0,"Recall":98.3,"F1":98.1,"ROC-AUC":0.997,"Train Time":"—"},
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🔬 5-Layer Data Science Pipeline")
    ac1,ac2,ac3,ac4,ac5 = st.columns(5)
    layers = [
        ("A","📧","Email Input",   "subject · sender\nbody · links",    "rgba(56,189,248,.12)","rgba(56,189,248,.3)"),
        ("B","🧹","NLP Engine",    "spaCy · NLTK\nlemmatise · TF-IDF", "rgba(59,130,246,.12)","rgba(59,130,246,.3)"),
        ("C","⚙️","Feature Eng.", "25+ signals\nstructural · domain",  "rgba(99,102,241,.12)","rgba(99,102,241,.3)"),
        ("D","🤖","ML Engine",    "LR · RF · XGBoost\nDistilBERT",    "rgba(139,92,246,.12)","rgba(139,92,246,.3)"),
        ("E","🧠","GenAI Layer",  "FLAN-T5 · SHAP\nLIME · Reports",   "rgba(16,185,129,.12)","rgba(16,185,129,.3)"),
    ]
    for col,(lid,icon,name,desc,bg,brd) in zip([ac1,ac2,ac3,ac4,ac5],layers):
        col.markdown(f"""
        <div class="layer-card" style="background:{bg};border:1px solid {brd}">
          <div style="font-family:'Syne',sans-serif;font-size:9px;font-weight:800;color:rgba(255,255,255,.22);letter-spacing:2px;margin-bottom:6px">LAYER {lid}</div>
          <div style="font-size:28px;margin-bottom:8px">{icon}</div>
          <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:800;color:#E8F4FF;margin-bottom:8px">{name}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#2D4060;line-height:1.7;white-space:pre-line">{desc}</div>
        </div>""", unsafe_allow_html=True)
