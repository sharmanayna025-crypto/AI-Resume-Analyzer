import math
from html import escape
from pathlib import Path

import spacy
import streamlit as st

from analyzer import Analysis, analyze, match_job, report_json, report_markdown

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")


# ------------------------------------------------------------------ setup
@st.cache_resource(show_spinner=False)
def load_nlp():
    return spacy.load("en_core_web_sm")


def load_css():
    css = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ------------------------------------------------------------------ html helpers
def grade(score: float) -> tuple[str, str, str]:
    if score >= 75:
        return "Strong", "good", "#15803D"
    if score >= 50:
        return "Good start", "warn", "#D97706"
    return "Needs work", "bad", "#B91C1C"


def ring(score: float, label: str | None = None) -> str:
    text, cls, color = grade(score)
    r = 62
    c = 2 * math.pi * r
    off = c * (1 - score / 100)
    return f"""
    <div class="ring-wrap">
      <div class="ring">
        <svg width="150" height="150" viewBox="0 0 150 150">
          <circle cx="75" cy="75" r="{r}" fill="none" stroke="#F0EADF" stroke-width="12"/>
          <circle cx="75" cy="75" r="{r}" fill="none" stroke="{color}" stroke-width="12"
                  stroke-linecap="round" stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}"/>
        </svg>
        <div class="num"><b>{score:.0f}</b><small>out of 100</small></div>
      </div>
      <span class="grade {cls}">{label or text}</span>
    </div>"""


def bar(label: str, got: float, mx: float, percent: bool = False, cls: str = "") -> str:
    pct = got / mx * 100 if mx else 0
    right = f"{pct:.0f}%" if percent else f"{got:.0f} / {mx:.0f}"
    return (f'<div class="bar-row"><div class="bar-label"><span>{escape(label)}</span><span>{right}</span></div>'
            f'<div class="bar {cls}"><div style="width:{pct:.0f}%"></div></div></div>')


def kv(label: str, value: str | None) -> str:
    val = f"<span>{escape(str(value))}</span>" if value not in (None, "") else '<span class="na">Not found</span>'
    return f'<div class="kv"><span>{label}</span>{val}</div>'


def chips(items: list[str], cls: str = "") -> str:
    if not items:
        return '<span class="chip mute">None 🎉</span>' if cls == "warn" else '<span class="chip mute">None</span>'
    return '<div class="chips">' + "".join(f'<span class="chip {cls}">{escape(i)}</span>' for i in items) + "</div>"


def card(title: str, body: str) -> str:
    head = f"<h4>{title}</h4>" if title else ""
    return f'<div class="card">{head}{body}</div>'


def grid(kind: str, *parts: str) -> str:
    return f'<div class="{kind}">' + "".join(parts) + "</div>"


def md(html: str):
    st.markdown(html, unsafe_allow_html=True)


TIP_ICONS = {"good": "✅", "warn": "⚠️", "bad": "⛔", "info": "💡"}


def tip(level: str, text: str) -> str:
    parts = escape(text).split("**")
    html = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
    return f'<div class="tip {level}"><span>{TIP_ICONS[level]}</span><div>{html}</div></div>'


def title(text: str):
    md(f'<div class="section-title">{text}</div>')


def radar(values: dict[str, float]) -> str:
    cx, cy, R = 180, 150, 108
    n = len(values)
    labels = list(values)

    def pt(i: int, r: float) -> tuple[float, float]:
        ang = -math.pi / 2 + 2 * math.pi * i / n
        return cx + r * math.cos(ang), cy + r * math.sin(ang)

    out = ['<svg viewBox="-50 0 460 300" width="460" role="img" aria-label="Skill category coverage">']
    for lvl in (0.25, 0.5, 0.75, 1):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (pt(i, R * lvl) for i in range(n)))
        out.append(f'<polygon points="{pts}" fill="none" stroke="#E8E1D6"/>')
    for i in range(n):
        x, y = pt(i, R)
        out.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#E8E1D6"/>')
    data = [pt(i, R * max(values[k], 6) / 100) for i, k in enumerate(labels)]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in data)
    out.append(f'<polygon points="{pts}" fill="rgba(15,118,110,.22)" stroke="#0F766E" stroke-width="2" stroke-linejoin="round"/>')
    for x, y in data:
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#fff" stroke="#D97706" stroke-width="2.5"/>')
    for i, k in enumerate(labels):
        x, y = pt(i, R + 22)
        anchor = "middle" if abs(x - cx) < 8 else ("end" if x < cx else "start")
        out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" font-size="13" font-weight="600" fill="#57534E" '
                   f'font-family="Inter,system-ui,sans-serif">{escape(k)}</text>')
    out.append("</svg>")
    return '<div class="radar">' + "".join(out) + "</div>"


# ------------------------------------------------------------------ tabs
def render_overview(a: Analysis):
    md(grid("grid-side",
            card("Overall score", ring(a.score)),
            card("Score breakdown", "".join(bar(k, v, m) for k, (v, m) in a.breakdown.items()))))
    md('<div class="gap"></div>')
    yrs = f"{a.years_exp:.0f}" if a.years_exp is not None else "—"
    stats = [
        (len(a.skill_list), "Skills detected"),
        (f"{sum(a.sections.values())}/{len(a.sections)}", "Sections found"),
        (yrs, "Years of experience"),
        (f"{a.ats_score:.0f}%", "ATS compatibility"),
    ]
    md(grid("grid4", *[f'<div class="stat"><b>{n}</b><span>{l}</span></div>' for n, l in stats]))
    title("Candidate details")
    md(grid("grid2",
            card("Contact", kv("Name", a.name) + kv("Email", a.email) + kv("Phone", a.phone)),
            card("Profiles & length", kv("LinkedIn", a.linkedin) + kv("GitHub", a.github)
                 + kv("Length", f"{a.page_count} page{'s' if a.page_count != 1 else ''} · {a.word_count} words"))))


def render_skills(a: Analysis):
    if a.skills:
        body = "".join(f'<div class="cat">{cat}</div>{chips(sk)}' for cat, sk in a.skills.items())
    else:
        body = '<span class="chip mute">No recognised skills found</span>'
    checks = "".join(
        f'<div class="check"><span class="dot {"ok" if ok else "off"}">{"✓" if ok else "–"}</span>{sec}</div>'
        for sec, ok in a.sections.items())
    right = card("Skill balance", radar(a.radar)) + '<div class="gap"></div>' + card("Resume sections", checks)
    md(grid("grid-wide", card(f"Detected skills ({len(a.skill_list)})", body), f"<div>{right}</div>"))
    if a.action_verbs:
        md('<div class="gap"></div>' + card("Strong action verbs detected", chips(a.action_verbs, "good")))


def render_ats_writing(a: Analysis):
    checks = "".join(
        f'<div class="check"><span class="dot {"ok" if ok else "no"}">{"✓" if ok else "✕"}</span>'
        f'<div>{escape(lbl)}{"" if ok else f"<small>{escape(hint)}</small>"}</div></div>'
        for lbl, ok, hint in a.ats_checks)
    md(grid("grid-side", card("ATS compatibility", ring(a.ats_score)), card("Parsing checklist", checks)))

    title("Writing quality")
    bullets = kv("Bullet points", a.bullets) + kv("Avg. words per bullet", f"{a.avg_bullet_words:.0f}" if a.bullets else "—") \
        + kv("Quantified results", a.metrics_count) + kv("First-person pronouns", a.pronouns)
    weak = chips([f"{p} ×{n}" for p, n in a.weak_phrases.items()], "warn")
    md(grid("grid2",
            card("Structure & style", bullets),
            card("Phrases to improve",
                 f'<div class="cat">Passive / weak phrasing</div>{weak}'
                 f'<div class="cat">Clichés</div>{chips(a.buzzwords, "warn")}')))


def render_roles(a: Analysis):
    md('<p class="lead">Roles your detected skills fit best, based on the core skills typically asked for in each.</p>')
    cards = []
    for i, (name, pct, have, missing) in enumerate(a.roles):
        best = '<span class="best">Best fit</span>' if i == 0 else ""
        cards.append(card("", f'<div class="role"><div class="top"><b>{name}{best}</b><span>{pct:.0f}%</span></div>'
                              f'{bar("Skill coverage", pct, 100, percent=True, cls="amber" if i else "")}'
                              f'<div class="cat">You have</div>{chips(have, "good")}'
                              f'<div class="cat">Worth adding</div>{chips(missing[:5], "mute")}</div>'))
    md(grid("grid2", *cards))


def render_match(a: Analysis):
    md('<p class="lead">Paste a job description to see how well your resume fits and what to add.</p>')
    jd = st.text_area("Job description", height=190, label_visibility="collapsed",
                      placeholder="Paste the full job description here…")
    if not jd.strip():
        return None

    m = match_job(a.text, a.skill_list, jd)
    label = {"Strong": "Strong match", "Good start": "Partial match", "Needs work": "Weak match"}[grade(m.score)[0]]
    total = len(m.matched_skills) + len(m.missing_skills)
    md('<div class="gap"></div>' + grid("grid-side",
        card("Match score", ring(m.score, label)),
        card("Match details", bar("Skill coverage", len(m.matched_skills), max(total, 1), percent=True)
             + bar("Keyword overlap", m.keyword_overlap, 100, percent=True))))
    md('<div class="gap"></div>' + grid("grid2",
        card("Matched skills", chips(m.matched_skills, "good")),
        card("Missing skills", chips(m.missing_skills, "bad"))))
    if m.top_keywords_missing:
        md('<div class="gap"></div>' + card("Keywords from the job post not in your resume",
                                            chips(m.top_keywords_missing, "mute")))
    if m.advice:
        title("How to tailor your resume")
        md("".join(tip("info", t) for t in m.advice))
    return m


def render_tips(a: Analysis):
    md("".join(tip(lvl, txt) for lvl, txt in a.suggestions))


# ------------------------------------------------------------------ page
def landing():
    md('<p style="text-align:center;color:#A8A29E;font-size:.88rem;margin-top:.5rem">'
       'Drag & drop a PDF above · max 10&nbsp;MB · text-based PDFs work best</p>')
    title("How it works")
    steps = [("1", "Upload", "Drop in your resume as a PDF. Text is extracted in your session."),
             ("2", "Analyze", "Get a score, skill map, ATS check, writing review and best-fit roles."),
             ("3", "Improve", "Paste a job description to find skill gaps and get tailored advice.")]
    md(grid("grid3", *[f'<div class="card step"><div class="n">{n}</div><b>{h}</b><p>{d}</p></div>'
                       for n, h, d in steps]))


def main():
    load_css()
    nlp = load_nlp()

    md("""
    <div class="hero">
      <div class="eyebrow">Career toolkit</div>
      <h1>AI Resume Analyzer</h1>
      <p>Upload your resume for an instant score, ATS check, skill insights and role recommendations —
         then compare it against any job description.</p>
      <div class="pills"><span>📄 PDF parsing</span><span>🧠 NLP-powered</span>
        <span>🛡 ATS check</span><span>🎯 Job matching</span></div>
    </div>""")
    md('<div class="gap"></div>')

    uploaded = st.file_uploader("Upload your resume (PDF)", type="pdf", label_visibility="collapsed")
    if not uploaded:
        landing()
        return

    try:
        with st.spinner("Analyzing your resume…"):
            a = analyze(uploaded, nlp)
    except Exception:
        st.error("We couldn't read that file. Please upload a valid, non-encrypted PDF.")
        return
    if not a.text:
        st.warning("No text found. This looks like a scanned/image PDF — please upload a text-based version.")
        return

    tabs = st.tabs(["Overview", "Skills", "ATS & Writing", "Role fit", "Job match", "Tips"])
    with tabs[0]:
        render_overview(a)
    with tabs[1]:
        render_skills(a)
    with tabs[2]:
        render_ats_writing(a)
    with tabs[3]:
        render_roles(a)
    with tabs[4]:
        match = render_match(a)
    with tabs[5]:
        render_tips(a)

    md('<div class="gap"></div>')
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇ Report (.md)", report_markdown(a, match), file_name="resume_analysis.md", mime="text/markdown")
    d2.download_button("⬇ Data (.json)", report_json(a, match), file_name="resume_analysis.json", mime="application/json")
    md('<div class="footer">Built with Streamlit · spaCy · pdfplumber</div>')


main()
