"""Resume analysis engine: text extraction, entity detection, scoring, JD matching.

Kept free of any Streamlit code so it can be unit-tested or reused elsewhere.
"""
from __future__ import annotations

import re
from datetime import date
from collections import Counter
from dataclasses import dataclass, field
from typing import IO

import pdfplumber

# --------------------------------------------------------------------------
# Skills database (category -> {canonical skill: [aliases]})
# --------------------------------------------------------------------------
SKILLS: dict[str, dict[str, list[str]]] = {
    "Programming": {
        "python": [], "java": [], "c++": [], "c#": [], "javascript": ["js"],
        "typescript": [], "go": ["golang"], "rust": [], "php": [], "kotlin": [],
        "swift": [], "r": [], "matlab": [], "bash": ["shell scripting"],
    },
    "Web": {
        "html": ["html5"], "css": ["css3"], "react": ["react.js", "reactjs"],
        "angular": [], "vue": ["vue.js"], "node.js": ["nodejs", "node"],
        "django": [], "flask": [], "fastapi": [], "spring boot": [],
        "rest api": ["restful", "rest apis"], "tailwind": [], "bootstrap": [],
    },
    "Data & ML": {
        "machine learning": ["ml"], "deep learning": [], "data analysis": ["data analytics"],
        "data visualization": [], "nlp": ["natural language processing"],
        "computer vision": [], "pandas": [], "numpy": [], "scikit-learn": ["sklearn"],
        "tensorflow": [], "pytorch": [], "keras": [], "statistics": [],
        "power bi": [], "tableau": [], "excel": [], "spark": ["pyspark"],
    },
    "Databases": {
        "sql": [], "mysql": [], "postgresql": ["postgres"], "mongodb": [],
        "sqlite": [], "redis": [], "firebase": [], "oracle": [],
    },
    "Cloud & DevOps": {
        "aws": ["amazon web services"], "azure": [], "gcp": ["google cloud"],
        "docker": [], "kubernetes": ["k8s"], "git": [], "github": [],
        "ci/cd": ["cicd", "jenkins", "github actions"], "linux": [],
    },
    "Soft skills": {
        "leadership": [], "communication": [], "teamwork": ["collaboration"],
        "problem solving": ["problem-solving"], "project management": [],
        "agile": ["scrum"], "time management": [],
    },
}

ALL_SKILLS = {s: aliases for group in SKILLS.values() for s, aliases in group.items()}
SKILL_CATEGORY = {s: cat for cat, group in SKILLS.items() for s in group}

SECTION_PATTERNS = {
    "Summary": r"\b(summary|objective|profile|about me)\b",
    "Experience": r"\b(experience|employment|work history|internship)s?\b",
    "Education": r"\b(education|academic|qualification)s?\b",
    "Skills": r"\b(skills|technical skills|core competencies|technologies)\b",
    "Projects": r"\bprojects?\b",
    "Certifications": r"\b(certifications?|certificates?|courses?|licenses?)\b",
}

ACTION_VERBS = {
    "developed", "built", "designed", "led", "managed", "created", "implemented",
    "improved", "increased", "reduced", "optimized", "launched", "delivered",
    "analyzed", "automated", "collaborated", "engineered", "deployed", "achieved",
    "streamlined", "mentored", "architected", "established", "spearheaded",
}

ROLES: dict[str, list[str]] = {
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "scikit-learn", "sql", "data visualization", "deep learning"],
    "Data Analyst": ["sql", "excel", "tableau", "power bi", "python", "data analysis", "data visualization", "statistics"],
    "ML Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker", "git", "aws"],
    "Frontend Developer": ["html", "css", "javascript", "react", "typescript", "git", "tailwind", "rest api"],
    "Backend Developer": ["python", "java", "node.js", "sql", "rest api", "docker", "git", "postgresql", "django", "flask"],
    "Full-Stack Developer": ["javascript", "react", "node.js", "html", "css", "sql", "rest api", "git", "mongodb"],
    "DevOps Engineer": ["linux", "docker", "kubernetes", "aws", "ci/cd", "git", "bash", "azure"],
}

WEAK_PHRASES = [
    "responsible for", "worked on", "helped with", "helped to", "duties included",
    "assisted in", "assisted with", "familiar with", "tried to", "various", "etc",
]
BUZZWORDS = [
    "team player", "hard worker", "hardworking", "detail-oriented", "results-driven",
    "self-motivated", "go-getter", "synergy", "think outside the box", "fast learner",
    "dynamic", "passionate",
]
BULLET_RE = re.compile(r"^\s*[\u2022\u25cf\u25aa\u25e6\-\*\u2013\u2014]\s+")

STOPWORDS = set(
    "a an the and or of to in for on with at by from as is are was were be been this that "
    "it its our your you we they their will can should must have has had not but if than "
    "such etc per within across using use used work working team role job candidate "
    "experience years year strong good ability required preferred responsibilities "
    "skills knowledge looking join about who what which also more other new "
    "need needs plus developer engineer plan strong understanding familiarity "
    "proficiency excellent responsible ideal candidates opportunity company".split()
)


@dataclass
class Analysis:
    text: str = ""
    word_count: int = 0
    page_count: int = 0
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    linkedin: str | None = None
    github: str | None = None
    skills: dict[str, list[str]] = field(default_factory=dict)  # category -> skills
    skill_list: list[str] = field(default_factory=list)
    sections: dict[str, bool] = field(default_factory=dict)
    action_verbs: list[str] = field(default_factory=list)
    metrics_count: int = 0
    breakdown: dict[str, tuple[float, float]] = field(default_factory=dict)  # label -> (got, max)
    score: float = 0.0
    suggestions: list[tuple[str, str]] = field(default_factory=list)  # (level, text)
    years_exp: float | None = None
    roles: list[tuple[str, float, list[str], list[str]]] = field(default_factory=list)  # name, %, have, missing
    radar: dict[str, float] = field(default_factory=dict)  # category -> 0..100
    weak_phrases: dict[str, int] = field(default_factory=dict)
    buzzwords: list[str] = field(default_factory=list)
    pronouns: int = 0
    bullets: int = 0
    avg_bullet_words: float = 0.0
    ats_checks: list[tuple[str, bool, str]] = field(default_factory=list)  # label, passed, hint
    ats_score: float = 0.0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _term_regex(term: str) -> re.Pattern:
    """Whole-word match that also works for terms like c++, c#, node.js."""
    return re.compile(rf"(?<![\w+#.]){re.escape(term)}(?![\w+#]|\.\w)", re.IGNORECASE)


_SKILL_REGEX = {
    skill: [_term_regex(t) for t in [skill, *aliases]] for skill, aliases in ALL_SKILLS.items()
}


def find_skills(text: str) -> list[str]:
    """Return canonical skills present in text (whole-word matching)."""
    return [s for s, patterns in _SKILL_REGEX.items() if any(p.search(text) for p in patterns)]


def extract_text(file: IO[bytes]) -> tuple[str, int]:
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n".join(pages).strip(), len(pages)


def extract_name(text: str, nlp) -> str | None:
    head = "\n".join(text.splitlines()[:8])
    doc = nlp(head)
    for ent in doc.ents:
        if ent.label_ == "PERSON" and 1 < len(ent.text.split()) <= 4 and "\n" not in ent.text:
            return ent.text.strip()
    # Fallback: first short line made only of letters
    for line in text.splitlines()[:5]:
        line = line.strip()
        if 1 < len(line.split()) <= 4 and re.fullmatch(r"[A-Za-z.\- ]+", line):
            return line.title() if line.isupper() else line
    return None


def extract_phone(text: str) -> str | None:
    for m in re.finditer(r"\+?\(?\d[\d\s().-]{8,16}\d", text):
        cand = m.group(0).strip()
        digits = re.sub(r"\D", "", cand)
        if 10 <= len(digits) <= 13 and not re.fullmatch(r"(?:19|20)\d{2}\D+(?:19|20)\d{2}", cand):
            return cand
    return None


def _first(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(0) if m else None


# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------
def analyze(file: IO[bytes], nlp) -> Analysis:
    text, pages = extract_text(file)
    a = Analysis(text=text, page_count=pages, word_count=len(text.split()))
    if not text:
        return a

    a.name = extract_name(text, nlp)
    a.email = _first(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    a.phone = extract_phone(text)
    a.linkedin = _first(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-%/]+", text)
    a.github = _first(r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+", text)

    # Skills
    a.skill_list = find_skills(text)
    for s in a.skill_list:
        a.skills.setdefault(SKILL_CATEGORY[s], []).append(s)

    # Sections
    lines = [ln.strip() for ln in text.splitlines()]
    headings = [ln for ln in lines if 0 < len(ln) <= 40]
    for sec, pat in SECTION_PATTERNS.items():
        a.sections[sec] = any(re.search(pat, h, re.IGNORECASE) for h in headings)

    # Content quality
    words = re.findall(r"[a-zA-Z]+", text.lower())
    a.action_verbs = sorted({w for w in words if w in ACTION_VERBS})
    a.metrics_count = len(re.findall(r"\d+(?:\.\d+)?\s*(?:%|\+|k\b|x\b)|\$\s?\d+|\b\d{2,}\s+(?:users|clients|projects|customers|members)", text, re.IGNORECASE))

    _extras(a, lines)
    _score(a)
    return a


def _experience_segment(lines: list[str]) -> str:
    """Text between the Experience heading and the next recognised heading."""
    start = None
    for i, ln in enumerate(lines):
        if 0 < len(ln) <= 40 and re.search(SECTION_PATTERNS["Experience"], ln, re.IGNORECASE):
            start = i + 1
            break
    if start is None:
        return ""
    others = [p for k, p in SECTION_PATTERNS.items() if k != "Experience"]
    out = []
    for ln in lines[start:]:
        if 0 < len(ln) <= 40 and any(re.fullmatch(rf"\W*{p}\W*", ln, re.IGNORECASE) or
                                      (len(ln.split()) <= 3 and re.search(p, ln, re.IGNORECASE)) for p in others):
            break
        out.append(ln)
    return "\n".join(out)


def estimate_years(text: str, lines: list[str]) -> float | None:
    seg = _experience_segment(lines) or text
    now = date.today().year
    spans = []
    for m in re.finditer(r"((?:19|20)\d{2})\s*(?:[-\u2013\u2014]|to)\s*(?:[A-Za-z]{3,9}\.?\s*)?((?:19|20)\d{2}|present|current|now|ongoing)", seg, re.IGNORECASE):
        start = int(m.group(1))
        end = now if not m.group(2).isdigit() else int(m.group(2))
        if start <= end <= now and end - start <= 40:
            spans.append((start, end))
    if not spans:
        return None
    spans.sort()
    total, (cs, ce) = 0, spans[0]
    for s_, e_ in spans[1:]:
        if s_ <= ce:
            ce = max(ce, e_)
        else:
            total += ce - cs
            cs, ce = s_, e_
    return float(total + ce - cs)


def _extras(a: Analysis, lines: list[str]) -> None:
    text, low = a.text, a.text.lower()
    a.years_exp = estimate_years(text, lines)

    # Role fit
    have = set(a.skill_list)
    roles = []
    for name, req in ROLES.items():
        got = [r for r in req if r in have]
        roles.append((name, len(got) / len(req) * 100, got, [r for r in req if r not in have]))
    a.roles = sorted(roles, key=lambda r: -r[1])[:4]

    # Radar (a category is "full" at 5 distinct skills)
    a.radar = {cat: min(len(a.skills.get(cat, [])) / 5, 1) * 100 for cat in SKILLS}

    # Writing quality
    a.weak_phrases = {p: n for p in WEAK_PHRASES if (n := len(re.findall(rf"\b{re.escape(p)}\b", low)))}
    a.buzzwords = [b for b in BUZZWORDS if re.search(rf"\b{re.escape(b)}\b", low)]
    a.pronouns = len(re.findall(r"\bI\b|\b(?i:my|me|myself)\b", text))
    blines = [ln for ln in lines if BULLET_RE.match(ln)]
    a.bullets = len(blines)
    a.avg_bullet_words = sum(len(b.split()) for b in blines) / len(blines) if blines else 0.0

    # ATS compatibility
    allowed = set("\u2022\u2013\u2014\u2019\u2018\u201c\u201d\u2026\u25cf\u25aa\u25e6\u00b7|\u00a0")
    odd = sum(1 for ch in text if ord(ch) > 127 and ch not in allowed)
    core_found = sum(a.sections.get(k, False) for k in ["Experience", "Education", "Skills", "Summary", "Projects"])
    a.ats_checks = [
        ("Email and phone readable as text", bool(a.email and a.phone), "Put contact details in plain text, not in images or headers."),
        ("Standard section headings", core_found >= 3, "Use conventional headings like Experience, Education and Skills."),
        ("Reasonable length (1\u20132 pages)", a.page_count <= 2, "ATS and recruiters favour concise resumes; trim to two pages."),
        ("Dates on your experience", bool(re.search(r"(?:19|20)\d{2}", text)), "Include years for roles, education and projects."),
        ("Bullet points used", a.bullets >= 5, "Use bullet points to make achievements scannable."),
        ("Clean characters (no garbled symbols)", odd / max(len(text), 1) < 0.01, "Avoid icons and decorative fonts that parse as junk."),
        ("Online profile linked", bool(a.linkedin or a.github), "Link LinkedIn or GitHub in the header."),
        ("Keyword-rich (8+ recognised skills)", len(a.skill_list) >= 8, "Mirror the tools and technologies named in job posts."),
    ]
    a.ats_score = sum(ok for _, ok, _ in a.ats_checks) / len(a.ats_checks) * 100


def _score(a: Analysis) -> None:
    # 1. Skills breadth (40 pts): 12+ distinct skills = full marks
    skills_pts = min(len(a.skill_list) / 12, 1) * 40
    # 2. Structure (25 pts)
    core = ["Experience", "Education", "Skills", "Projects", "Summary"]
    structure_pts = sum(a.sections.get(s, False) for s in core) / len(core) * 25
    # 3. Contact (15 pts)
    contact = [a.name, a.email, a.phone, a.linkedin or a.github]
    contact_pts = sum(bool(c) for c in contact) / len(contact) * 15
    # 4. Content quality (20 pts): verbs, metrics, length
    verbs_pts = min(len(a.action_verbs) / 6, 1) * 8
    metrics_pts = min(a.metrics_count / 3, 1) * 7
    length_pts = 5 if 250 <= a.word_count <= 900 else 2.5 if a.word_count >= 120 else 0
    content_pts = verbs_pts + metrics_pts + length_pts

    a.breakdown = {
        "Skills": (skills_pts, 40),
        "Structure": (structure_pts, 25),
        "Contact info": (contact_pts, 15),
        "Content quality": (content_pts, 20),
    }
    a.score = round(sum(v for v, _ in a.breakdown.values()), 1)

    # Suggestions
    s = a.suggestions
    for sec in core:
        if not a.sections.get(sec):
            s.append(("warn", f"Add a clear **{sec}** section — recruiters and ATS parsers look for it."))
    if not a.email:
        s.append(("bad", "No email address detected. Make sure it is in plain text, not an image."))
    if not a.phone:
        s.append(("warn", "No phone number detected."))
    if not (a.linkedin or a.github):
        s.append(("warn", "Add a LinkedIn or GitHub link to back up your experience."))
    if len(a.skill_list) < 8:
        s.append(("warn", "Few recognised skills found. Add a dedicated skills section with specific tools and technologies."))
    if len(a.action_verbs) < 4:
        s.append(("warn", "Start bullet points with strong action verbs (built, led, optimized, deployed…)."))
    if a.metrics_count < 2:
        s.append(("warn", "Quantify your impact — e.g. “reduced load time by 40%” or “served 10k+ users”."))
    if a.word_count < 250:
        s.append(("warn", "Your resume looks short. Expand on projects and responsibilities."))
    elif a.word_count > 900:
        s.append(("info", "Your resume is long. Aim for 1–2 pages and keep only the most relevant content."))
    if a.weak_phrases:
        s.append(("warn", "Replace passive phrases (" + ", ".join(f"“{p}”" for p in list(a.weak_phrases)[:3]) +
                  ") with results-focused verbs."))
    if a.buzzwords:
        s.append(("info", "Cut clichés like " + ", ".join(f"“{b}”" for b in a.buzzwords[:3]) +
                  " — show these qualities through achievements instead."))
    if a.pronouns > 2:
        s.append(("info", "Avoid first-person pronouns (I, my). Resumes read better in implied first person."))
    failed = [h for _, ok, h in a.ats_checks if not ok]
    if failed:
        s.append(("info", f"ATS tip: {failed[0]}"))
    if not s:
        s.append(("good", "Excellent — your resume covers all the essentials. Tailor it to each job description for best results."))


# --------------------------------------------------------------------------
# Job description matching
# --------------------------------------------------------------------------
@dataclass
class Match:
    score: float
    matched_skills: list[str]
    missing_skills: list[str]
    keyword_overlap: float
    top_keywords_missing: list[str]
    advice: list[str] = field(default_factory=list)


def _keywords(text: str) -> Counter:
    words = re.findall(r"[a-zA-Z][a-zA-Z+#.\-]{2,}", text.lower())
    return Counter(w.strip(".-") for w in words if w not in STOPWORDS)


def match_job(resume_text: str, resume_skills: list[str], jd: str) -> Match:
    jd_skills = find_skills(jd)
    matched = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]
    skill_ratio = len(matched) / len(jd_skills) if jd_skills else 0.0

    jd_kw, res_kw = _keywords(jd), _keywords(resume_text)
    top = [w for w, _ in jd_kw.most_common(40)]
    overlap = sum(w in res_kw for w in top) / len(top) if top else 0.0
    skill_parts = {p for sk in jd_skills for p in re.split(r"[\s/]+", sk)}
    missing_kw = [w for w in top if w not in res_kw and w not in skill_parts][:10]

    # Weighted: skills dominate when the JD names recognisable skills
    score = (0.65 * skill_ratio + 0.35 * overlap) if jd_skills else overlap
    advice = []
    if missing:
        advice.append(f"If you have real experience with {', '.join(missing[:4])}, add it to your Skills section and a project or role bullet.")
    if missing_kw:
        advice.append(f"Work relevant terms from the posting into your bullets: {', '.join(missing_kw[:5])}.")
    if score < 0.5:
        advice.append("This is a stretch match as written — lead your summary with the strengths that overlap with the role.")
    elif score >= 0.75:
        advice.append("Strong alignment. Tailor your summary line to the job title and you're ready to apply.")
    return Match(round(score * 100, 1), matched, missing, round(overlap * 100, 1), missing_kw, advice)


def report_markdown(a: Analysis, m: Match | None) -> str:
    lines = [
        "# Resume Analysis Report", "",
        f"**Candidate:** {a.name or 'N/A'}  ",
        f"**Email:** {a.email or 'N/A'}  ",
        f"**Overall score:** {a.score:.0f}/100", "",
        "## Score breakdown",
        *[f"- {k}: {v:.1f}/{mx}" for k, (v, mx) in a.breakdown.items()], "",
        "## Detected skills",
        *[f"- **{cat}:** {', '.join(sk)}" for cat, sk in a.skills.items()], "",
        "## Best-fit roles",
        *[f"- {n}: {p:.0f}%" for n, p, _, _ in a.roles], "",
        f"## ATS compatibility: {a.ats_score:.0f}%",
        *[f"- [{'x' if ok else ' '}] {lbl}" for lbl, ok, _ in a.ats_checks], "",
        "## Suggestions",
        *[f"- {t.replace('**', '')}" for _, t in a.suggestions],
    ]
    if m:
        lines += ["", "## Job match", f"- Match score: {m.score:.0f}%",
                  f"- Matched: {', '.join(m.matched_skills) or '—'}",
                  f"- Missing: {', '.join(m.missing_skills) or '—'}"]
    return "\n".join(lines)


def report_json(a: Analysis, m: Match | None) -> str:
    import json
    from dataclasses import asdict
    d = asdict(a)
    d.pop("text", None)
    if m:
        d["job_match"] = asdict(m)
    return json.dumps(d, indent=2, default=str)
