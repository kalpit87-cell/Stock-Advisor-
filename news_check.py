"""Headline scan for the symbols in today's report. It does NOT score sentiment; it surfaces recent
headlines and highlights risk keywords so YOU read them before acting.
Uses Google News RSS (needs internet; fails soft). Search by NSE symbol can be ambiguous: always read the headline.
"""
import re
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET

RISK = ["fraud", "sebi", "probe", "raid", "ed ", "cbi", "default", "downgrade", "pledge", "resign", "penalty",
        "insolvency", "nclt", "auditor", "qualified", "delist", "suspend", "short seller", "halt", "recall", "ban"]
RISK_RE = re.compile("|".join(re.escape(w) for w in RISK), re.I)


def parse_rss(xml_text: str, n: int = 6):
    root = ET.fromstring(xml_text)
    out = []
    for item in root.iter("item"):
        out.append({"title": (item.findtext("title") or "").strip(), "date": (item.findtext("pubDate") or "").strip(),
                    "link": (item.findtext("link") or "").strip()})
        if len(out) >= n:
            break
    return out


def flag(headlines):
    for h in headlines:
        h["risk"] = sorted({m.group(0).strip().lower() for m in RISK_RE.finditer(h["title"] + " ")})
    return headlines


def fetch_headlines(symbol: str, n: int = 6, timeout: int = 10):
    q = urllib.parse.quote(f"{symbol} NSE share when:7d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return parse_rss(r.read().decode("utf-8", "replace"), n)


def news_section(symbols, fetcher=fetch_headlines):
    lines = ["## 5. Headline scan (last 7 days; read before acting)"]
    if not symbols:
        return lines + ["Nothing to scan.", ""]
    for s in symbols:
        try:
            hs = flag(fetcher(s))
        except Exception as e:  # fail soft: no internet, blocked, bad XML
            lines.append(f"- **{s}**: news fetch failed ({type(e).__name__}); check manually")
            continue
        if not hs:
            lines.append(f"- **{s}**: no recent headlines found")
            continue
        risky = [h for h in hs if h["risk"]]
        lines.append(f"- **{s}**: {len(hs)} headlines" + (f", **{len(risky)} with risk words**" if risky else ""))
        for h in (risky or hs)[:3]:
            tag = f" [{', '.join(h['risk'])}]" if h["risk"] else ""
            lines.append(f"    - {h['title']}{tag}")
    lines.append("")
    return lines
