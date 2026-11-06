"""Build paper/v2/refs.bib from the source records, not from memory.

Each reference is fetched from where it is registered:
  - "doi":    Crossref record for a known DOI (title must match the expected title);
  - "search": Crossref title search; accepts the first hit whose title matches exactly
              (after lower-casing and dropping punctuation), so no DOI is typed by hand;
  - "arxiv":  the arXiv API record for the id (title must match).
"manual" entries have no registry record (software, a video, an old journal with no DOI, our
own 2025 paper); each says where it was checked, and the check is listed in refs_check.md.

The raw responses are saved in paper/v2/refs_sources/ so every field can be traced.
Run:  python3 scripts/report/build_bib.py            (fetches; needs network)
      python3 scripts/report/build_bib.py --offline  (rebuilds from the saved responses)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "v2"
SRC_DIR = OUT_DIR / "refs_sources"
UA = "evo2-report-bib/1.0 (reference check script)"

# (key, kind, identifier, expected title)
REFS = [
    # The paper the report answers, and its current restatement
    ("milano2019moderate", "doi", "10.1162/artl_a_00274",
     "Moderate Environmental Variation Across Generations Promotes the Evolution of Robust Solutions"),
    ("milano2017arxiv", "arxiv", "1710.07913", None),
    ("carvalho2024morphological", "doi", "10.1162/evco_a_00336",
     "The Role of Morphological Variation in Evolutionary Robotics: Maximizing Performance and Robustness"),
    # Related experiments and arguments
    ("milano2024qualitative", "search", None,
     "Qualitative differences between evolutionary strategies and reinforcement learning methods for control of autonomous agents"),
    ("gupta2021embodied", "search", None, "Embodied intelligence via learning and evolution"),
    ("dohare2024loss", "search", None, "Loss of plasticity in deep continual learning"),
    ("canino2019fluctuating", "search", None,
     "Fluctuating environments select for short-term phenotypic variation leading to long-term exploration"),
    ("bredeche2012environment", "search", None,
     "Environment-driven distributed evolutionary adaptation in a population of autonomous robotic agents"),
    ("nisioti2022plasticity", "doi", "10.1145/3512290.3528826",
     "Plasticity and evolvability under environmental variability: the joint role of fitness-based selection and niche-limited competition"),
    ("kashtan2007varying", "search", None, "Varying environments can speed up evolution"),
    ("parter2008facilitated", "search", None,
     "Facilitated Variation: How Evolution Learns from Past Environments To Generalize to New Environments"),
    ("hamon2023eco", "search", None,
     "Eco-evolutionary Dynamics of Non-episodic Neuroevolution in Large Multi-agent Environments"),
    ("lehman2025knightian", "arxiv", "2501.13075", None),
    ("nisioti2025transfer", "arxiv", "2505.22696", None),
    ("nisioti2026continual", "arxiv", "2610.01583", None),
    ("lu2024jaxlife", "arxiv", "2409.00853", None),
    ("wang2019poet", "arxiv", "1901.01753", None),
    ("clune2019aigas", "arxiv", "1905.10985", None),
    ("xland2021", "arxiv", "2107.12808", None),
    ("dennis2020paired", "arxiv", "2012.02096", None),
    ("bhatia2021evogym", "arxiv", "2201.09863", None),
    ("mouret2015illuminating", "arxiv", "1504.04909", None),
    ("fernando2017baldwin", "arxiv", "1702.08411", None),
    ("bejjani2025ecological", "arxiv", "2510.18221",
     "The Emergence of Complex Behavior in Large-Scale Ecological Environments"),
    # Methods and evaluation
    ("salimans2017evolution", "arxiv", "1703.03864", None),
    ("schulman2017proximal", "arxiv", "1707.06347", None),
    ("agarwal2021deep", "arxiv", "2108.13264", None),
    ("henderson2018deep", "arxiv", "1709.06560", "Deep Reinforcement Learning that Matters"),
    ("colas2018seeds", "arxiv", "1806.08295", None),
    ("cobbe2020leveraging", "arxiv", "1912.01588", None),
    ("tobin2017domain", "search", None,
     "Domain randomization for transferring deep neural networks from simulation to the real world"),
    ("lange2023evosax", "search", None, "evosax: JAX-Based Evolution Strategies"),
    ("lu2022discovered", "arxiv", "2210.05639", None),
]

# No registry record. "checked" says where each field was read; the date is the check date.
MANUAL = {
    "ge2026arena": {
        "type": "misc",
        "fields": {"author": "Ge, Yuxu and Cheng, Yifei",
                   "title": "Neuroevolution Arena: Nested Ecological Evaluation of Update-and-Inheritance Regimes across Neural Architectures",
                   "year": "2026", "howpublished": "arXiv:2608.10323v1",
                   "note": "Preprint, 10 August 2026",
                   "url": "https://arxiv.org/abs/2608.10323"},
        "checked": "6 Nov 2026 UTC: metadata read directly from the title page of arXiv:2608.10323v1; saved in results/audit/literature/ge2026arena.pdf. API was rate limited; this is a direct-PDF check, not a registry check.",
    },
    "hinton1987learning": {
        "type": "article",
        "fields": {"author": "Hinton, Geoffrey E. and Nowlan, Steven J.",
                   "title": "How Learning Can Guide Evolution",
                   "journal": "Complex Systems", "volume": "1",
                   "pages": "495--502", "year": "1987"},
        "checked": "4 Nov 2026: complex-systems.com/?p=1656 (title, authors) and cs.toronto.edu/~hinton/absps/evolution.htm (Complex Systems 1, 495-502, 1987); issue number not shown, so left out",
    },
    "bradbury2018jax": {
        "type": "misc",
        "fields": {"author": "Bradbury, James and Frostig, Roy and Hawkins, Peter and Johnson, Matthew James and Katariya, Yash and Leary, Chris and Maclaurin, Dougal and Necula, George and Paszke, Adam and VanderPlas, Jake and Wanderman-Milne, Skye and Zhang, Qiao",
                   "title": "{JAX}: composable transformations of {P}ython+{N}um{P}y programs",
                   "howpublished": "\\url{http://github.com/jax-ml/jax}",
                   "note": "Version 0.11.2 used in Study 3 (requirements\\_part3.txt)", "year": "2018"},
        "checked": "5 Nov 2026: official jax-ml/jax CITATION.bib (authors, title, URL, year), saved in results/audit/reference_sources/jax_citation.bib; experimental version remains from requirements_part3.txt, not the generic citation version",
    },
    "palfree_evolution_simulator": {
        "type": "misc",
        "fields": {"author": "Palfree, Jasper",
                   "title": "Evolution Simulator ({MinuteLabs.io})",
                   "howpublished": "\\url{https://github.com/minutelabsio/evolution-simulator}, commit 55af956",
                   "note": "GPL-3.0", "year": "2021"},
        "checked": "5 Nov 2026: upstream commit API confirms 55af956a085189e1f307037555fc4d2b6ee2e95a dated 17 May 2021; upstream LICENSE confirms GPL-3.0; minutelabs.io/about credits Jasper Palfree for coding, concept and creation. Raw commit/license records saved in results/audit/reference_sources",
    },
    "helps2018primer": {
        "type": "misc",
        "fields": {"author": "{Primer}",
                   "title": "Simulating Natural Selection",
                   "howpublished": "Primer, YouTube video, 14 November 2018, \\url{https://www.youtube.com/watch?v=0ZGbIKd0XrM}",
                   "year": "2018"},
        "checked": "5 Nov 2026: original YouTube video metadata confirms title, channel Primer and 14 Nov 2018; cite the verified publishing channel rather than an unverified personal attribution",
    },
    "guntaka2025evolving": {
        "type": "unpublished",
        "fields": {"author": "Guntaka, Abhinav R.",
                   "title": "Evolving Intelligence: What Biology Can Teach {AI} About Getting Smarter",
                   "note": "Manuscript", "year": "2025"},
        "checked": "title and author from page 1 of paper/original_paper.pdf; year and venue TO CONFIRM with AA (the repo calls it \"Evolving (Artificial) Intelligence\")",
    },
}


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", re.sub(r"<[^>]+>", "", t or "").lower())


def curl(url: str) -> bytes:
    for attempt in range(3):
        r = subprocess.run(["curl", "-sS", "-m", "40", "-A", UA, "-w", "\n%{http_code}", url],
                           capture_output=True)
        body, _, code = r.stdout.rpartition(b"\n")
        if r.returncode == 0 and code.strip() == b"200":
            return body
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"fetch failed: {url} (curl {r.returncode}, http {code!r}) {r.stderr[:200]!r}")


def fetch(key: str, kind: str, ident: str | None, title: str | None) -> Path:
    if kind == "doi":
        url = "https://api.crossref.org/works/" + urllib.parse.quote(ident, safe="/")
        path = SRC_DIR / f"{key}.crossref.json"
    elif kind == "search":
        url = ("https://api.crossref.org/works?rows=5&query.bibliographic="
               + urllib.parse.quote(title))
        path = SRC_DIR / f"{key}.crossref_search.json"
    elif kind == "arxiv":
        url = "https://export.arxiv.org/api/query?id_list=" + ident
        path = SRC_DIR / f"{key}.arxiv.xml"
    else:
        raise ValueError(kind)
    path.write_bytes(curl(url))
    time.sleep(1.0 if kind != "arxiv" else 3.0)   # arXiv asks for 3 s between calls
    return path


def tex(s: str) -> str:
    s = re.sub(r"<[^>]+>", "", s or "")           # Crossref titles can carry JATS tags
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    return s


def from_crossref(msg: dict, expected: str | None) -> tuple[str, dict]:
    t = (msg.get("title") or [""])[0]
    sub = (msg.get("subtitle") or [""])[0]
    if sub and expected and norm(t + sub) == norm(expected):
        t = f"{t}: {sub}"                          # Crossref keeps the subtitle apart
    if expected and norm(t) != norm(expected):
        raise ValueError(f"title mismatch: got {t!r}, expected {expected!r}")
    authors = " and ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ")
                           for a in msg.get("author", []))
    year = (msg.get("published-print") or msg.get("issued") or msg.get("published-online")
            or {}).get("date-parts", [[None]])[0][0]
    f = {"author": tex(authors), "title": "{" + tex(t) + "}", "year": str(year),
         "doi": msg["DOI"]}
    ctype = msg.get("type")
    container = tex((msg.get("container-title") or [""])[0])
    if ctype == "journal-article":
        btype = "article"
        f["journal"] = container
    elif ctype == "proceedings-article":
        btype = "inproceedings"
        f["booktitle"] = container
    else:
        btype = "misc"
        if container:
            f["howpublished"] = container
    for src, dst in (("volume", "volume"), ("issue", "number"), ("page", "pages"),
                     ("article-number", "pages"), ("publisher", "publisher")):
        if msg.get(src) and dst not in f:
            f[dst] = tex(str(msg[src])).replace("-", "--") if dst == "pages" else tex(str(msg[src]))
    return btype, f


def from_arxiv(path: Path, ident: str) -> tuple[str, dict]:
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entry = ET.parse(path).getroot().find("a:entry", ns)
    if entry is None or entry.find("a:title", ns) is None:
        raise ValueError(f"arXiv {ident}: no entry")
    t = " ".join(entry.find("a:title", ns).text.split())
    names = [" ".join(a.find("a:name", ns).text.split()) for a in entry.findall("a:author", ns)]

    def last_first(n: str) -> str:
        if n in {"Open Ended Learning Team", "Open-Ended Learning Team"}:
            return "{" + n + "}"
        if n == "Christian Schroeder de Witt":
            return "de Witt, Christian Schroeder"
        parts = n.split()
        return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else n

    year = entry.find("a:published", ns).text[:4]
    cat = entry.find("x:primary_category", ns)
    f = {"author": " and ".join(("{" + tex(n) + "}") if n in {"Open Ended Learning Team", "Open-Ended Learning Team"} else tex(last_first(n)) for n in names), "title": "{" + tex(t) + "}",
         "year": year, "eprint": ident, "archivePrefix": "arXiv",
         "primaryClass": cat.get("term") if cat is not None else "",
         "howpublished": f"arXiv:{ident}"}
    jref = entry.find("x:journal_ref", ns)
    if jref is not None and jref.text:
        f["note"] = "Journal reference on arXiv: " + tex(" ".join(jref.text.split()))
    return "misc", f


def entry_text(key: str, btype: str, f: dict) -> str:
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v not in (None, "", "None"))
    return f"@{btype}{{{key},\n{body}\n}}\n"


MANUAL["carvalho2023morphologicalpreprint"] = {
    "type": "misc",
    "fields": {"author": "Carvalho, Jonata Tyska and Nolfi, Stefano",
               "title": "{The Role of Morphological Variation in Evolutionary Robotics: Maximizing Performance and Robustness}",
               "year": "2023", "eprint": "2208.02809v2", "archivePrefix": "arXiv",
               "url": "https://arxiv.org/abs/2208.02809v2",
               "note": "Version 2, 11 October 2023; preprint version consulted"},
    "checked": "Recovered full-text title page; version/date/hash in refs_sources/carvalho2023morphologicalpreprint.manual.json. Final journal version not claimed as inspected.",
}


def build(offline: bool) -> int:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    out, report, failures = [], [], []
    for key, kind, ident, title in REFS:
        suffix = {"doi": "crossref.json", "search": "crossref_search.json", "arxiv": "arxiv.xml"}[kind]
        path = SRC_DIR / f"{key}.{suffix}"
        try:
            if not offline:
                fetch(key, kind, ident, title)
            if kind == "doi":
                btype, f = from_crossref(json.loads(path.read_text())["message"], title)
            elif kind == "search":
                items = json.loads(path.read_text())["message"]["items"]
                hits = [it for it in items if norm((it.get("title") or [""])[0]) == norm(title)
                        or norm((it.get("title") or [""])[0] + (it.get("subtitle") or [""])[0]) == norm(title)]
                if not hits:
                    raise ValueError("no Crossref hit with this exact title: "
                                     + "; ".join((it.get("title") or ["?"])[0][:70] for it in items[:3]))
                btype, f = from_crossref(hits[0], title)
            else:
                btype, f = from_arxiv(path, ident)
                if title and norm(f["title"]) != norm(title):
                    raise ValueError("arXiv title mismatch")
            out.append(entry_text(key, btype, f))
            report.append(f"| `{key}` | {kind} | {f.get('doi') or f.get('eprint')} | "
                          f"{f['title'].strip('{}')[:90]} | `refs_sources/{path.name}` |")
        except Exception as e:  # noqa: BLE001  (every failure is listed, none is skipped silently)
            failures.append(f"| `{key}` | {kind} | {ident or title[:60]} | FAILED: {e} |")
    for key, m in MANUAL.items():
        out.append(entry_text(key, m["type"], m["fields"]))
        report.append(f"| `{key}` | manual | — | {m['fields']['title'][:90]} | {m['checked']} |")

    # A missing source must not silently replace a complete bibliography with
    # an incomplete one or make an automated build appear successful.
    if failures:
        print(f"Bibliography not written: {len(failures)} source checks failed")
        for failure in failures:
            print(failure)
        return 1

    (OUT_DIR / "refs.bib").write_text(
        "% Generated by scripts/report/build_bib.py. Do not edit by hand.\n\n" + "\n".join(out))
    lines = ["# Reference check (generated by scripts/report/build_bib.py)", "",
             "Fields of registry entries come from the saved Crossref or arXiv response named in the last column.",
             "Manual entries say where they were checked; any marked TO CHECK or TO CONFIRM must be settled before submission.", "",
             "| Key | Source | Id | Title | Evidence |", "|---|---|---|---|---|", *report]
    (OUT_DIR / "refs_check.md").write_text("\n".join(lines) + "\n")
    print(f"{len(out)} entries written, {len(failures)} failed")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    sys.exit(build(ap.parse_args().offline))

