"""Download reference PDFs for the thesis bibliography.

For each reference with a known arXiv ID or open-access URL, download the PDF
into docs/references/<key>.pdf. Books and paywalled journal articles cannot be
auto-downloaded; they are listed as 'manual' so the user knows.

Run:  python3 docs/download_references.py
"""
from pathlib import Path
import urllib.request
import urllib.error
import time

BASE = Path(__file__).parent
REF_DIR = BASE / "references"
REF_DIR.mkdir(exist_ok=True)

# (key, source_kind, url_or_note)
#   "arxiv:XXXX.XXXXX"   -> downloaded from arxiv.org/pdf/XXXX.XXXXX.pdf
#   "url:https://..."    -> downloaded directly
#   "manual:..."         -> not downloadable, note shown to user
REFERENCES = [
    ("Park2023",           "arxiv:2304.03442"),
    ("Russell1980",        "manual:Journal article, Russell 1980 — paywalled (APA PsycNet)"),
    ("Tsai2019",           "arxiv:1906.00295"),
    ("RussellNorvig2020",  "manual:Textbook — not auto-downloadable"),
    ("Breazeal2003",       "manual:Journal article — paywalled"),
    ("RaoGeorgeff1995",    "manual:Conference paper — paywalled"),
    ("Picard1997",         "manual:Book — not auto-downloadable"),
    ("Ekman1992",          "manual:Journal article — paywalled"),
    ("Zadeh2016",          "arxiv:1606.06259"),
    ("Liu2018",            "arxiv:1806.00064"),
    ("Zadeh2017",          "arxiv:1707.07250"),
    ("Zadeh2018",          "arxiv:1802.00927"),
    ("Yu2021",             "arxiv:2102.04830"),
    ("Han2021",            "arxiv:2109.00412"),
    ("Yang2023",           "url:https://aclanthology.org/2023.acl-long.421.pdf"),
    ("Hazarika2020",       "arxiv:2005.03545"),
    ("Zhang2023",          "url:https://aclanthology.org/2023.emnlp-main.359.pdf"),
    ("Mai2025",            "manual:IEEE TAFFC 2025 paper — paywalled (DOI: 10.1109/TAFFC.2025.3530172). No arXiv preprint available; access via IEEE Xplore institutional subscription."),
    ("Hu2022",             "arxiv:2211.11256"),
    ("Liang2023",          "arxiv:2302.12247"),
    ("StoneVeloso2000",    "manual:Journal article — paywalled"),
    ("Frijda1986",         "manual:Book — not auto-downloadable"),
    ("Vaswani2017",        "arxiv:1706.03762"),
    ("Brown2020",          "arxiv:2005.14165"),
    ("He2021",             "arxiv:2111.09543"),
    ("Diaz2019",           "arxiv:1908.10996"),
    ("Liu2025MMAFFBen",    "arxiv:2502.11451"),
    ("Mohammad2018",       "url:https://aclanthology.org/S18-1001.pdf"),
    ("LiangXB2021",        "arxiv:2106.14448"),
    ("Wortsman2022",       "arxiv:2203.05482"),
    ("Izmailov2018",       "arxiv:1803.05407"),
    ("Furlanello2018",     "arxiv:1805.04770"),
    ("Oord2018",           "arxiv:1807.03748"),
]


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def download(url: str, dest: Path):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}")
        data = resp.read()
    dest.write_bytes(data)
    return len(data)


def main():
    ok, skip, fail = 0, 0, 0
    for key, src in REFERENCES:
        dest = REF_DIR / f"{key}.pdf"
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"  cached  {key}.pdf  ({dest.stat().st_size:,} bytes)")
            ok += 1
            continue

        if src.startswith("manual:"):
            note = src.split("manual:", 1)[1]
            print(f"  manual  {key}  — {note}")
            skip += 1
            continue

        if src.startswith("arxiv:"):
            arxiv_id = src.split("arxiv:", 1)[1]
            url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        elif src.startswith("url:"):
            url = src.split("url:", 1)[1]
        else:
            print(f"  ?       {key}  — unknown source {src!r}")
            fail += 1
            continue

        try:
            n = download(url, dest)
            print(f"  ok      {key}.pdf  ({n:,} bytes)  from {url}")
            ok += 1
            time.sleep(1.5)  # be kind to servers
        except urllib.error.HTTPError as e:
            print(f"  fail    {key}  — HTTP {e.code} from {url}")
            fail += 1
        except Exception as e:
            print(f"  fail    {key}  — {type(e).__name__}: {e}  from {url}")
            fail += 1

    print()
    print(f"Total: {len(REFERENCES)}   downloaded/cached: {ok}   manual: {skip}   failed: {fail}")


if __name__ == "__main__":
    main()
