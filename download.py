#!/usr/bin/env python3
"""Download NSE daily files for research use: PR package, delivery file, (optionally) UDiFF bhavcopy.

Examples
  python download.py --check                                   # test that the links work, change nothing
  python download.py --start 2016-01-01 --out data/raw         # full history (resumable, polite, ~1 hour)
  python download.py --update --out data/raw --then-build      # daily: fetch what is new, then rebuild the database
  python download.py --start 2024-10-14 --end 2024-10-18 --sources pr,sec

What it does
  * sources: pr  = PRddmmyy.zip (prices, index, corporate actions, board meetings)  <- needed
             sec = sec_bhavdata_full_ddmmyyyy.csv (delivery %)                       <- optional, on by default
             cm  = UDiFF bhavcopy zip (prices only; redundant with pr)               <- off by default
  * resumable: files already on disk are skipped; dates NSE reports as "not found" (holidays) are remembered in
    _manifest.json so they are not re-requested; failed dates are retried on the next run
  * validates every download (a blocked request often returns a web page with status 200) and writes atomically
  * pre-flight check: if a source returns nothing for the last 10 weekdays, it stops and tells you the link may be wrong or blocked
  * weekends are tried too (special Saturday/Sunday sessions exist, e.g. Budget day, Muhurat); they cost one request each, once

The URL templates below are the NSE archive links as I know them and could NOT be tested from my build environment.
If --check fails, open the file's link on nseindia.com (Market Data > Daily reports / bhavcopy), copy its pattern and pass it with
--url-pr / --url-sec / --url-cm (use {ddmmyy}, {ddmmyyyy}, {yyyymmdd} placeholders). Respect NSE's terms of use; personal research only.
"""
import argparse, io, json, os, random, re, subprocess, sys, time, zipfile
from datetime import date, datetime, timedelta, timezone

import requests

IST = timezone(timedelta(hours=5, minutes=30))
URLS = {
    "pr": "https://nsearchives.nseindia.com/archives/equities/bhavcopy/pr/PR{ddmmyy}.zip",
    "sec": "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
    "cm": "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
}
NAMES = {"pr": "PR{ddmmyy}.zip", "sec": "sec_bhavdata_full_{ddmmyyyy}.csv", "cm": "BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000_csv.zip"}
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9", "Referer": "https://www.nseindia.com/"}
WARMUP = "https://www.nseindia.com/"


# ------------------------------------------------------------------ validation
def validate_pr(b: bytes, d=None):
    if not zipfile.is_zipfile(io.BytesIO(b)):
        return False, "not a zip file (probably a blocked/HTML response)"
    names = zipfile.ZipFile(io.BytesIO(b)).namelist()
    if not any(re.match(r"^pd\d{6,8}\.csv$", os.path.basename(n), re.I) for n in names):
        return False, f"zip has no pd file: {names[:5]}"
    return True, ""


def sec_file_date(b: bytes):
    """Trade date (a date) written inside a sec_bhavdata_full file (DATE1 column of the first data row), or None."""
    try:
        lines = b[:4000].decode("utf-8", "replace").splitlines()
        head = [h.strip() for h in lines[0].split(",")]
        row = [x.strip() for x in lines[1].split(",")]
        return datetime.strptime(row[head.index("DATE1")], "%d-%b-%Y").date()
    except (ValueError, IndexError):
        return None


def validate_sec(b: bytes, d=None):
    if b"DELIV_PER" not in b[:600]:
        return False, "no DELIV_PER header (probably a blocked/HTML response)"
    inner = sec_file_date(b)
    if d is not None and inner is not None and inner != d:
        return False, f"other-date: file requested for {d} contains data for {inner} (NSE serves a copy on non-trading days)"
    return True, ""


def validate_cm(b: bytes, d=None):
    if not zipfile.is_zipfile(io.BytesIO(b)):
        return False, "not a zip file"
    z = zipfile.ZipFile(io.BytesIO(b))
    head = z.read(z.namelist()[0])[:600]
    return (True, "") if b"TckrSymb" in head else (False, "csv header missing TckrSymb")


VALIDATE = {"pr": validate_pr, "sec": validate_sec, "cm": validate_cm}


# ------------------------------------------------------------------ helpers
def fmt(template: str, d: date) -> str:
    return template.format(ddmmyy=d.strftime("%d%m%y"), ddmmyyyy=d.strftime("%d%m%Y"), yyyymmdd=d.strftime("%Y%m%d"))


def daterange(start: date, end: date, weekends: bool = True):
    d = start
    while d <= end:
        if weekends or d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def default_end() -> date:
    now = datetime.now(IST)
    return now.date() if now.hour >= 19 else now.date() - timedelta(days=1)   # files appear after the evening


class Fetcher:
    def __init__(self, warmup_url=WARMUP, delay=0.7, retries=4, timeout=(10, 60), verbose=True):
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self.warmup_url, self.delay, self.retries, self.timeout, self.verbose = warmup_url, delay, retries, timeout, verbose
        self.n_requests = 0
        self.warm()

    def warm(self):
        try:
            self.s.get(self.warmup_url, timeout=self.timeout)       # collects cookies like a browser visit
        except requests.RequestException as e:
            if self.verbose:
                print(f"  (warm-up request failed: {type(e).__name__}; continuing)")

    def get(self, url: str, validate, d=None):
        """Returns (status, bytes_or_None, reason): status in ok | missing | error."""
        reason = ""
        for attempt in range(self.retries + 1):
            if attempt:
                wait = min(60, 2 ** attempt) + random.random()
                time.sleep(wait)
            try:
                self.n_requests += 1
                r = self.s.get(url, timeout=self.timeout)
            except requests.RequestException as e:
                reason = f"{type(e).__name__}"
                continue
            if r.status_code == 404:
                return "missing", None, "404"
            if r.status_code == 200:
                ok, why = validate(r.content, d)
                if why.startswith("other-date"):
                    return "missing", None, why
                if ok:
                    time.sleep(self.delay + random.random() * 0.3)
                    return "ok", r.content, ""
                reason = f"invalid content: {why}"
                self.warm()
                continue
            reason = f"HTTP {r.status_code}"
            if r.status_code in (401, 403, 429):
                self.warm()
        return "error", None, reason


def atomic_write(path: str, data: bytes):
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def load_manifest(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return {}


# ------------------------------------------------------------------ commands
def preflight(fetcher, srcs, urls, end: date, days: int = 10):
    """Each source must deliver at least one file among the last `days` weekdays; otherwise the link is wrong or blocked."""
    bad = []
    wk = [d for d in daterange(end - timedelta(days=days * 2), end, weekends=False)][-days:][::-1]
    for src in srcs:
        found, last = None, ""
        for d in wk:
            st, _, why = fetcher.get(fmt(urls[src], d), VALIDATE[src], d)
            if st == "ok":
                found = d
                break
            last = f"{d}: {why}"
        print(f"  [{src}] " + (f"OK (e.g. {found})" if found else f"NO FILE in the last {days} weekdays; last result {last}"))
        if not found:
            bad.append(src)
    return bad


def purge_mismatched(out: str) -> int:
    """Delete sec_bhavdata_full files whose internal date differs from the date in the file name."""
    n = 0
    for f in sorted(os.listdir(out)):
        m = re.match(r"^sec_bhavdata_full_(\d{8})\.csv$", f, re.I)
        if not m:
            continue
        named = datetime.strptime(m.group(1), "%d%m%Y").date()
        inner = sec_file_date(open(os.path.join(out, f), "rb").read(4000))
        if inner is not None and inner != named:
            os.remove(os.path.join(out, f))
            n += 1
            print(f"  removed {f} (contains data for {inner})")
    print(f"{n} mismatched delivery file(s) removed")
    return 0


def run(a) -> int:
    if a.purge_mismatched:
        return purge_mismatched(a.out)
    srcs = [s.strip() for s in a.sources.split(",") if s.strip()]
    unknown = [s for s in srcs if s not in URLS]
    if unknown:
        raise SystemExit(f"unknown source(s): {unknown}; choose from {list(URLS)}")
    urls = {"pr": a.url_pr or URLS["pr"], "sec": a.url_sec or URLS["sec"], "cm": a.url_cm or URLS["cm"]}
    os.makedirs(a.out, exist_ok=True)
    end = datetime.strptime(a.end, "%Y-%m-%d").date() if a.end else default_end()
    fetcher = Fetcher(a.warmup_url, a.delay, a.retries)

    if a.check:
        print(f"Link check as of {end} (nothing is saved):")
        if a.check_date:
            d = datetime.strptime(a.check_date, "%Y-%m-%d").date()
            for src in srcs:
                st, content, why = fetcher.get(fmt(urls[src], d), VALIDATE[src], d)
                print(f"  [{src}] {fmt(urls[src], d)} -> {st} {why} {len(content) if content else ''}")
            return 0
        bad = preflight(fetcher, srcs, urls, end)
        if bad:
            print("\nProblem with:", bad, "\nOpen the same file on nseindia.com, copy the link pattern and pass --url-<source>.")
            return 1
        print("\nAll selected links work.")
        return 0

    # start date
    if a.start:
        start = datetime.strptime(a.start, "%Y-%m-%d").date()
    elif a.update:
        have = sorted(f for f in os.listdir(a.out) if re.match(r"^PR\d{6}\.zip$", f, re.I))
        if not have:
            raise SystemExit("--update needs existing PR files in --out, or give --start")
        last = max(datetime.strptime(f[2:8], "%d%m%y").date() for f in have)
        start = last + timedelta(days=1)
        print(f"Update mode: latest PR file is {last}; fetching from {start}")
    else:
        raise SystemExit("give --start YYYY-MM-DD, or --update")
    if start > end:
        print("Nothing to do: start is after end.")
        return 0

    if not a.no_preflight:
        print("Pre-flight check:")
        bad = preflight(fetcher, srcs, urls, end)
        if bad:
            raise SystemExit(f"Stopping: no file found for {bad} in the last 10 weekdays. The link pattern may be wrong or the request "
                             f"blocked. Run --check, then pass --url-<source> with the right pattern. (Use --no-preflight to override.)")

    mpath = os.path.join(a.out, "_manifest.json")
    man = load_manifest(mpath)
    stats = {s: {"downloaded": 0, "existing": 0, "missing": 0, "failed": 0} for s in srcs}
    failed = []
    recent_cut = end - timedelta(days=3)
    days = list(daterange(start, end, weekends=not a.skip_weekends))
    t0 = time.time()
    for i, d in enumerate(days, 1):
        for src in srcs:
            path = os.path.join(a.out, fmt(NAMES[src], d))
            key = d.strftime("%Y%m%d")
            miss = set(man.get(src, {}).get("missing", []))
            if os.path.exists(path) and os.path.getsize(path) > 0:
                stats[src]["existing"] += 1
                continue
            if key in miss and not a.recheck_missing:
                stats[src]["missing"] += 1
                continue
            st, content, why = fetcher.get(fmt(urls[src], d), VALIDATE[src], d)
            if st == "ok":
                atomic_write(path, content)
                stats[src]["downloaded"] += 1
                print(f"  {d} [{src}] saved ({len(content)/1024:.0f} KB)")
            elif st == "missing":
                stats[src]["missing"] += 1
                if d < recent_cut:                          # very recent dates may simply not be published yet
                    man.setdefault(src, {}).setdefault("missing", [])
                    if key not in man[src]["missing"]:
                        man[src]["missing"].append(key)
            else:
                stats[src]["failed"] += 1
                failed.append((src, str(d), why))
                print(f"  {d} [{src}] FAILED: {why}")
        if i % 50 == 0:
            json.dump(man, open(mpath, "w"))
            print(f"  ... {i}/{len(days)} days, {time.time()-t0:.0f}s elapsed")
    json.dump(man, open(mpath, "w"))
    with open(os.path.join(a.out, "failed.txt"), "w") as f:
        f.writelines(f"{s},{d},{w}\n" for s, d, w in failed)

    print("\nSummary (missing = holidays/no file at NSE):")
    for s, v in stats.items():
        print(f"  {s:4} downloaded {v['downloaded']:5}  already had {v['existing']:5}  missing {v['missing']:5}  failed {v['failed']:3}")
    if failed:
        print(f"{len(failed)} failures listed in {a.out}/failed.txt: just run the same command again to retry them.")
    if a.then_build:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_db.py"), "--raw", a.out, "--out", a.db_out]
        print("\nRunning:", " ".join(cmd))
        subprocess.run(cmd, check=False)
    return 1 if failed else 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", help="first date YYYY-MM-DD")
    ap.add_argument("--end", help="last date (default: today after 19:00 IST, else yesterday)")
    ap.add_argument("--update", action="store_true", help="start the day after the latest PR file already in --out")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--sources", default="pr,sec", help="comma list from: pr, sec, cm (default pr,sec)")
    ap.add_argument("--check", action="store_true", help="only test the links (pre-flight) and exit")
    ap.add_argument("--check-date", help="with --check: test one specific date, e.g. an old one")
    ap.add_argument("--then-build", action="store_true", help="run build_db.py afterwards")
    ap.add_argument("--db-out", default="data/daily", help="output base for --then-build")
    ap.add_argument("--delay", type=float, default=0.7, help="seconds between requests (be polite; NSE blocks fast clients)")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--skip-weekends", action="store_true")
    ap.add_argument("--recheck-missing", action="store_true", help="re-request dates previously recorded as not found")
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("--purge-mismatched", action="store_true", help="delete delivery files whose inner date differs from the file name, then exit")
    ap.add_argument("--url-pr"); ap.add_argument("--url-sec"); ap.add_argument("--url-cm")
    ap.add_argument("--warmup-url", default=WARMUP)
    return ap


def main(argv=None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
