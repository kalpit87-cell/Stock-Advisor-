"""Tests download.py against a local mock server (no internet needed).
Run:  python tests/test_download.py"""
import io, json, os, sys, tempfile, threading, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import download


def make_pr_zip():
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("Pd151024.csv", "MKT,SERIES,SYMBOL\nN,EQ,AAA\n")
        z.writestr("Bc151024.csv", "SERIES,SYMBOL\n")
    return b.getvalue()


SEC = b"SYMBOL, SERIES, DATE1, PREV_CLOSE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, LAST_PRICE, CLOSE_PRICE, AVG_PRICE, TTL_TRD_QNTY, TURNOVER_LACS, NO_OF_TRADES, DELIV_QTY, DELIV_PER\n" + b"AAA, EQ, 15-Oct-2024, 1, 1, 1, 1, 1, 1, 1, 10, 1, 1, 5, 50\n" * 300
HITS = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        HITS[self.path] = HITS.get(self.path, 0) + 1
        n = HITS[self.path]
        body, code = b"", 404
        if self.path == "/":
            body, code = b"ok", 200
        elif self.path == "/pr/PR151024.zip":
            if n == 1:
                code = 403                                # transient block: client must retry
            else:
                body, code = make_pr_zip(), 200
        elif self.path == "/pr/PR161024.zip":
            body, code = b"<html>Access denied</html>", 200   # blocked page with status 200: must be rejected
        elif self.path in ("/sec/sec_bhavdata_full_15102024.csv", "/sec/sec_bhavdata_full_20102024.csv"):
            body, code = SEC, 200            # the 20th (a Sunday) serves a copy of the 15th, like NSE does on non-trading days
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def args(base, out, *extra):
    return ["--out", out, "--delay", "0", "--retries", "2", "--warmup-url", base + "/", "--url-pr", base + "/pr/PR{ddmmyy}.zip",
            "--url-sec", base + "/sec/sec_bhavdata_full_{ddmmyyyy}.csv", *extra]


def test_all():
    srv, base = start()
    download.time.sleep = lambda s: None                  # no real waiting in tests
    with tempfile.TemporaryDirectory() as out:
        a = args(base, out, "--start", "2024-10-12", "--end", "2024-10-20", "--sources", "pr,sec")
        rc = download.main(a)
        files = sorted(os.listdir(out))
        assert "PR151024.zip" in files and "sec_bhavdata_full_15102024.csv" in files, files
        assert not any(f.endswith(".part") for f in files)
        assert "sec_bhavdata_full_20102024.csv" not in files, files        # copy of another day must not be saved
        assert HITS["/pr/PR151024.zip"] >= 2              # 403 on the first request, retried successfully
        assert rc == 1                                    # PR161024 returned an invalid page -> failure reported
        failed = open(os.path.join(out, "failed.txt")).read()
        assert "pr,2024-10-16" in failed and "invalid content" in failed, failed
        man = json.load(open(os.path.join(out, "_manifest.json")))
        assert "20241012" in man["pr"]["missing"] and "20241014" in man["sec"]["missing"]   # holidays/weekends remembered
        assert "20241016" not in man["pr"]["missing"]     # failures are NOT remembered as missing
        before = dict(HITS)
        download.main(a + ["--no-preflight"])            # re-run: only the failed date may be re-requested
        # existing files and older remembered-missing dates are not requested again (the last 3 days are rechecked on purpose)
        for k in ("/pr/PR151024.zip", "/sec/sec_bhavdata_full_15102024.csv", "/pr/PR141024.zip", "/sec/sec_bhavdata_full_14102024.csv",
                  "/pr/PR121024.zip", "/pr/PR131024.zip"):
            assert HITS.get(k) == before.get(k), (k, HITS.get(k), before.get(k))
        # update mode starts after the latest PR file
        rc = download.main(args(base, out, "--update", "--end", "2024-10-15", "--sources", "pr", "--no-preflight"))
        assert rc == 0
    with tempfile.TemporaryDirectory() as out:           # wrong link: pre-flight must stop with a clear message
        bad = args(base, out, "--start", "2024-10-14", "--end", "2024-10-18", "--sources", "pr")
        bad[bad.index("--url-pr") + 1] = base + "/nothing/PR{ddmmyy}.zip"
        try:
            download.main(bad); assert False
        except SystemExit as e:
            assert "link pattern may be wrong" in str(e)
        assert download.main(args(base, out, "--check", "--end", "2024-10-18", "--sources", "sec")) == 0
    srv.shutdown()


def test_validators():
    from datetime import date
    assert download.validate_sec(SEC, date(2024, 10, 15))[0]
    ok, why = download.validate_sec(SEC, date(2024, 10, 20))
    assert not ok and why.startswith("other-date")
    assert download.validate_pr(make_pr_zip())[0] and not download.validate_pr(b"<html>")[0]
    assert download.validate_sec(SEC)[0] and not download.validate_sec(b"<html>blocked</html>")[0]


def test_purge():
    with tempfile.TemporaryDirectory() as out:
        open(os.path.join(out, "sec_bhavdata_full_15102024.csv"), "wb").write(SEC)
        open(os.path.join(out, "sec_bhavdata_full_20102024.csv"), "wb").write(SEC)
        download.purge_mismatched(out)
        assert sorted(os.listdir(out)) == ["sec_bhavdata_full_15102024.csv"]


if __name__ == "__main__":
    test_validators(); print("PASS test_validators")
    test_all(); print("PASS test_all")
    test_purge(); print("PASS test_purge")

