#!/usr/bin/env python3
"""
Pi Finder: Accurate, incremental Pi digit computation and GitHub uploader.

- Accurate Chudnovsky implementation with incremental terms
- Batch-oriented progress and performance tracking
- Optional Flask API to control runs and query status
- Uploads digits and performance artifacts to GitHub (REST API)
- Professional, efficient logging (no emojis)

Usage (CLI):
  python3 pi_finder.py --digits 100000 --batch 5000 --upload --branch main

Environment:
  - Requires a GitHub token with repo scope in GITHUB_TOKEN (or override via --token-env)
  - Default repo: quantumsoldship/metapi

Endpoints (when --server is used):
  GET /api/start?digits=N&batch=M&branch=BRANCH&upload_every=BATCHES  - start computation
  GET /api/performance                                                - current stats
  GET /api/status                                                     - basic status
  GET /api/digits                                                     - current digits
  POST /api/upload                                                    - force upload artifacts now
"""

import argparse
import base64
import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Optional

import psutil
import requests
from flask import Flask, jsonify, request

# ----------------------------
# Chudnovsky Series (Incremental)
# ----------------------------

class ChudnovskyIncremental:
    """
    Incremental Chudnovsky series accumulator.

    Each term adds ~14.181647... digits of precision.
    We maintain M, L, X, S so we can add terms without recomputing earlier values.
    """
    C_CONST = 426880
    C_SQRT = 10005
    X_FACTOR = -262537412640768000  # -640320^3

    def __init__(self, max_digits: int):
        # We compute using Python integers and Decimal-like rationals via fractions (as Decimal division at each step)
        # Here we use Python integers and rational accumulation via numerator/denominator stored separately as integers to avoid precision drift.
        # However, for simplicity and speed, we store S as a rational: S_num / S_den.
        # Initialize k = 0 term:
        self.k = 0
        # M_0 = 1
        self.M_num = 1
        self.M_den = 1
        # L_0 = 13591409
        self.L = 13591409
        # X_0 = 1
        self.X = 1
        # S_0 = L_0
        self.S_num = self.L
        self.S_den = 1

        # Precompute integer sqrt(10005) at high precision using Python integers by scaling
        # We'll compute pi at the end via integer rational arithmetic with scaled sqrt.
        # Scale for sqrt to ensure enough digits: scale_factor = 10^(scale_exp)
        # digits ~= max_digits + guard
        self.guard = 20
        self.scale_exp = max_digits + self.guard
        self.scale = 10 ** self.scale_exp
        self.sqrt_10005_scaled = self._isqrt(10005 * (self.scale ** 2))  # floor(sqrt(10005)*scale)

    @staticmethod
    def _isqrt(n: int) -> int:
        """Integer square root."""
        return int(math.isqrt(n))

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        """Greatest common divisor."""
        while b:
            a, b = b, a % b
        return abs(a)

    @staticmethod
    def _reduce(num: int, den: int) -> tuple[int, int]:
        """Reduce fraction to simplest terms."""
        g = ChudnovskyIncremental._gcd(num, den)
        if g != 0:
            num //= g
            den //= g
        return num, den

    def _add_rational(self, a_num: int, a_den: int) -> None:
        """S := S + a_num/a_den"""
        # S_num/S_den + a_num/a_den = (S_num*den + a_num*S_den) / (S_den*den)
        num = self.S_num * a_den + a_num * self.S_den
        den = self.S_den * a_den
        self.S_num, self.S_den = self._reduce(num, den)

    def terms_needed_for_digits(self, digits: int) -> int:
        # Each term adds ~14.181647... digits. Use ceiling with a small guard.
        return max(1, math.ceil((digits + self.guard) / 14.181647462725477))

    def add_terms_until(self, target_terms: int) -> int:
        """
        Add terms until self.k == target_terms.
        Returns the number of new terms added.
        """
        added = 0
        while self.k < target_terms:
            # Next k
            k = self.k + 1

            # Update M:
            # M_k = M_{k-1} * ((6k-5)*(2k-1)*(6k-1)) / (k^3)
            a = (6 * k - 5) * (2 * k - 1) * (6 * k - 1)
            b = k * k * k

            # Update M_num/M_den
            # M := M * a/b
            self.M_num *= a
            self.M_den *= b
            self.M_num, self.M_den = self._reduce(self.M_num, self.M_den)

            # Update L, X
            self.L += 545140134
            self.X *= self.X_FACTOR

            # term = (M * L) / X
            # i.e., term_num/term_den = (M_num * L) / (M_den * X)
            term_num = self.M_num * self.L
            term_den = self.M_den * self.X

            # Handle sign from X (X alternates sign)
            if term_den < 0:
                term_den = -term_den
                term_num = -term_num

            # Reduce term
            term_num, term_den = self._reduce(term_num, term_den)

            # S += term
            self._add_rational(term_num, term_den)

            self.k = k
            added += 1

        return added

    def compute_pi_string(self) -> str:
        """
        Compute pi as a decimal string "3.<digits...>" using current S.

        pi = (426880 * sqrt(10005)) / S

        We already hold sqrt(10005) as an integer scaled by 10^scale_exp.
        So:
            pi_scaled = (C_CONST * sqrt_scaled * scale) / S   [to shift decimal point correctly]
        Then format into string with scale_exp digits after decimal point.
        """
        # numerator = 426880 * sqrt(10005) * 10^scale_exp
        numerator = self.C_CONST * self.sqrt_10005_scaled
        # Now divide by S_num/S_den => numerator * S_den / S_num
        num = numerator * self.S_den
        den = self.S_num

        # Protect against division by zero
        if den == 0:
            raise ZeroDivisionError("Chudnovsky S denominator is zero")

        # Compute scaled pi integer with scale accounted by sqrt scale
        # sqrt already scaled by 10^scale_exp, so pi = num/den with decimal point at scale_exp places.
        pi_int = num // den
        # Rounding: check remainder to round last digit
        rem = num % den
        if rem * 2 >= den:
            pi_int += 1

        # Convert to string with decimal point after first digit '3'
        s = str(pi_int)
        if len(s) <= self.scale_exp:
            s = "0" * (self.scale_exp + 1 - len(s)) + s
        integer_part = s[:-self.scale_exp] if self.scale_exp > 0 else s
        fractional_part = s[-self.scale_exp:] if self.scale_exp > 0 else ""

        return f"{integer_part}.{fractional_part}"

# ----------------------------
# GitHub Uploader
# ----------------------------

class GitHubUploader:
    def __init__(self, repo: str, branch: str = "main", token_env: str = "GITHUB_TOKEN"):
        """
        repo: "owner/repo" e.g., "quantumsoldship/metapi"
        """
        self.repo = repo
        self.branch = branch
        self.token_env = token_env
        self.base_url = "https://api.github.com"

        token = os.getenv(token_env, "").strip()
        if not token:
            self.token = None
        else:
            self.token = token

    def _headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get_file_sha(self, path: str) -> Optional[str]:
        url = f"{self.base_url}/repos/{self.repo}/contents/{path}"
        params = {"ref": self.branch}
        r = requests.get(url, headers=self._headers(), params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data.get("sha")
        return None

    def upload_file(self, local_path: str, repo_path: str, message: str) -> dict:
        """
        Create or update a file in the repository.
        """
        if not os.path.isfile(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        with open(local_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("utf-8")

        sha = self._get_file_sha(repo_path)
        payload = {
            "message": message,
            "content": content_b64,
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha

        url = f"{self.base_url}/repos/{self.repo}/contents/{repo_path}"
        r = requests.put(url, headers=self._headers(), data=json.dumps(payload), timeout=60)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"GitHub upload failed: {r.status_code} {r.text}")
        return r.json()

# ----------------------------
# Pi Finder Orchestrator
# ----------------------------

class PiFinder:
    def __init__(
        self,
        output_file: str = "pi_digits.txt",
        performance_file: str = "performance.json",
        repo: str = "quantumsoldship/metapi",
        branch: str = "main",
        token_env: str = "GITHUB_TOKEN",
        upload_every_batches: int = 1,
    ):
        self.output_file = output_file
        self.performance_file = performance_file
        self.repo = repo
        self.branch = branch
        self.token_env = token_env
        self.upload_every_batches = max(1, int(upload_every_batches))

        self._lock = threading.Lock()
        self._running = False
        self._start_time = None

        self._stats = {
            "calculations_per_second": 0.0,
            "total_calculations": 0,
            "memory_usage_mb": 0.0,
            "cpu_percent": 0.0,
            "current_batch": 0,
            "terms_used": 0,
            "target_digits": 0,
            "branch": branch,
            "repo": repo,
            "started_at": None,
            "last_update": None,
        }

        # Prepare files
        open(self.output_file, "w").close()
        open(self.performance_file, "w").close()

    def _log(self, msg: str) -> None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}", flush=True)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def _write_digits(self, pi_str: str, digits_to_keep: int) -> None:
        """
        Write "3.<digits>" trimmed to digits_to_keep after decimal.
        """
        if "." in pi_str:
            integer_part, frac = pi_str.split(".", 1)
        else:
            integer_part, frac = pi_str[0], pi_str[1:]
        frac = frac[:digits_to_keep]
        with open(self.output_file, "w") as f:
            f.write(f"{integer_part}.{frac}")

    def _write_performance(self) -> None:
        with open(self.performance_file, "w") as f:
            json.dump(self._stats, f, indent=2)

    def _upload_artifacts(self) -> None:
        if not os.getenv(self.token_env):
            self._log(f"Skipping upload: token environment variable {self.token_env} is not set")
            return
        uploader = GitHubUploader(self.repo, self.branch, self.token_env)

        commit_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        message = f"Update Pi digits and performance at {commit_time}"

        # Choose repo paths under artifacts/
        repo_digits_path = "artifacts/pi_digits.txt"
        repo_perf_path = "artifacts/performance.json"

        self._log("Uploading artifacts to GitHub...")
        uploader.upload_file(self.output_file, repo_digits_path, message)
        uploader.upload_file(self.performance_file, repo_perf_path, message)
        self._log("Upload complete")

    def compute(
        self,
        max_digits: int = 100000,
        batch_size: int = 5000,
        upload: bool = False,
    ) -> None:
        """
        Incrementally compute Pi digits and optionally upload after each batch.
        """
        with self._lock:
            if self._running:
                raise RuntimeError("Computation already running")
            self._running = True
            self._start_time = time.time()
            self._stats["started_at"] = datetime.utcnow().isoformat() + "Z"
            self._stats["target_digits"] = int(max_digits)

        try:
            self._log(f"Starting computation: target_digits={max_digits}, batch_size={batch_size}, upload={upload}, repo={self.repo}, branch={self.branch}")

            engine = ChudnovskyIncremental(max_digits)
            total_digits_written = 0
            batch_index = 0

            # Initial write to avoid empty file
            self._write_digits("3.", 0)
            self._write_performance()

            while self.running and total_digits_written < max_digits:
                batch_index += 1
                t0 = time.time()

                # Determine terms needed for this batch target
                target_digits_after_batch = min(max_digits, total_digits_written + batch_size)
                target_terms = engine.terms_needed_for_digits(target_digits_after_batch)

                added_terms = engine.add_terms_until(target_terms)
                pi_str = engine.compute_pi_string()

                # Write only up to target_digits_after_batch
                self._write_digits(pi_str, target_digits_after_batch)
                total_digits_written = target_digits_after_batch

                elapsed = time.time() - self._start_time
                batch_elapsed = max(1e-9, time.time() - t0)

                # Update stats
                proc = psutil.Process(os.getpid())
                mem_mb = proc.memory_info().rss / (1024 * 1024)
                cpu_pct = psutil.cpu_percent(interval=None)

                with self._lock:
                    self._stats.update({
                        "total_calculations": total_digits_written,
                        "calculations_per_second": (total_digits_written / elapsed) if elapsed > 0 else 0.0,
                        "memory_usage_mb": mem_mb,
                        "cpu_percent": cpu_pct,
                        "current_batch": batch_index,
                        "terms_used": engine.k,
                        "last_update": datetime.utcnow().isoformat() + "Z",
                        "last_batch_seconds": batch_elapsed,
                        "batch_added_terms": added_terms,
                    })

                self._write_performance()

                self._log(
                    f"Batch {batch_index}: digits={total_digits_written}/{max_digits} "
                    f"(+{min(batch_size, max_digits - (total_digits_written - min(batch_size, total_digits_written)))} this batch), "
                    f"terms={engine.k}, batch_time={batch_elapsed:.2f}s, rate={self._stats['calculations_per_second']:.2f}/s"
                )

                if upload and (batch_index % self.upload_every_batches == 0):
                    try:
                        self._upload_artifacts()
                    except Exception as e:
                        self._log(f"Upload failed: {e}")

            self._log(f"Completed: total_digits={total_digits_written}, time={time.time() - self._start_time:.2f}s")

        finally:
            with self._lock:
                self._running = False

    def stop(self) -> None:
        with self._lock:
            self._running = False
        self._log("Stop requested")

    def get_digits(self) -> str:
        if not os.path.exists(self.output_file):
            return "3."
        with open(self.output_file, "r") as f:
            return f.read().strip()

    def get_stats(self) -> dict:
        return dict(self._stats)

# ----------------------------
# Flask API (optional)
# ----------------------------

app = Flask(__name__)
_finder: Optional[PiFinder] = None
_thread: Optional[threading.Thread] = None

@app.route("/api/start")
def api_start():
    global _finder, _thread
    if _finder and _finder.running:
        return jsonify({"error": "Computation already running"}), 400

    digits = int(request.args.get("digits", 100000))
    batch = int(request.args.get("batch", 5000))
    branch = request.args.get("branch", "main")
    upload_every = int(request.args.get("upload_every", 1))
    upload = request.args.get("upload", "true").lower() in ("1", "true", "yes")

    _finder = PiFinder(
        output_file="pi_digits.txt",
        performance_file="performance.json",
        repo="quantumsoldship/metapi",
        branch=branch,
        token_env="GITHUB_TOKEN",
        upload_every_batches=upload_every,
    )

    _thread = threading.Thread(target=_finder.compute, kwargs={
        "max_digits": digits,
        "batch_size": batch,
        "upload": upload
    }, daemon=True)
    _thread.start()

    return jsonify({
        "status": "started",
        "target_digits": digits,
        "batch_size": batch,
        "branch": branch,
        "upload": upload,
        "upload_every_batches": upload_every
    })

@app.route("/api/performance")
def api_performance():
    if not _finder:
        return jsonify({"error": "No computation running"}), 400
    stats = _finder.get_stats()
    stats["running"] = _finder.running
    return jsonify(stats)

@app.route("/api/status")
def api_status():
    if not _finder:
        return jsonify({"running": False})
    return jsonify({"running": _finder.running})

@app.route("/api/digits")
def api_digits():
    if not _finder:
        return jsonify({"digits": "3."})
    return jsonify({"digits": _finder.get_digits()})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if not _finder:
        return jsonify({"error": "No computation running"}), 400
    try:
        _finder._upload_artifacts()
        return jsonify({"status": "uploaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Pi Finder with GitHub uploader")
    parser.add_argument("--digits", type=int, default=100000, help="Total digits to compute")
    parser.add_argument("--batch", type=int, default=5000, help="Batch size for progress and upload")
    parser.add_argument("--repo", type=str, default="quantumsoldship/metapi", help="GitHub repository owner/name")
    parser.add_argument("--branch", type=str, default="main", help="GitHub branch to commit to")
    parser.add_argument("--upload", action="store_true", help="Upload artifacts after each batch")
    parser.add_argument("--upload-every", type=int, default=1, help="Upload every N batches")
    parser.add_argument("--token-env", type=str, default="GITHUB_TOKEN", help="Environment variable containing GitHub token")
    parser.add_argument("--output", type=str, default="pi_digits.txt", help="Output digits file")
    parser.add_argument("--perf", type=str, default="performance.json", help="Performance json file")
    parser.add_argument("--server", action="store_true", help="Run Flask API server instead of direct computation")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=5000, help="Flask port")

    args = parser.parse_args()

    if args.server:
        print("Starting Pi Finder API server")
        print("Endpoints:")
        print("  GET  /api/start?digits=N&batch=M&branch=BRANCH&upload=true&upload_every=BATCHES")
        print("  GET  /api/performance")
        print("  GET  /api/status")
        print("  GET  /api/digits")
        print("  POST /api/upload")
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
        return

    finder = PiFinder(
        output_file=args.output,
        performance_file=args.perf,
        repo=args.repo,
        branch=args.branch,
        token_env=args.token_env,
        upload_every_batches=args.upload_every,
    )
    finder.compute(
        max_digits=args.digits,
        batch_size=args.batch,
        upload=args.upload,
    )

if __name__ == "__main__":
    main()