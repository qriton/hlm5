"""Download the CounterFact dataset used by run_cert_counterfact_gpt2xl.py and
run_editors_counterfact.py.

Source: https://rome.baulab.info/data/dsets/counterfact.json (the official
CounterFact release from Meng et al. 2022, "Locating and Editing Factual
Associations in GPT", hosted by the ROME project). EXPECTED_SHA256 below was
computed from the reference file used for the release artifacts on 2026-07-08.

Run: python scripts/fetch_counterfact.py
Out: scripts/data/counterfact.json (NOT committed; see .gitignore)
"""
import hashlib
import urllib.request
from pathlib import Path

URL = "https://rome.baulab.info/data/dsets/counterfact.json"
OUT = Path(__file__).resolve().parent / "data" / "counterfact.json"

# sha256 of the reference CounterFact copy used for the paper, computed 2026-07-08.
EXPECTED_SHA256 = "d017056125178a13728594e66a801357a8db9ed7973a7425554bb4271de9fc6f"

CHUNK = 1 << 20  # 1 MiB


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} -> {OUT}")
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(URL) as resp, open(OUT, "wb") as f:
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
            h.update(chunk)
            size += len(chunk)

    digest = h.hexdigest()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"sha256 mismatch: got {digest}, expected {EXPECTED_SHA256}. "
            f"The downloaded file at {OUT} may be corrupt, truncated, or the "
            "upstream dataset changed; not deleting it automatically -- "
            "inspect before reusing.")

    print(f"OK: {size:,} bytes, sha256={digest}")
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
