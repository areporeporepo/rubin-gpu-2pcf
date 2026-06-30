"""Safely inspect the RSP token's SHAPE (never prints the value).

Rubin/Gafaelfawr tokens look like:  gt-<22chars>.<22chars>  (a 'gt-' prefix and a '.').
A 401 often means the value in .env is incomplete, the wrong (revoked) token, or
missing the read:tap scope.

    .venv/bin/python examples/check_token.py
"""
import os
import pathlib


def _load_dotenv(path=".env"):
    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()
t = os.environ.get("RSP_TOKEN", "")
print("present:           ", bool(t))
print("length:            ", len(t))
print("starts with 'gt-': ", t.startswith("gt-"))
print("contains '.':      ", "." in t)
print("has whitespace:    ", any(c.isspace() for c in t))
print("looks like a full RSP token:", t.startswith("gt-") and "." in t and not any(c.isspace() for c in t))
