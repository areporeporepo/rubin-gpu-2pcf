"""Query DP1 from ANYWHERE (Colab/laptop) using an RSP token with read:tap scope.

This lets you extract the catalog on the SAME machine that has the GPU, instead
of the RSP notebook -> download -> upload dance.

SECURITY: never hardcode or commit the token. Export it first:
    export RSP_TOKEN=gt-xxxxxxxx...        # the secret from data.lsst.cloud/settings/tokens
Then:
    python examples/tap_query_external.py            # -> dp1_ECDFS.csv

(Verify the TAP endpoint URL against the DP1 "api" tutorials if it changes.)
"""
from __future__ import annotations

import os

# Data release table to query. Set RELEASE="dp2" the day DP2 lands (this summer)
# — DP2 exposes dp2.Object over a ~3000 deg² footprint (DDFs + contiguous region),
# so widen RADIUS_DEG / pick a region inside the DP2 footprint. The rest is identical.
RELEASE = os.environ.get("RUBIN_RELEASE", "dp1")  # "dp1" | "dp2"

# real DP1 extragalactic fields (for DP2, choose a center in its footprint)
FIELDS = {"ECDFS": (53.13, -28.10), "EDFS": (59.10, -48.73), "SV_95_-25": (95.00, -25.00)}
FIELD = "ECDFS"
RA0, DEC0 = FIELDS[FIELD]
RADIUS_DEG = 1.0
TAP_URL = "https://data.lsst.cloud/api/tap"

ADQL = f"""
SELECT coord_ra, coord_dec
FROM {RELEASE}.Object
WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec),
              CIRCLE('ICRS', {RA0}, {DEC0}, {RADIUS_DEG})) = 1
"""
# (DP1's dp1.Object has no detect_isPrimary column; add quality cuts later after
#  checking real column names via:
#    SELECT column_name FROM tap_schema.columns WHERE table_name = 'dp1.Object')


def _load_dotenv(path=".env"):
    """Minimal, dependency-free .env loader (KEY=VALUE per line)."""
    import pathlib

    p = pathlib.Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    _load_dotenv()
    token = os.environ.get("RSP_TOKEN")
    if not token:
        raise SystemExit(
            "No RSP_TOKEN. Create a .env file (gitignored) with one line:\n"
            "    RSP_TOKEN=<your-token>\n"
            "or run: export RSP_TOKEN=<your-token>"
        )

    import pyvo
    import requests

    def _scrub(msg):  # never let the token appear in any output
        return str(msg).replace(token, "<REDACTED>") if token else str(msg)

    try:
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {token}"
        tap = pyvo.dal.TAPService(TAP_URL, session=session)
        print(f"Querying {RELEASE}.Object around {FIELD} ({RA0}, {DEC0}) r={RADIUS_DEG} ...")
        df = tap.search(ADQL).to_table().to_pandas()
    except Exception as exc:
        raise SystemExit(f"TAP query failed: {_scrub(exc)[:600]}")

    df = df.rename(columns={"coord_ra": "ra", "coord_dec": "dec"})
    out = f"{RELEASE}_{FIELD}.csv"
    df[["ra", "dec"]].to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} objects)")
    print(f"next: python examples/run_on_dp1.py {out} {RA0} {DEC0} {RADIUS_DEG}")


if __name__ == "__main__":
    main()
