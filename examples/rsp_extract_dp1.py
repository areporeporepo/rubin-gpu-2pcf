"""STEP 1 — run this INSIDE the Rubin Science Platform (data.lsst.cloud) Notebook
aspect. It pulls the real DP1 Object catalog (RA, Dec) for one extragalactic
field via TAP/ADQL and writes a small CSV you can download.

The RSP is CPU-only, so we only EXTRACT here; the GPU w(theta) runs elsewhere
(examples/run_on_dp1.py on Colab/GCP).

No-code alternative: paste the ADQL below into the Portal's ADQL query box
(data.lsst.cloud/portal/app), run it, and "Save" the result table as CSV.
"""

# --- field choices (real DP1 extragalactic fields; good for galaxy clustering) ---
FIELDS = {
    "ECDFS": (53.13, -28.10),
    "EDFS": (59.10, -48.73),
    "SV_95_-25": (95.00, -25.00),
}
FIELD = "ECDFS"
RA0, DEC0 = FIELDS[FIELD]
RADIUS_DEG = 1.0

ADQL = f"""
SELECT coord_ra, coord_dec, detect_isPrimary
FROM dp1.Object
WHERE CONTAINS(POINT('ICRS', coord_ra, coord_dec),
              CIRCLE('ICRS', {RA0}, {DEC0}, {RADIUS_DEG})) = 1
  AND detect_isPrimary = 1
"""
# Optional refinement (verify exact column names in the DP1 schema browser /
#   SELECT column_name FROM tap_schema.columns WHERE table_name='dp1.Object'):
#   AND i_psfFlux / i_psfFluxErr > 5        -- i-band SNR > 5
#   AND i_extendedness > 0.5                -- galaxies only

if __name__ == "__main__":
    # In the RSP, the authenticated TAP service is provided by lsst.rsp:
    from lsst.rsp import get_tap_service

    try:
        service = get_tap_service("tap")
    except TypeError:  # older RSP signature
        service = get_tap_service()

    print(f"Querying dp1.Object around {FIELD} ({RA0}, {DEC0}) r={RADIUS_DEG} deg ...")
    job = service.search(ADQL)
    tbl = job.to_table()
    df = tbl.to_pandas().rename(columns={"coord_ra": "ra", "coord_dec": "dec"})
    out = f"dp1_{FIELD}.csv"
    df[["ra", "dec"]].to_csv(out, index=False)
    print(f"wrote {out}  ({len(df)} objects)  -- download this, then run run_on_dp1.py")
