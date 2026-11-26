"""Classify ONE URL and explain the decision.

    python src/predict.py "http://paypal.secure-login.xyz/verify"
    python src/predict.py "http://example.com" --live          # try live DNS/WHOIS (needs dnspython + python-whois)
    python src/predict.py "http://x.com" --domain-age-days 12 --mx-count 0   # supply known values manually
Unknown DNS/WHOIS values are imputed with training medians by the model pipeline.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from explain import occlusion_attribution
from features import extract_lexical, lookup_dns_whois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--model", default="models/best_model.joblib")
    ap.add_argument("--meta", default="results/meta.json")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--domain-age-days", type=float)
    ap.add_argument("--reg-length-days", type=float)
    ap.add_argument("--mx-count", type=float)
    ap.add_argument("--a-count", type=float)
    ap.add_argument("--ns-count", type=float)
    ap.add_argument("--ttl", type=float)
    ap.add_argument("--whois-privacy", type=float)
    a = ap.parse_args()

    model, meta = joblib.load(a.model), json.loads(Path(a.meta).read_text())
    row = extract_lexical(a.url)
    ext = lookup_dns_whois(a.url) if a.live else {}
    manual = {"domain_age_days": a.domain_age_days, "registration_length_days": a.reg_length_days, "dns_mx_count": a.mx_count,
              "dns_a_count": a.a_count, "dns_ns_count": a.ns_count, "dns_ttl": a.ttl, "whois_privacy": a.whois_privacy}
    row.update({k: v for k, v in ext.items()})
    row.update({k: v for k, v in manual.items() if v is not None})
    s = pd.Series({f: row.get(f, np.nan) for f in meta["features"]}, dtype=float)
    base, contrib = occlusion_attribution(model, s, meta["train_medians"])
    print(f"\nURL: {a.url}\nP(phishing) = {base:.3f}  ->  {'PHISHING' if base >= 0.5 else 'LEGITIMATE'}")
    print("\nTop factors (effect on log-odds of phishing; + = towards phishing):")
    for f, c in contrib.head(8).items():
        v = s[f]
        print(f"  {f:28s} value={'n/a' if np.isnan(v) else round(v, 3):>8}  effect={c:+.3f}  [{meta['groups'][f]}]")


if __name__ == "__main__":
    main()
