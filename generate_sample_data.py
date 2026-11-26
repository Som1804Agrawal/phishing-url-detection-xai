"""Generate a SYNTHETIC sample dataset so the whole pipeline can be tested offline.

!!  The generated data is ARTIFICIAL. It is NOT the Kaggle dataset and results obtained
!!  on it must never be reported as real experimental results.

Columns: url, label (1 = phishing, 0 = legitimate), DNS columns, WHOIS columns.
"""
import argparse
import random

import numpy as np
import pandas as pd

BRANDS = ["paypal", "amazon", "google", "microsoft", "apple", "netflix", "facebook", "sbi", "hdfc", "icici", "dropbox", "linkedin"]
WORDS = ["news", "shop", "blog", "travel", "health", "tech", "food", "sports", "music", "learn", "photo", "garden",
         "bank", "cloud", "mail", "movies", "books", "games", "jobs", "city", "river", "stone", "green", "blue"]
LEGIT_TLDS = ["com", "org", "net", "edu", "in", "co.uk", "gov", "io"]
BAD_TLDS = ["xyz", "top", "tk", "ml", "ga", "click", "info", "icu", "live", "work", "com", "net"]
KEYS = ["login", "verify", "secure", "account", "update", "confirm", "signin", "billing", "wallet", "password"]
SHORT = ["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy"]


def rs(rng, n):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(n))


def legit_url(rng):
    tld = rng.choice(BAD_TLDS) if rng.random() < 0.10 else rng.choice(LEGIT_TLDS)
    host = f"{rng.choice(WORDS)}{rng.choice(['', '-' + rng.choice(WORDS)])}{rng.choice(['', str(rng.randint(1, 99))])}.{tld}"
    if rng.random() < 0.6:
        host = "www." + host
    elif rng.random() < 0.15:
        host = rng.choice(["blog", "docs", "support", "login", "mail"]) + "." + host
    path = "/".join(rng.choice(WORDS) for _ in range(rng.randint(0, 3)))
    url = f"{'https' if rng.random() < 0.92 else 'http'}://{host}/{path}"
    if rng.random() < 0.28:
        url += f"{rs(rng, rng.randint(6, 14))}"  # ids / hashes in legit URLs
    if rng.random() < 0.2:
        url += f"?id={rng.randint(1, 9999)}&ref={rng.choice(WORDS)}"
    if rng.random() < 0.12:
        url += "/" + rng.choice(KEYS)
    if rng.random() < 0.35:
        url = url.rstrip("/") + rng.choice([".html", ".php", ".aspx"])  # legit pages can contain keywords too
    return url


def phish_url(rng):
    style = rng.random()
    brand, key = rng.choice(BRANDS), rng.choice(KEYS)
    if style < 0.12:  # IP host
        host = ".".join(str(rng.randint(1, 254)) for _ in range(4))
        return f"http://{host}/{key}/{rs(rng, rng.randint(4, 10))}.php"
    if style < 0.32:  # brand in subdomain
        return f"{'http' if rng.random() < 0.6 else 'https'}://{brand}.{key}-{rs(rng, rng.randint(3, 8))}.{rng.choice(BAD_TLDS)}/{key}"
    if style < 0.5:  # long hyphenated
        return f"http://{brand}-{key}-{rng.choice(KEYS)}-{rs(rng, 5)}.{rng.choice(BAD_TLDS)}/{rs(rng, 8)}/index.html"
    if style < 0.58:  # @ trick
        return f"http://{rng.choice(WORDS)}.com@{rs(rng, 6)}.{rng.choice(BAD_TLDS)}/{key}"
    if style < 0.66:  # shortener
        return f"https://{rng.choice(SHORT)}/{rs(rng, 7)}"
    if style < 0.74:  # random-looking domain, deep path
        return f"{'http' if rng.random() < 0.7 else 'https'}://{rs(rng, rng.randint(8, 16))}.{rng.choice(BAD_TLDS)}/{key}/{rs(rng, 6)}?session={rs(rng, 20)}&user={rng.randint(100, 99999)}"
    # looks clean (compromised / look-alike site using ordinary words)
    return f"{'https' if rng.random() < 0.6 else 'http'}://www.{rng.choice(WORDS)}{rng.choice(WORDS)}.{rng.choice(LEGIT_TLDS)}/{rng.choice(WORDS)}/{rng.choice(['', rs(rng, 6) + '/'])}{key if rng.random() < 0.5 else 'index'}{rng.choice(['.html', '.php', '', '/'])}"


def dns_whois(np_rng, phishing: bool):
    compromised = phishing and np_rng.random() < 0.25  # aged hijacked domain -> class overlap
    bad = phishing and not compromised
    age = float(np.clip(np_rng.lognormal(mean=4.3 if bad else 7.0, sigma=1.4 if bad else 1.4), 0, 9000))
    if bad:
        reg = float(np_rng.choice([365, 730], p=[.9, .1]))
    else:
        reg = float(np_rng.choice([365, 730, 1825, 3650], p=[.5, .2, .15, .15]))
    row = {
        "dns_a_count": int(np_rng.choice([1, 2, 3, 4], p=[.55, .25, .1, .1] if bad else [.35, .3, .2, .15])),
        "dns_ttl": int(np_rng.choice([60, 300, 3600, 14400, 86400], p=[.2, .4, .25, .1, .05] if bad else [.03, .2, .4, .2, .17])),
        "dns_mx_count": int(np_rng.choice([0, 1, 2, 5], p=[.6, .22, .12, .06] if bad else [.2, .3, .28, .22])),
        "dns_ns_count": int(np_rng.choice([1, 2, 3, 4], p=[.1, .7, .12, .08] if bad else [.05, .55, .2, .2])),
        "domain_age_days": round(age),
        "registration_length_days": reg,
        "whois_privacy": int(np_rng.random() < (0.65 if bad else 0.3)),
    }
    miss = 0.12 if phishing else 0.05  # failed lookups (taken-down domains etc.)
    for k in list(row):
        if np_rng.random() < miss / 2:
            row[k] = np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--phish-ratio", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/sample_phishing_synthetic.csv")
    a = ap.parse_args()
    rng, np_rng = random.Random(a.seed), np.random.default_rng(a.seed)
    rows = []
    for _ in range(a.n):
        ph = rng.random() < a.phish_ratio
        rows.append({"url": phish_url(rng) if ph else legit_url(rng), "label": int(ph), **dns_whois(np_rng, ph)})
    df = pd.DataFrame(rows).drop_duplicates("url").sample(frac=1, random_state=a.seed)
    df.to_csv(a.out, index=False)
    print(f"Wrote {len(df)} rows to {a.out} (phishing share: {df.label.mean():.2%}) - SYNTHETIC DATA")


if __name__ == "__main__":
    main()
