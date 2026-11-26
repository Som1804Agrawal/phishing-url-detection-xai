"""Feature extraction for phishing URL detection.

Three feature families are used (see the paper/report):
  * Lexical  - computed directly from the URL string (always available)
  * DNS      - properties of the domain's DNS records
  * WHOIS    - domain registration properties

Lexical features are extracted here. DNS/WHOIS features are either
  (a) already present as columns in the dataset, or
  (b) fetched live with `lookup_dns_whois()` (needs the optional packages
      `dnspython` and `python-whois`; slow, so use it for single URLs or small samples).
"""
import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd

SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "update", "bank", "confirm",
                    "signin", "password", "paypal", "wallet", "billing", "support", "free", "bonus"]
SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "cutt.ly", "rb.gy"]
SUSPICIOUS_TLDS = {"xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "info", "buzz", "icu", "live"}
IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

LEXICAL_FEATURES = [
    "url_length", "hostname_length", "path_length", "query_length", "num_dots", "num_hyphens",
    "num_underscores", "num_digits", "digit_ratio", "num_special_chars", "has_ip_host", "has_at_symbol",
    "num_subdomains", "uses_https", "has_port", "suspicious_tld", "num_suspicious_words",
    "url_entropy", "num_query_params", "has_double_slash_in_path", "is_shortener",
]
DNS_FEATURES = ["dns_a_count", "dns_ttl", "dns_mx_count", "dns_ns_count"]
WHOIS_FEATURES = ["domain_age_days", "registration_length_days", "whois_privacy"]
ALL_FEATURES = LEXICAL_FEATURES + DNS_FEATURES + WHOIS_FEATURES

FEATURE_GROUP = {**{f: "Lexical" for f in LEXICAL_FEATURES},
                 **{f: "DNS" for f in DNS_FEATURES},
                 **{f: "WHOIS" for f in WHOIS_FEATURES}}


_WHOIS_TOKENS = {"whois", "age", "creation", "created", "expiration", "expiry", "expire", "expires", "registrar",
                 "registration", "registrant", "registered", "dnssec"}
_DNS_TOKENS = {"dns", "ttl", "mx", "ns", "nameserver", "nameservers", "soa", "txt", "cname", "aaaa", "resolve", "resolved"}


def infer_group(name: str) -> str:
    """Guess the feature family of a dataset column from its NAME (used when the CSV ships ready-made features).

    Override with --groups-json if your column names do not follow these keywords."""
    tokens = set(re.split(r"[^a-z0-9]+", str(name).lower()))
    if tokens & _WHOIS_TOKENS:
        return "WHOIS"
    if tokens & _DNS_TOKENS:
        return "DNS"
    return "Lexical"


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _port(p):
    try:
        return p.port
    except ValueError:
        return None


def extract_lexical(url: str) -> dict:
    """Return the lexical feature dict for one URL string."""
    url = str(url).strip()
    full = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else "http://" + url
    p = urlparse(full)
    host = (p.hostname or "").lower()
    path, query = p.path or "", p.query or ""
    parts = host.split(".") if host else []
    tld = parts[-1] if parts else ""
    is_ip = bool(IP_RE.match(host))
    n_sub = 0 if is_ip else max(len(parts) - 2, 0)
    low = url.lower()
    n_digits = sum(ch.isdigit() for ch in url)
    return {
        "url_length": len(url),
        "hostname_length": len(host),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": n_digits,
        "digit_ratio": n_digits / max(len(url), 1),
        "num_special_chars": sum(not ch.isalnum() for ch in url),
        "has_ip_host": int(is_ip),
        "has_at_symbol": int("@" in url),
        "num_subdomains": n_sub,
        "uses_https": int(p.scheme == "https"),
        "has_port": int(_port(p) is not None),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "num_suspicious_words": sum(w in low for w in SUSPICIOUS_WORDS),
        "url_entropy": _entropy(url),
        "num_query_params": len([q for q in query.split("&") if q]),
        "has_double_slash_in_path": int("//" in path),
        "is_shortener": int(any(host == s or host.endswith("." + s) for s in SHORTENERS)),
    }


def build_lexical_frame(urls) -> pd.DataFrame:
    return pd.DataFrame([extract_lexical(u) for u in urls])


def lookup_dns_whois(url: str, timeout: float = 3.0) -> dict:
    """Best-effort LIVE DNS + WHOIS lookup. Missing data -> NaN (handled by the imputer).

    Requires: pip install dnspython python-whois.  Slow (network); not meant for 500k rows.
    """
    out = {k: np.nan for k in DNS_FEATURES + WHOIS_FEATURES}
    host = urlparse(url if "://" in url else "http://" + url).hostname or ""
    if not host or IP_RE.match(host):
        return out
    try:
        import dns.resolver
        res = dns.resolver.Resolver()
        res.lifetime = timeout
        a = res.resolve(host, "A")
        out["dns_a_count"] = len(a)
        out["dns_ttl"] = a.rrset.ttl
        for rtype, key in (("MX", "dns_mx_count"), ("NS", "dns_ns_count")):
            try:
                out[key] = len(res.resolve(host, rtype))
            except Exception:
                out[key] = 0
    except Exception:
        pass
    try:
        import whois
        from datetime import datetime
        w = whois.whois(host)
        cre, exp = w.creation_date, w.expiration_date
        cre = cre[0] if isinstance(cre, list) else cre
        exp = exp[0] if isinstance(exp, list) else exp
        now = datetime.now()
        if cre:
            out["domain_age_days"] = max((now - cre.replace(tzinfo=None)).days, 0)
        if cre and exp:
            out["registration_length_days"] = max((exp.replace(tzinfo=None) - cre.replace(tzinfo=None)).days, 0)
        txt = str(w.text).lower()
        out["whois_privacy"] = int("privacy" in txt or "redacted" in txt)
    except Exception:
        pass
    return out
