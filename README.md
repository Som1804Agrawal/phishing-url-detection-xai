# Data folder

## `sample_phishing_synthetic.csv`  (SYNTHETIC - for testing only)
~8,000 artificially generated rows created by `src/generate_sample_data.py`.
Columns: `url`, `label` (1 = phishing, 0 = legitimate), 4 DNS columns, 3 WHOIS columns.
**It is NOT the Kaggle dataset. Never report results from this file as real experimental results.**

## Real dataset (Kaggle Phishing URL Features Dataset, 579,920 records, 74 features)
1. Download `final_dataset.csv` from https://www.kaggle.com/datasets/elifzelik/phishing-url-features-dataset and place it here as `data/final_dataset.csv`.
2. Check the column names (`head -1 data/phishing_urls.csv`) and run:
   ```
   python src/train_evaluate.py --data data/final_dataset.csv --url-col none --label-col label --positive-label 1 --groups-json data/groups.json
   ```
   * If the file has a URL column, lexical features are computed from it; DNS/WHOIS columns are used when the file contains
     columns named `dns_a_count, dns_ttl, dns_mx_count, dns_ns_count, domain_age_days, registration_length_days, whois_privacy`
     (rename your columns to these, or edit `DNS_FEATURES` / `WHOIS_FEATURES` in `src/features.py`).
   * If the file has no URL column, all numeric columns except the label are used as features.
   * Live DNS/WHOIS look-ups for ~580k URLs are impractical; use `--sample 50000` or the dataset's own columns.
3. Cite the dataset (author, title, URL, access date) in the paper and report.

`groups.json` maps the DNS and WHOIS columns of final_dataset.csv; every other column is treated as Lexical.
