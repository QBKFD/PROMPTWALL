# Evaluation results

Generated 2026-11-19T19:01:26Z from commit `5ae3181ea961dfeda7b804fb8e046bc7564f1f5e-dirty`. 95 % CIs: percentile bootstrap, 10000 resamples, seed 0.

## Prompt-injection classifier (TF-IDF + logistic regression)

Trained on deepset/prompt-injections train (546 rows). Selected by 5-fold CV on train: features=word+char, C=100, threshold=0.45 (CV F1 0.9177).

Precision, recall and FPR show Wilson 95 % intervals (informative at zero counts); F1 shows the bootstrap 95 % interval.

| Test set | n | Precision | Recall | F1 [95 % CI] | FP | FN | FPR |
|---|---|---|---|---|---|---|---|
| deepset_test (prompt injection vs benign (in-distribution)) | 116 | 1.000 [0.932, 1.000] | 0.883 [0.778, 0.942] | 0.938 [0.887, 0.977] | 0 | 7 | 0.000 [0.000, 0.064] |
| jackhhao_jailbreak_test (jailbreak vs benign (out-of-distribution)) | 398 | 0.357 [0.311, 0.406] | 1.000 [0.973, 1.000] | 0.526 [0.475, 0.576] | 250 | 0 | 0.965 [0.935, 0.982] |
| lakera_gandalf_test (injections only -> recall only (out-of-distribution)) | 112 | – – | 0.982 [0.937, 0.995] | – – | 0 | 2 | – – |

Confusion matrices (tp / fp / fn / tn): deepset_test: 53 / 0 / 7 / 56; jackhhao_jailbreak_test: 139 / 250 / 0 / 9; lakera_gandalf_test: 110 / 0 / 2 / 0

Rows removed because they duplicate a train row: deepset_test: {'exact': 0, 'near_jaccard>=0.9': 0}; jackhhao_jailbreak_test: {'exact': 0, 'near_jaccard>=0.9': 2}; lakera_gandalf_test: {'exact': 0, 'near_jaccard>=0.9': 0}

## PII rules (gretel_pii test split, n=5594 documents)

Span level, overlap match (a predicted span counts if it overlaps a gold span of the same type).

| Entity | Gold spans | Precision | Recall | F1 [95 % CI] | FP | FN | Exact-match F1 |
|---|---|---|---|---|---|---|---|
| email | 1422 | 0.968 | 0.961 | 0.964 [0.953, 0.974] | 45 | 56 | 0.959 |
| iban | 203 | 0.974 | 0.768 | 0.859 [0.813, 0.900] | 4 | 47 | 0.844 |
| phone | 946 | 0.686 | 0.928 | 0.789 [0.764, 0.812] | 401 | 68 | 0.783 |

Document level (contains any of email/IBAN/phone): precision 0.884, recall 0.932, F1 0.908 [0.896, 0.919], FP 149, FN 83 (n=5594).

False positives by the gold label they overlap ('(unlabelled text)' = no gold label):

- email: {'name': 27, 'company': 9, 'user_name': 7, 'first_name': 2}
- iban: {'(unlabelled text)': 4}
- phone: {'(unlabelled text)': 318, 'driver_license_number': 32, 'street_address': 22, 'date': 15, 'credit_card_number': 5, 'bank_routing_number': 2, 'passport_number': 2, 'iban': 2, 'ssn': 1, 'ipv4': 1, 'credit_card_security_code': 1}

## Secrets rules

| Evaluation | Precision | Recall | F1 [95 % CI] | FP | FN |
|---|---|---|---|---|---|
| Gretel test, gold = api_key spans (90 gold spans) | 0.663 | 0.656 | 0.659 [0.558, 0.747] | 30 | 31 |
| Gretel test, gold = api_key or password (191 gold spans) | 0.910 | 0.424 | 0.579 [0.493, 0.659] | 8 | 110 |
| Synthetic, self-made (n=880; sanity check only) | 1.000 | 1.000 | 1.000 | 0 | 0 |
