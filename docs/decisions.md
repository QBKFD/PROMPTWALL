# Decisions log

Decisions are recorded here *before* the results they could be biased by were seen.

## 2026-11-19: Stage 2 (input controls)

Made while the author was away and asked for the conservative option at each fork.

1. **PII rule development used only the Gretel train split** (a 5,000-document
   random sample, seed 13, `eval/dev_pii.py`). The rules were frozen before the test
   split was scored. One later change, the secret keyword boundary (`DB_PASSWORD=`
   was missed), came from a unit test, not from test data.
2. **IBAN checksum on.** On the train sample, mod-97 validation gave span F1 0.86
   vs 0.63 without it. Gretel labels some invalid IBANs as IBANs, so this costs recall
   on this dataset; we keep it because unchecked IBAN-shaped strings are mostly noise.
3. **Phone rule is conservative about precision.** Bare digit runs without separators
   or a `+`/`00` prefix are not treated as phones (too often account numbers). Shape
   rejections are general-purpose (dates, IPv4, decimals, thousands grouping, card
   grouping, US SSN/EIN, Nordic personal numbers, year ranges); a Gretel-specific
   driver-licence shape was deliberately *not* added to avoid fitting the dev data.
4. **Secrets are evaluated two ways on Gretel**: against `api_key` spans (the plan),
   and against `api_key` or `password` spans, because the rules deliberately catch
   `password=...` assignments. Both are reported; neither is picked as "the" number.
5. **Injection classifier selection**: 18 configs (word / char / word+char TF-IDF ×
   C in {0.3, 1, 3, 10, 30, 100}), stratified 5-fold CV on deepset train (no exact
   duplicates inside train, so grouping was not needed), selected by mean F1 at 0.5;
   threshold then tuned on out-of-fold probabilities. Result: word+char, C=100,
   threshold 0.45, CV F1 0.918. C=100 is at the edge of the grid; the top configs
   are within ~0.01 CV F1 of each other.
6. **Duplicates**: test rows that exactly or nearly (char-5-gram Jaccard >= 0.9)
   duplicate a deepset train row are removed before scoring and the count is reported
   (deepset test: 0, jackhhao test: 2 near, gandalf test: 0).
7. **Transformer criterion (pre-registered before any test score was computed):**
   a small multilingual transformer replaces TF-IDF only if its mean 5-fold CV F1 on
   deepset train beats TF-IDF's (0.918) by at least 0.03, measured with the same folds.
   Test-set scores are reported for both models only after this choice is made, and
   the choice is not revisited after seeing them.
8. **Default policy actions**: secrets → block; PII → mask on input and output;
   injection → block; any check error → block (fail closed).

### Recorded after the test sets were scored (no model or threshold changed)

9. **Out-of-distribution false positives.** On jackhhao benign prompts the TF-IDF
   classifier flags 250 of 259 (FPR 0.965); 89 % even among prompts under 100 characters.
   deepset's benign rows are mostly short questions (median 42 chars), while jackhhao's
   benign rows are instruction-style prompts ("You are...", "Write a..."), which
   deepset's labels associate with injections. This is a training-data limitation; fixing
   it means different or additional training data, which is the author's call. The model
   and threshold were **not** changed after seeing this.
10. **Policy action kept at `block` for the classifier** (fail-safe default), with this
    caveat: on instruction-style traffic it would block most benign requests. Switching
    to `log` is a one-line change in config/policy.yaml.
