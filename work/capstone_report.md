# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Muhammad Arham Hassan
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/Arhxmz/Flyrank-Internship-ML
- **Date:** September 19,2026


## 0. Abstract

Which published client pages are likely to decline in organic search traffic, and should be prioritized for editorial refresh? Using February 2026 search performance signals to predict March 2026 click decline on FlyRank's anonymized content warehouse, a gradient-boosted classifier was trained and validated against a transparent visibility-based baseline on a client-grouped, zero-overlap test split. The model reached 0.907 AUC and 63.8% Precision@10%, versus 0.721 AUC and 32.1% Precision@10% for the baseline — nearly doubling the hit rate editors would see in the top of a prioritized queue. The output is a ranked action playbook with reason codes, intended to direct limited editorial review time toward the pages most likely to need it.

## 1. Problem framing

**Decision:** which pages get queued for editorial refresh this cycle, versus left alone.
**Unit of analysis:** one published content page, for one client, aggregated over a month.
**Output:** a ranked score and a three-tier action label (refresh_priority / monitor / no_action) with a reason code.
**Action:** an SEO editor reviews flagged pages, updates outdated facts, improves depth, re-optimizes on-page elements.
**Cost of a wrong call:** a false positive wastes editorial time on a page that didn't need it; a false negative lets real organic traffic loss go unaddressed for another cycle.
**Why ML helps:** dozens of interacting signals (position, clicks, impressions, staleness) don't reduce to a single hand-written threshold, and their relationships likely differ across content types and clients — exactly the kind of pattern a rule-based system alone struggles to capture.

## 2. Data safety

Data source: FlyRank ML Internship warehouse (`FlyRank/internship-warehouse`), an anonymized release of real search performance data. Table used: `fact_content_daily_performance`, joined where needed with client-level identifiers already present in that table. February 2026 was used for features, March 2026 for the outcome — the dataset's final month (June 2026) was never touched, reserved as a sealed test window.

Deliberately excluded: FlyRank's own product-computed decision fields (`health_score`, `priority_score`, `action_type`) — these are outputs of an existing rule-based system, never used as inputs. `trend_direction`/`trend_pct`-equivalent label-derived signals were kept strictly on the label side, never the feature side. `dim_content`'s update-date field was excluded from this model entirely after testing revealed it reflects each page's *current* state rather than a true point-in-time snapshot — using it would have leaked information from after the February feature window.

All identifiers (`client_hash_id`, `content_hash_id`) are salted hash keys, used only for grouping and joins, never as model features. No client names, domains, URLs, or private queries appear anywhere in this repo.

## 3. Baseline

Built in Week 4: `stale_but_visible = is_stale × has_visibility × impressions`, where a page scores above zero only if it's gone 90+ days without an update and still holds real search visibility. Verified first on real bucket tables: staleness confirmed against click volume (2.92 → 0.98 → 0.10 avg clicks across freshness buckets), and CTR-vs-position confirmed against the CTR-fix logic's core assumption (1.06% → 0.09% CTR across position buckets).

For the Feb→March model comparison, staleness wasn't available in a point-in-time-safe form, so the baseline is represented by its visibility component alone — ranking by raw February impressions — scored on the identical held-out test set as the model, for a fair comparison.

## 4. Model / analysis

Method: `HistGradientBoostingClassifier` (scikit-learn) — chosen for handling missing values (sparse GA4 coverage) natively, and as a reasonable, defensible baseline-beater for tabular data.

Features: `avg_position`, `feb_clicks`, `feb_impressions` — all aggregated from February 2026, entirely before the March label window. Staleness was deliberately left out (see Section 2) due to the point-in-time reliability issue discovered during methodology testing.

Label (proxy, one sentence): `is_declining = March total clicks < 80% of February total clicks` — a genuine forward-looking definition, not a within-month split.

## 5. Evaluation

Split: grouped by `client_hash_id` (`GroupShuffleSplit`, 70/30), confirmed to have **zero** client overlap between train and test — no client's pages appear in both.

| Metric | Baseline (visibility-only) | Model | Improvement |
|---|---|---|---|
| AUC | 0.721 | 0.907 | +0.186 |
| Precision@10% | 0.321 | 0.638 | +0.317 |

Base rate of decline in this label definition: 16.2%.

**Error analysis:** reviewing the model's own top-ranked output surfaced a real weakness — many of the highest-confidence "declining" pages had near-zero February clicks (often just 1), meaning a drop from 1 click to 0 technically satisfies the label but isn't meaningful decline. A production version would need a minimum click/impression floor before a page is eligible for the "declining" label at all, the same kind of visibility gate the Week 4 baseline already applied.

## 6. Interpretation

The model substantially outperforms a visibility-only rule, suggesting position and click/impression patterns together carry more signal than raw visibility alone — consistent with the Week 4 signal checks, which confirmed both staleness and CTR-vs-position as real, directionally correct signals. The largest practical gain is in Precision@10%: on the realistic slice of pages an editor with limited bandwidth would actually review, the model's picks are correct roughly twice as often as the baseline's.

A genuine limitation surfaced during data exploration, not modeling: `dim_content`'s update-date field could not be trusted as a point-in-time signal, which removed staleness — a confirmed, real signal — from this particular model. This is a data infrastructure limitation, not evidence against staleness as a signal.

## 7. Recommendation

Ranked output in three tiers: **refresh_priority** (probability ≥ 0.70, 3,009 pages) for this cycle's editorial review, **monitor** (0.40–0.70, 13,154 pages) to revisit next cycle if the signal persists, **no_action** (< 0.40, 118,075 pages) to leave alone. Every flagged page carries a reason code (`predicted_decline_risk`) and its probability, so an editor sees not just that a page was flagged, but how confidently.

Confidence: directional and decision-support only. This ranks *where* editorial attention is best spent; it does not claim refreshing a page will reverse its trend, and does not predict Google's algorithm. Before production use, a minimum traffic floor should be added to filter out the near-zero-click noise identified in Section 5's error analysis.

## 8. Reproducibility

Repo: https://github.com/Arhxmz/Flyrank-Internship-ML
Notebook: `work/notebooks/capstone.ipynb` — runs top to bottom from a fresh kernel with no errors.
Data access: requires a Hugging Face `HF_TOKEN` (read-only, gated dataset access) entered via `getpass` at runtime — never stored in the repo.
Random seed: `random_state=42` used throughout (split and model).
Environment: see `requirements.txt` at repo root.
Artifacts: `work/outputs/model_vs_baseline.png`, `work/outputs/action_tier_breakdown.png`, `work/outputs/top10_for_paper.json` — all committed, all regenerated by the notebook.

## 9. Acknowledgments & data credit

Built on the FlyRank ML Internship dataset — https://flyrank.ai

---

> **Claims checklist:** all claims above are observed, measured, directional, or decision-support language — no causal claims, no "predicted Google's algorithm," no client-identifying details. Base rate (16.2%) is reported alongside Precision@10% so the improvement isn't mistaken for a high score inflated by class imbalance.
