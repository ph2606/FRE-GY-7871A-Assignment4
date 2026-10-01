# FRE-GY 7871A — Assignment 4

Panagiotis Housos | ph2606 | NLP and the Investment Process | Fall 2026

[Open the executed notebook](assignment4.ipynb). The separate six-page PDF studies
AI-risk wording, coverage bias and AI-equity returns from **September 8 through
September 30, 2026**. The text window ends at the final **16:00 New York cash-market
close**. Text after that close lies outside the requested return window, so this
does not claim a complete 24-hour September 30 news census.

The portfolio holds equal daily target weights in NVDA, MSFT, GOOGL, AMZN, META and
AVGO. SPY and QQQ are benchmarks. Adjusted September 4 prices supply the prior
close; September 7 was Labor Day. The study has 17 market sessions, with fewer
complete Granger equations after differences, lags and conservative news timing.
Benchmark spreads are simple return differences, not CAPM alpha.

## Data and measurement

News comes from dated Google News RSS searches and the Guardian AI archive.
Hacker News stories and comments come from the public Algolia API. Queries begin
with broad AI terms; risk/safety wording is classified afterward. Main social
sentiment uses comments' own text, not their inherited thread headline. News
sentiment uses headlines; downloaded Guardian bodies are not pooled into the
main score. X motivates the event window but supplies no usable corpus; Reddit
and Bluesky access probes also failed. HN is a selected technology community.

All source payloads, retrieval clocks, hashes, publication clocks, query caps,
row-level exclusions and source types are stored locally. HN queries are split
recursively until the API cap is resolved. Google News results remain ranked and
capped, so the corpus is not exhaustive. Known institutional, corporate and
press-release sources are excluded from the media comparison; unclassified
promotional material and near-duplicate syndication can remain.

Exact normalized duplicates, insufficient text, non-AI text and the explicit
link-spam rule are audited. Historical followers, views and HN comment scores
are unavailable. No missing metric is treated as zero. Bot-name/prolific-author
flags are heuristic, not verified automation. Current story points are used
only for descriptive popularity and filtering sensitivity, never as a historical
trading signal. Language-detector scores are also sensitivity measures, not
validated English-language probabilities.

VADER 3.3.2 compound measures lexical/rule-based valence. Hu–Liu negative-token
share is the dictionary comparison connected to the assigned COVID-news paper.
The risk rules identify topics, not endorsement, extinction probability or
validated sentiment toward AI. A 60-text AI-assisted development review found
negation, quotation and target errors; no independent human accuracy is claimed.

## Timing and inference

RSS timestamps frequently appear normalized. Their conservative availability is
the following New York midnight. Guardian availability is the later publication
or modification time. HN uses its creation timestamp. Items enter the first
cash close at or after that time, with weekend text entering Monday. Current
retrieved wording may still differ from historical wording; this is a
retrospective study, not a proven real-time backtest.

The primary risk index weights mean news-headline valence and mean comment
valence equally. A missing component leaves the index missing. The one-lag
Granger regression predicts basket-minus-SPY return from its own lag and the
lagged change in that index. Two lags and raw/QQQ-relative returns form a six-test
core family with Holm correction. Ordinary F-tests, coefficient intervals and
4,999 restricted-residual fixed-design wild bootstrap diagnostics are reported.
This bootstrap holds lagged regressors fixed; it is not a recursive VAR bootstrap.

Reverse direction, individual sources, author/thread/outlet weighting, stricter
filters, English-score restriction, extra timing delay, Hu–Liu wording,
buy-and-hold weights and leave-one-equation influence checks are included. Lags
are created before deleting incomplete rows. Missing prices never silently
reweight the portfolio, and missing sessions are not compressed. ADF diagnostics
skip internal gaps and have very limited power in this window.

Coverage comparisons use a broad-AI background, common headline units and
publisher groups with topic, date and length controls. These measure sample
asymmetries, not the truth of risk estimates or a causal reader-demand effect.
The assigned paper's 91%/54% figures are not VADER percentages.

## Reproduce

Use Python 3.12. In PowerShell from this repository:

```powershell
python -m venv .venv
$assignmentPython = '.\.venv\Scripts\python.exe'
& $assignmentPython -m pip install -r requirements-lock.txt
& $assignmentPython -m ipykernel install --user --name assignment4 --display-name 'Python 3 (Assignment 4)'

# Download the declared historical window (no API key required).
& $assignmentPython scripts/01_collect_news.py --cutoff '2026-09-30T20:00:00Z'
& $assignmentPython scripts/02_collect_social.py --cutoff-utc '2026-09-30T20:00:00Z'
& $assignmentPython scripts/03_collect_market.py
& $assignmentPython -m src.language_audit

# Recalculate, validate and build the report/notebook.
& $assignmentPython scripts/04_score_text.py
& $assignmentPython scripts/05_analyze.py
& $assignmentPython -m pytest tests -q
& $assignmentPython scripts/06_build_deliverables.py
& $assignmentPython scripts/07_execute_notebook.py
xelatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report.tex
Copy-Item -LiteralPath report.pdf -Destination Panagiotis_Housos_Assignment4.pdf
```

For an intentional re-download, add `--refresh` to the acquisition commands;
old raw vintages are preserved. Never move the cutoff beyond observed time.
Requests are cached, but publisher pages, rankings, deletions and adjusted prices
can change. With datasets excluded from Git, a fresh download reproduces the
method rather than guaranteeing the exact saved snapshot. `SNAPSHOT.md` records
the submitted input hashes and aggregate counts. The current source endpoints
must remain accessible. XeLaTeX uses Cambria/Calibri with Latin Modern fallbacks.

The report and notebook narrative share `scripts/study_content.py`; exhibits
come from calculated outputs. After any new acquisition, rebuild both artifacts
to avoid stale narrative. Tests use synthetic fixtures only to check software;
they never supply observations to the empirical analysis.

## Submission and repository policy

Submit `Panagiotis_Housos_Assignment4.pdf` separately with the repository URL.
The repository retains code, the notebook with saved aggregate outputs,
dependencies, tests and [AI_USE.md](AI_USE.md). **No datasets are committed**:
`data/`, `outputs/`, `notes/`, generated report files, credentials and caches are
ignored. Full texts, document-level language/qualitative audits, individual
prices and derived data tables remain local. No trade or institutional
submission was made.
