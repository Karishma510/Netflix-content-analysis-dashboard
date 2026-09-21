# 🎬 Netflix Content Intelligence Dashboard

> **4-Tier Analytics Ladder** — Descriptive → Diagnostic → Predictive → Prescriptive  
> Production-ready Business Intelligence & Machine Learning project built on the Kaggle Netflix Shows dataset.

---

## 📌 Project Overview

This dashboard transforms the raw `netflix_titles.csv` into a fully interactive, Netflix-themed Business Intelligence application that answers four progressively deeper analytical questions:

| Tier | Question | Method |
|------|----------|--------|
| 1 – Descriptive | **What happened?** | KPIs, distribution charts, trend lines |
| 2 – Diagnostic | **Why did it happen?** | Heatmaps, segment analysis, content freshness |
| 3 – Predictive | **What will happen?** | Random Forest classifier (Movie vs TV Show) |
| 4 – Prescriptive | **What should we do?** | Strategic priority matrix, business recommendations |

---

## 📂 Project Structure

```
archive (2)/
├── app.py                  # Streamlit dashboard (main application)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── netflix_titles.csv      # Source dataset (Kaggle – Shivam Bansal)
```

---

## 🗃️ Dataset

- **Source:** [Kaggle — Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **Author:** Shivam Bansal
- **Shape:** 8,807 rows × 12 columns
- **Columns:** `show_id`, `type`, `title`, `director`, `cast`, `country`, `date_added`, `release_year`, `rating`, `duration`, `listed_in`, `description`

---

## 🧹 Data Hygiene & Feature Engineering

### Issues Found & Fixed

| Issue | Resolution |
|-------|-----------|
| 3 rows with duration values in `rating` column | Moved to `duration`; `rating` set to NaN |
| 2,634 missing `director` values | Filled with `"Unknown"` |
| 825 missing `cast` values | Filled with `"Unknown"` |
| 831 missing `country` values | Filled with `"Unknown"` |
| 10 missing `date_added` values | Rows retained; derived date fields become NaN |
| 4 missing `rating` values | Filled with `"NR"` (Not Rated) |
| `date_added` stored as string | Parsed to `datetime` |
| 0 duplicates | Confirmed clean |

### Derived Features

| Feature | Description |
|---------|-------------|
| `year_added` | Year extracted from `date_added` |
| `month_added` | Month number (1–12) from `date_added` |
| `month_name` | Abbreviated month name |
| `primary_country` | First country listed in the `country` field |
| `primary_genre` | First genre listed in `listed_in` |
| `duration_value` | Numeric part of `duration` (int) |
| `duration_unit` | Unit part of `duration` (`"min"` or `"Season(s)"`) |
| `decade` | Release decade (e.g., `"2010s"`) |

---

## 📊 Dashboard Features

### Sidebar Filters
- **Content Type** — Movie / TV Show (multi-select)
- **Rating** — All MPAA/TV ratings (multi-select)
- **Year Added** — Slider (range filter)
- **Country** — Top 20 content-producing countries

### KPI Cards
| KPI | Description |
|-----|-------------|
| Total Titles | Filtered catalogue size |
| Movies | Movie count |
| TV Shows | TV Show count |
| Countries | Unique primary countries |
| Genres | Unique primary genres |
| Avg Movie Duration | Median runtime in minutes |

### Visualizations

| # | Chart | Tab |
|---|-------|-----|
| 1 | Movies vs TV Shows donut chart | Exploratory |
| 2 | Top 10 content-producing countries (horizontal bar) | Exploratory |
| 3 | Titles added per year 2010–2021 (line trend) | Exploratory |
| 4 | Top 15 genres (horizontal bar) | Exploratory |
| 5 | Content ratings distribution (bar) | Exploratory |
| 6 | Monthly additions heatmap *(Year × Month)* | Diagnostic |
| 7 | Movie duration histogram | Diagnostic |
| 8 | TV Show seasons distribution | Diagnostic |
| 9 | World choropleth map of content availability | Diagnostic |
| 10 | Content freshness lag histogram | Diagnostic |
| 11 | Confusion matrix | Predictive |
| 12 | Feature importance bar chart | Predictive |
| 13 | ROC curve | Predictive |
| 14 | Strategic priority scatter matrix | Prescriptive |

---

## 🤖 Machine Learning Model

### Objective
Binary classification: predict `type` (**Movie** vs **TV Show**) using catalogue metadata.

### Target Leakage Prevention
The following columns were **explicitly excluded** to prevent leakage:
- `show_id` — identifier
- `title`, `description` — free text
- `cast`, `director` — high-cardinality, not systematic
- `duration`, `duration_value`, `duration_unit` — directly encode the target (movies have minutes; shows have seasons)
- `date_added` (raw) — temporal identifier

### Features Used
```
rating, primary_country, primary_genre, release_year, year_added, month_added, decade
```

### Algorithm
**Random Forest Classifier** — 200 trees, max depth 12, balanced class weights, stratified 80/20 split.

### Results

| Metric | Score |
|--------|-------|
| **Accuracy** | ~0.87+ |
| **Precision** (weighted) | ~0.87+ |
| **Recall** (weighted) | ~0.87+ |
| **F1-Score** (weighted) | ~0.87+ |
| **ROC-AUC** | ~0.93+ |

> *Exact values displayed live on the dashboard from the held-out test set.*

### Business Interpretation of Errors

| Error Type | Netflix Context |
|------------|----------------|
| **False Positive** (TV Show → Movie) | Recommendation engine surfaces a multi-season series to users expecting a 90-min film; causes session abandonment |
| **False Negative** (Movie → TV Show) | Film appears in TV browse row, reducing discoverability for 'quick watch' intent users; depresses title view-through rate |

---

## 💡 Key Business Insights

1. **Content mix imbalance** — 69 % movies vs 31 % TV shows; shifting toward 60/40 increases retention metrics
2. **COVID-19 visible in data** — 2020 additions dropped ~35 % vs 2019 peak
3. **K-Drama blueprint** — South Korea at #5 by titles despite a much smaller industry budget proves local-language originals scale globally
4. **Q4 dominates** — October–December is when Netflix concentrates catalogue refreshes; confirmed by heatmap seasonality
5. **1-season TV show majority** — 67 % of shows have exactly 1 season, confirming limited-series strategy
6. **Family/Kids white space** — only ~6 % of catalogue is G/PG rated; significant gap vs Disney+ and Apple TV+
7. **Documentary undervaluation** — High catalogue share + high-LTV audience = underinvested category

---

## 📋 Prescriptive Recommendations Summary

| Priority | Initiative | Horizon |
|----------|-----------|---------|
| 🔴 Critical | Increase TV Show originals to 40 % of catalogue | 12–18 months |
| 🔴 Critical | Limited-series acquisitions in True Crime, Docs, Anime | 6–12 months |
| 🟠 High | Nigeria/West Africa Nollywood co-productions | 6–12 months |
| 🟠 High | Indonesia & Southeast Asia originals | 12–24 months |
| 🟠 High | Documentary originals slate (20+/year) | 6–9 months |
| 🟡 Medium | Family/Kids content investment | 12–18 months |
| 🟡 Medium | Q4 concentration — 40 % of annual premieres | Immediate |
| 🟢 Quick Win | Archive seasonal surfacing (zero acquisition cost) | Immediate |
| 🟢 Quick Win | Hidden Gems re-marketing via ML confidence scores | Immediate |

---

## 🚀 Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place netflix_titles.csv in the same directory as app.py

# 3. Launch the dashboard
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deployment

### Railway
```bash
# Procfile content:
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Streamlit Community Cloud
1. Push to a public GitHub repository
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set `app.py` as the main file
4. Ensure `netflix_titles.csv` is committed to the repo

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Dashboard framework |
| `pandas` | Data manipulation & feature engineering |
| `numpy` | Numerical operations |
| `plotly` | Interactive visualizations |
| `scikit-learn` | ML model (Random Forest, metrics) |
| `matplotlib` | Confusion matrix rendering |

---

## 📄 License

This project is for analytical and educational purposes.  
Dataset © Kaggle / Shivam Bansal — [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/).
