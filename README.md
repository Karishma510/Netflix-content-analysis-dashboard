# 🎬 Netflix Content Intelligence Dashboard
### IBM AICTE Business Intelligence Project

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Dash](https://img.shields.io/badge/Dash-2.17-informational?logo=plotly)
![Plotly](https://img.shields.io/badge/Plotly-5.22-purple?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Project Overview

This project is a fully interactive **Business Intelligence (BI) Dashboard** built on the Netflix Movies and TV Shows dataset. It was developed as part of the **IBM AICTE Internship Program** to demonstrate end-to-end BI skills including data cleaning, exploratory data analysis (EDA), KPI identification, trend analysis, and interactive visualization.

The dashboard provides strategic insights into Netflix's global content catalog — covering content type distribution, geographic reach, genre trends, audience ratings, seasonal patterns, and catalog growth over time.

---

## 📦 Dataset

| Attribute | Details |
|-----------|---------|
| **Source** | [Kaggle — Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| **Author** | Shivam Bansal |
| **Records** | 8,807 titles |
| **Columns** | 12 (show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description) |
| **Coverage** | Titles added to Netflix up to mid-2021 |

### Column Descriptions

| Column | Description |
|--------|-------------|
| `show_id` | Unique ID for each title |
| `type` | Movie or TV Show |
| `title` | Title name |
| `director` | Director(s) name |
| `cast` | Cast members |
| `country` | Country of origin |
| `date_added` | Date added to Netflix |
| `release_year` | Original release year |
| `rating` | Audience rating (TV-MA, PG-13, etc.) |
| `duration` | Duration in minutes (Movie) or seasons (TV Show) |
| `listed_in` | Genres |
| `description` | Short synopsis |

---

## 📊 Key Performance Indicators (KPIs)

| KPI | Value |
|-----|-------|
| Total Titles | 8,807 |
| Movies | 6,131 (69.6%) |
| TV Shows | 2,676 (30.4%) |
| Unique Countries | 748 |
| Unique Genres | 42 |
| Average Movie Duration | ~99 minutes |
| Most Common Rating | TV-MA (3,207 titles) |
| Top Content Country | United States (3,690 titles) |
| Catalog Year Range | 1925 – 2021 |

---

## 🚀 Features

- **6 Interactive Tabs**: Overview, Geography, Content, Trends, Insights, Explorer
- **Global Filters**: Content type, Rating, and Year Added range sliders
- **KPI Cards**: Real-time updating summary metrics
- **Charts**: Donut, bar, choropleth map, line, area, histogram, heatmap, stacked bar
- **Data Explorer**: Sortable/filterable table of 200 records per view
- **Business Insights**: Structured analysis of trends, drivers, risks, and opportunities
- **Dark Netflix-themed UI**: Red accent palette matching Netflix branding

---

## 🗂️ Project Structure

```
📦 Netflix BI Dashboard
├── app.py                  # Main Dash application (dashboard + all logic)
├── netflix_titles.csv      # Dataset (Kaggle)
├── requirements.txt        # Python dependencies
├── README.md               # This documentation
└── Project_Report.docx     # Full IBM AICTE project report
```

---

## ⚙️ Setup & Run

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Step 1 — Clone / Download

Place all files in the same directory:
```
netflix_titles.csv
app.py
requirements.txt
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Dashboard

```bash
python app.py
```

### Step 4 — Open in Browser

```
http://127.0.0.1:8050
```

---

## 🔍 Exploratory Data Analysis Summary

### Data Quality
| Issue | Count | Action Taken |
|-------|-------|-------------|
| Missing Director | 2,634 (29.9%) | Filled with "Unknown" |
| Missing Cast | 825 (9.4%) | Filled with "Unknown" |
| Missing Country | 831 (9.4%) | Filled with "Unknown" |
| Missing Date Added | 10 (0.1%) | Left as NaT |
| Missing Rating | 4 (0.0%) | Filled with "Not Rated" |
| Missing Duration | 3 (0.0%) | Left as NaN |

### Feature Engineering
- `year_added` / `month_added` — extracted from `date_added`
- `primary_country` — first country from multi-value `country` field
- `primary_genre` — first genre from multi-value `listed_in` field
- `decade` — release decade (e.g., 1990s, 2010s)
- `duration_value` — numeric duration extracted from string
- `age_at_add` — content age when added to Netflix

---

## 📈 Business Insights

### ✅ Trends
- Netflix catalog grew **5× from 2015 to 2019**
- Q4 (Oct–Dec) consistently shows the **highest monthly additions**
- TV Show proportion is increasing — reflecting shift to episodic content strategy

### 🌍 Geographic Drivers
- US, India, and UK account for **~62% of all content**
- South Korea and Japan are the **fastest growing non-English markets**
- Spanish-language content (Spain + Mexico + Colombia) shows strong growth

### 🎬 Content Drivers
- **International Movies** is the #1 genre (2,752 titles)
- Dramas and Comedies are universally dominant formats
- Most TV Shows have only **1 season** — limited series format is dominant

### ⚠️ Risks
- High US content dependency (42%) creates platform risk
- 30% missing director data hurts content attribution and recommendations
- Catalog concentration in English/Hindi languages

### 💡 Opportunities
- Korean, Japanese, and Spanish originals — high growth, high ROI segments
- Children & Family and Documentary genres are **underserved**
- Multi-season serialized shows can increase subscriber lock-in and retention

---

## 🤖 Prediction Model Note

A content-type classification model (Movie vs TV Show) can be built using features like:
- `primary_genre`, `primary_country`, `rating`, `release_year`
- Using **Logistic Regression** or **Random Forest Classifier** (scikit-learn)

However, since the dataset is descriptive metadata (not behavioral engagement data), predictive modeling is supplementary to the BI objective of this project. The dashboard focuses on **descriptive and diagnostic analytics** as the primary deliverable.

---

## 📋 IBM AICTE Submission Checklist

- [x] Dataset loaded and cleaned
- [x] EDA performed (missing values, distributions, outliers)
- [x] KPIs identified and documented
- [x] Interactive dashboard with 6 tabs
- [x] Geographic analysis with world map
- [x] Trend analysis with time-series charts
- [x] Business insights section (trends, risks, opportunities)
- [x] Data explorer table with filter/sort
- [x] `requirements.txt` with pinned versions
- [x] `README.md` with dataset source
- [x] `Project_Report.docx` generated

---

## 📝 License

This project is for educational purposes as part of the **IBM AICTE Internship Program**.  
Dataset credit: [Shivam Bansal on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) — CC0: Public Domain.

---

*Built with Python · Dash · Plotly | IBM AICTE Business Intelligence Project*
