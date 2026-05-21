# 🏠 Egypt Real Estate Appraiser

ML-based property price prediction for Egypt, trained on **74,240 listings** scraped from [Dubizzle.com.eg](https://www.dubizzle.com.eg) using an async two-stage pipeline.

---

## 📊 Model Results

**Stratified approach:** one dedicated Random Forest or Gradient Boosting model per property type.

| Property Type | Model | Records | MAE (M EGP) | R² | MAPE |
|--------------|-------|--------:|:-----------:|:--:|:----:|
| Apartment | Random Forest | 43,399 | 2.056 | 0.534 | 28.1% |
| Villa | Random Forest | 13,975 | 7.933 | 0.602 | 29.6% |
| Town House | Gradient Boosting | 5,565 | 3.816 | 0.561 | 24.1% |
| Twin House | Gradient Boosting | 3,387 | 4.512 | 0.633 | 23.2% |
| Duplex | Gradient Boosting | 3,435 | 3.322 | 0.410 | 31.4% |
| Penthouse | Gradient Boosting | 2,577 | 3.373 | 0.503 | 23.6% |
| Studio | Random Forest | 994 | 1.224 | 0.588 | 25.0% |
| **Global** | — | **74,240** | **3.532** | **0.758** | **27.87%** |

### Dataset Evolution

| Dataset | Origin | Records | R² | MAPE |
|---------|--------|--------:|:--:|:----:|
| Kaggle | PropertyFinder.eg (public) | 16,083 | 0.49 | 57% |
| data.xlsx | Team's first Dubizzle scrape (no new features) | 81,076 | 0.69 | 33% |
| **Dubizzle Final** | **Improved scrape + Ownership/Payment/Completion** | **74,240** | **0.758** | **27.87%** |

> **data.xlsx** = team's first custom scraping attempt. Same field schema as Kaggle (no Ownership, Payment, Completion). The need to add these features drove the improved scraping schema in the final version.

---

## 📁 Repository Structure

```
egypt-real-estate-appraiser/
│
├── Egypt_RE_Appraiser_Final.ipynb   # Full pipeline: EDA + models + figures
├── train.py                          # Retrain: python train.py
├── app.py                            # Streamlit app: streamlit run app.py
├── dubizzle_scraper.py               # Stage 2 scraper (reads URLs.xlsx → dubizzle_listings.xlsx)
│
├── requirements.txt                  # App dependencies
├── requirements_scraper.txt          # Scraper: aiohttp + lxml + beautifulsoup4 + openpyxl
├── README.md
├── .gitignore
│
└── [not pushed — too large for git]
    ├── model_final.pkl               # 8 models + encoders (upload to Streamlit Files)
    ├── dubizzle_listings.xlsx        # Raw scraped data
    ├── URLs.xlsx                     # Stage 1 output (107,276 URLs)
    └── cleaned_dubizzle_final.csv    # 74,240 clean records
```

---

## ⚙️ Features Used

| Feature | Type | Notes |
|---------|------|-------|
| `Built-Up Area (m²)` | Numeric | Strongest numeric predictor |
| `Bedrooms` | Numeric | Bathrooms excluded (r=0.79 with bedrooms) |
| `Gov+City` | Label encoded | 181 bounded combinations |
| `City+District` | Label encoded | **1,473 combinations — most important feature** |
| `Ownership` | 1/0/0.5 | Primary=1, Resale=0 |
| `Payment Option` | 1/0/0.5 | Cash=1, Installment=0 |
| `Completion Status` | 1/0/0.5 | Ready=1, Off-plan=0 |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (or upload model_final.pkl directly)
python train.py

# 3. Launch app
streamlit run app.py
```

---

## ☁️ Streamlit Cloud Deployment

1. Push code to GitHub (exclude `model_final.pkl` — add to `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App
3. Select: repo · branch: `main` · file: `app.py`
4. Click **Deploy**
5. In App Settings → **Files** → upload `model_final.pkl`
6. App is live at `https://your-app-name.streamlit.app`

---

## 🕷️ Scraping Pipeline

Two-stage pipeline: Stage 1 scrapes listing cards (fast); Stage 2 visits each property page for full details.

### Stage 1 — `dubizzle_scraper_stage1.py` (Card Scraper)

Uses **`requests`** (synchronous). Reads Dubizzle sitemaps → visits category pages → extracts data from listing cards. Captures: Title, Price, Area, Beds, Baths, Payment Type, Ownership, Completion Status, Location, URL.

```bash
pip install -r requirements_scraper.txt
python dubizzle_scraper_stage1.py
# Output: dubizzle_properties_YYYYMMDD_HHMMSS.xlsx
```

> **data.xlsx** was produced by an earlier version of this stage (same card-level approach, before Ownership/Payment/Completion were extracted).

### Stage 2 — `dubizzle_scraper.py` (Detail Page Scraper)

Uses **`aiohttp`** (async, 40 concurrent workers). Reads `URLs.xlsx` → visits each individual property page → extracts complete details including compound, district, city.

```bash
python dubizzle_scraper.py                    # full run (~4-6 hours @ 40 workers)
python dubizzle_scraper.py --limit 500        # quick test
python dubizzle_scraper.py --workers 20       # adjust concurrency
python dubizzle_scraper.py --resume           # continue after interruption
```

**Files:**
- Input: `URLs.xlsx` (listing URLs from Stage 1)
- Output: `dubizzle_listings.xlsx`
- Checkpoint: `.checkpoint.csv` (auto-created — delete to restart)

**robots.txt compliance:**
- ✅ Allowed: `/en/properties/{category}/{city}/`
- ❌ Blocked: `/en/properties/` and `?product=all`

---

## 👥 Team

| Member | Role |
|--------|------|
| Member 1 | Data collection & two-stage scraping |
| Member 2 | Data cleaning, EDA & feature engineering |
| Member 3 | ML models, evaluation & Streamlit app |
| Member 4 | Documentation & presentation |

---

*Python · aiohttp · Scikit-learn · Streamlit | DEPI Data Science Track 2026*
