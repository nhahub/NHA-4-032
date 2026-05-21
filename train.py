"""
train.py — Egypt Real Estate Appraiser (Final Pipeline)
========================================================
Loads dubizzle_listings.xlsx → cleans → encodes → trains stratified
models (one per property type) → saves model_final.pkl

Fixes applied:
  - Status filter: keep only Status=OK
  - Property Type: remove 2 rows with compound names leaked in
  - District fix: missing → Compound → City; District==City → Compound
  - Bathrooms removed: correlated 0.79 with bedrooms

New features vs Kaggle:
  + Ownership  (Primary=1, Resale=0)
  + Payment    (Cash=1, Installment=0)
  + Completion (Ready=1, Off-plan=0)
  + City+District bounded location (1,473 combinations)

Usage:
    python train.py
    python train.py --input dubizzle_listings.xlsx --output model_final.pkl
"""

import pandas as pd, numpy as np, joblib, os, warnings, argparse
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


VALID_TYPES = ['Apartment','Stand Alone Villa','Town House','Twin House',
               'Duplex','Penthouse','Studio','iVilla','Hotel Apartment','Roof']
TYPE_MAP = {
    'Apartment':'Apartment','Stand Alone Villa':'Villa','Town House':'Town House',
    'Duplex':'Duplex','Twin House':'Twin House','Penthouse':'Penthouse',
    'Studio':'Studio','iVilla':'Villa','Hotel Apartment':'Other','Roof':'Other',
}


# ── 1. Load & Fix ─────────────────────────────────────────────────────────────
def load_and_fix(path):
    ext = os.path.splitext(path)[-1].lower()
    df  = pd.read_excel(path) if ext in ('.xlsx','.xls') else pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Status filter
    if 'Status' in df.columns:
        before = len(df)
        df = df[df['Status'] == 'OK'].copy()
        print(f"  Status filter: {before:,} → {len(df):,} (kept OK only)")

    # Property type filter (remove leaked compound names)
    if 'Property Type' in df.columns:
        df = df[df['Property Type'].isin(VALID_TYPES)].copy()

    # Strip whitespace
    for c in ['Property Type','City','Governorate','District','Compound',
              'Ownership','Payment Option','Completion Status']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace('nan', np.nan)

    # District fix
    df['district_fixed'] = df['District'].copy()
    df.loc[df['district_fixed'].isna(), 'district_fixed'] = \
        df.loc[df['district_fixed'].isna(), 'Compound']
    df.loc[df['district_fixed'].isna(), 'district_fixed'] = \
        df.loc[df['district_fixed'].isna(), 'City']
    same = df['district_fixed'].str.strip() == df['City'].str.strip()
    has_cmp = df['Compound'].notna()
    df.loc[same & has_cmp, 'district_fixed'] = df.loc[same & has_cmp, 'Compound']

    # Bounded location features
    df['gov_city']      = df['Governorate'].str.strip() + ' — ' + df['City'].str.strip()
    df['city_district'] = df['City'].str.strip() + ' / ' + df['district_fixed'].str.strip()
    print(f"  gov_city combinations    : {df['gov_city'].nunique()}")
    print(f"  city_district combinations: {df['city_district'].nunique()}")
    return df


# ── 2. Clean ──────────────────────────────────────────────────────────────────
def clean(df):
    df = df.copy()
    df['prop_type'] = df['Property Type'].map(TYPE_MAP).fillna('Other')
    df['price_m']   = df['Price (EGP)'] / 1_000_000

    p_hi = df['price_m'].quantile(0.995)
    a_hi = df['Built-Up Area (m²)'].quantile(0.995)
    n_before = len(df)
    df = df[(df['price_m'] >= 0.5) & (df['price_m'] <= p_hi)]
    df = df[(df['Built-Up Area (m²)'] >= 20) & (df['Built-Up Area (m²)'] <= a_hi)]
    df = df[df['Bedrooms'].between(1, 8)]
    df = df.drop_duplicates(subset=['Price (EGP)','Built-Up Area (m²)','Bedrooms','City','district_fixed'])
    df = df.dropna(subset=['price_m','Built-Up Area (m²)','Bedrooms','Governorate','City'])
    print(f"  Removed: {n_before - len(df):,}  |  Clean: {len(df):,}")

    df['ownership_enc']  = df['Ownership'].map({'Primary':1,'Resale':0}).fillna(0.5)
    df['payment_enc']    = df['Payment Option'].map({'Cash':1,'Installment':0,'Cash or Installment':0.5}).fillna(0.5)
    df['completion_enc'] = df['Completion Status'].map({'Ready':1,'Off-plan':0}).fillna(0.5)
    df['log_price_m']    = np.log(df['price_m'])
    return df


# ── 3. Encode ─────────────────────────────────────────────────────────────────
def encode(df):
    le_gc = LabelEncoder(); df['gc_enc'] = le_gc.fit_transform(df['gov_city'].fillna('Unknown'))
    le_cd = LabelEncoder(); df['cd_enc'] = le_cd.fit_transform(df['city_district'].fillna('Unknown'))
    return df, le_gc, le_cd


# ── 4. Metrics ────────────────────────────────────────────────────────────────
def get_metrics(yt, yp):
    return dict(
        MAE=round(mean_absolute_error(yt, yp), 3),
        RMSE=round(np.sqrt(mean_squared_error(yt, yp)), 3),
        R2=round(r2_score(yt, yp), 4),
        MAPE=round(np.mean(np.abs((yt - yp) / yt)) * 100, 2),
    )


# ── 5. Train stratified ───────────────────────────────────────────────────────
def train_stratified(df, features):
    VALID = [t for t, c in df['prop_type'].value_counts().items() if c >= 200]
    type_models, type_results = {}, {}
    all_true, all_pred = [], []

    print(f"\n  {'Type':<14} {'n':>6} {'MAE':>8} {'R²':>8} {'MAPE':>8} {'CVR²':>8}")
    print("  " + "─" * 55)

    for ptype in VALID:
        sub = df[df['prop_type'] == ptype].dropna(subset=features)
        X = sub[features]; y_log = sub['log_price_m']; y_raw = sub['price_m']
        X_tr, X_te, y_tr_log, _ = train_test_split(X, y_log, test_size=0.2, random_state=42)
        _,    _,    _,        yte = train_test_split(X, y_raw, test_size=0.2, random_state=42)

        best_model, best_name, best_r2 = None, '', -99
        for name, model in [
            ('Random Forest',
             RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=3,
                                   random_state=42, n_jobs=-1)),
            ('Gradient Boosting',
             GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1,
                                       subsample=0.8, random_state=42)),
        ]:
            model.fit(X_tr, y_tr_log)
            r2 = r2_score(yte.values, np.exp(model.predict(X_te)))
            if r2 > best_r2:
                best_r2, best_name, best_model = r2, name, model

        yp  = np.exp(best_model.predict(X_te))
        m   = get_metrics(yte.values, yp)
        cv  = cross_val_score(best_model, X_tr, y_tr_log, cv=5, scoring='r2').mean()
        m.update({'CVR2': round(cv, 4), 'count': len(sub), 'model_name': best_name})
        type_models[ptype]  = best_model
        type_results[ptype] = m
        all_true.extend(yte.values); all_pred.extend(yp)
        print(f"  {ptype:<14} {len(sub):>6,} {m['MAE']:>8.3f} {m['R2']:>8.4f} {m['MAPE']:>7.1f}% {cv:>8.4f}")

    t, p = np.array(all_true), np.array(all_pred)
    g = dict(R2=round(r2_score(t,p),4), MAE=round(mean_absolute_error(t,p),3),
             MAPE=round(np.mean(np.abs((t-p)/t))*100,2))
    print("  " + "─" * 55)
    print(f"  Global  R²={g['R2']}  MAE={g['MAE']}M  MAPE={g['MAPE']}%")
    return type_models, type_results, VALID, g


# ── Main ──────────────────────────────────────────────────────────────────────
def main(input_path, output_path):
    print("=" * 65)
    print("  Egypt Real Estate Appraiser — Training Pipeline")
    print("  Dubizzle Dataset · Stratified by Property Type")
    print("=" * 65)

    print(f"\n[1/4] Load & fix: {input_path}")
    df_raw = load_and_fix(input_path)
    print(f"  Raw records: {len(df_raw):,}")

    print("\n[2/4] Clean...")
    df = clean(df_raw)

    print("\n[3/4] Encode...")
    df, le_gc, le_cd = encode(df)
    features    = ['Built-Up Area (m²)','Bedrooms','gc_enc','cd_enc',
                   'ownership_enc','payment_enc','completion_enc']
    feat_labels = ['Area (m²)','Bedrooms','Gov+City','City+District',
                   'Ownership','Payment','Completion']

    print("\n[4/4] Train stratified models...")
    type_models, type_results, valid_types, global_m = train_stratified(df, features)

    # Build cascading dropdown maps for Streamlit app
    gov_city_map, city_district_map = {}, {}
    for gc in le_gc.classes_:
        if not isinstance(gc, str): continue
        parts = gc.split(' — ', 1)
        if len(parts) == 2: gov_city_map.setdefault(parts[0], []).append(gc)
    for cd in le_cd.classes_:
        if not isinstance(cd, str): continue
        parts = cd.split(' / ', 1)
        if len(parts) == 2: city_district_map.setdefault(parts[0], []).append(cd)

    bundle = {
        'approach'         : 'stratified_dubizzle',
        'type_models'      : type_models,
        'type_results'     : type_results,
        'le_gc'            : le_gc,
        'le_cd'            : le_cd,
        'features'         : features,
        'feat_labels'      : feat_labels,
        'valid_types'      : valid_types,
        'gov_city_map'     : gov_city_map,
        'city_district_map': city_district_map,
        'gov_cities'       : [g for g in le_gc.classes_ if isinstance(g, str)],
        'city_districts'   : [c for c in le_cd.classes_ if isinstance(c, str)],
        'global_metrics'   : global_m,
    }
    joblib.dump(bundle, output_path)
    print(f"\n  ✅  {len(type_models)} models → {output_path}")
    print(f"     Global: R²={global_m['R2']}  MAE={global_m['MAE']}M  MAPE={global_m['MAPE']}%")
    print("\n  Run:  streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Egypt Real Estate Appraiser — Train")
    p.add_argument("--input",  default="dubizzle_listings.xlsx")
    p.add_argument("--output", default="model_final.pkl")
    args = p.parse_args()
    main(args.input, args.output)
