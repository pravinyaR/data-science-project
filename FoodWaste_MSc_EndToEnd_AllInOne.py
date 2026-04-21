#!/usr/bin/env python
# coding: utf-8

# ## Put these files in `DATA_DIR`
# - `Food Waste data and research - by country (1).csv`
# - `FoodBalanceSheets_E_All_Data_(Normalized).zip`
# - `ConsumerPriceIndices_E_All_Data_(Normalized).zip`
# - `Production_Crops_Livestock_E_All_Data_(Normalized).zip` (preferred) *(fallback: `Production_Indices_E_All_Data_(Normalized).zip`)*
# - `Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip` (preferred) *(fallback: `Trade_Indices_E_All_Data_(Normalized).zip`)*
# 
# ## Outputs created
# - `food_waste_faostat_model_input.csv`
# - `outputs/` (CSVs + model) and `outputs/figures/` (PNG graphs)
# 

# In[2]:


from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import re

# -----------------------------
# CONFIG: set this folder path
# -----------------------------
# If all files are in the same folder as this notebook, keep Path('.')
# Otherwise set a full path, e.g., Path('/Users/you/Desktop/project')
DATA_DIR = Path("/Users/eswarkalla/Desktop/Pravinya") 
YEAR_START = 2019
YEAR_END = 2021
CHUNKSIZE = 200_000

TARGET_COL = 'combined figures (kg/capita/year)'

OUTPUT_DIR = DATA_DIR / 'outputs'
FIG_DIR = OUTPUT_DIR / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5


# In[3]:


# ============================================================
# A) DATA INTEGRATION HELPERS (FAOSTAT ZIP -> features -> merge)
# ============================================================

def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def parse_m49(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def find_file(folder: Path, must_contain, exts=(".zip", ".csv")):
    must_contain = [m.lower() for m in must_contain]
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        name = p.name.lower()
        if all(m in name for m in must_contain):
            return p
    return None

def ensure_main_csv(path: Path, extract_dir: Path) -> Path:
    if path is None:
        return None
    if path.suffix.lower() == ".csv":
        return path

    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(path, "r") as z:
        normalized_csvs = [n for n in z.namelist() if n.lower().endswith("_all_data_(normalized).csv")]
        if normalized_csvs:
            main_csv = normalized_csvs[0]
        else:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise FileNotFoundError(f"No CSV found inside: {path}")
            csvs = sorted(csvs, key=lambda n: z.getinfo(n).file_size, reverse=True)
            main_csv = csvs[0]

        out_path = extract_dir / Path(main_csv).name
        if not out_path.exists():
            z.extract(main_csv, path=extract_dir)
            extracted_path = extract_dir / main_csv
            if extracted_path != out_path and extracted_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_path.replace(out_path)

    return out_path

def build_chunk_reader(csv_path: Path, force_python=False):
    allowed = {
        "area code", "area code m49", "area",
        "reporter country code", "reporter countries code", "reporter code",
        "reporter country", "reporter countries", "reporter",
        "partner country", "partner countries", "partner",
        "item", "item code", "element", "element code",
        "year", "year code", "months", "months code",
        "unit", "value", "flag", "note",
    }

    def usecols(col_name):
        return norm_text(col_name) in allowed

    if force_python:
        return pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNKSIZE,
                           engine="python", on_bad_lines="skip")

    try:
        return pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNKSIZE, low_memory=False)
    except Exception:
        return pd.read_csv(csv_path, usecols=usecols, chunksize=CHUNKSIZE,
                           engine="python", on_bad_lines="skip")

def standardize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    cols = list(chunk.columns)
    cols_norm = {norm_text(c): c for c in cols}

    def pick(*aliases):
        for a in aliases:
            a_n = norm_text(a)
            if a_n in cols_norm:
                return cols_norm[a_n]
        for a in aliases:
            a_n = norm_text(a)
            for k, v in cols_norm.items():
                if a_n in k:
                    return v
        return None

    area_code_col = pick("Area Code (M49)", "Area Code", "Reporter Countries Code", "Reporter Country Code", "Reporter Code")
    area_col = pick("Area", "Reporter Countries", "Reporter Country", "Reporter")
    partner_col = pick("Partner Countries", "Partner Country", "Partner")
    item_col = pick("Item")
    element_col = pick("Element")
    year_col = pick("Year")
    value_col = pick("Value")
    month_col = pick("Months")
    unit_col = pick("Unit")

    out = pd.DataFrame(index=chunk.index)
    out["area_code"] = parse_m49(chunk[area_code_col]) if area_code_col else pd.Series([pd.NA] * len(chunk), dtype="Int64")
    out["area"] = chunk[area_col].astype(str) if area_col else ""
    out["partner"] = chunk[partner_col].astype(str) if partner_col else ""
    out["item"] = chunk[item_col].astype(str) if item_col else ""
    out["element"] = chunk[element_col].astype(str) if element_col else ""
    out["year"] = pd.to_numeric(chunk[year_col], errors="coerce") if year_col else np.nan
    out["value"] = pd.to_numeric(chunk[value_col], errors="coerce") if value_col else np.nan
    if month_col:
        out["month"] = chunk[month_col].astype(str)
    if unit_col:
        out["unit"] = chunk[unit_col].astype(str)

    out = out[out["year"].between(YEAR_START, YEAR_END)]
    out = out[out["area_code"].notna() & out["value"].notna()].copy()

    out["item_l"] = out["item"].str.lower()
    out["element_l"] = out["element"].str.lower()
    out["partner_l"] = out["partner"].str.lower()
    return out

def contains_any(series: pd.Series, patterns) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for p in patterns:
        mask |= series.str.contains(p, regex=True, na=False)
    return mask

class FeatureAccumulator:
    def __init__(self):
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(int))
        self.kinds = {}

    def add(self, df: pd.DataFrame, feat_name: str, within_year: str = "sum"):
        if df.empty:
            return
        if feat_name in self.kinds and self.kinds[feat_name] != within_year:
            raise ValueError(f"Conflicting aggregation kind for {feat_name}")
        self.kinds.setdefault(feat_name, within_year)

        if within_year == "sum":
            g = df.groupby(["area_code", "year"], dropna=False)["value"].sum(min_count=1)
            for (a, y), v in g.items():
                if pd.notna(v):
                    self.sums[feat_name][(int(a), int(y))] += float(v)

        elif within_year == "mean":
            g = df.groupby(["area_code", "year"], dropna=False)["value"].agg(["sum", "count"])
            for (a, y), row in g.iterrows():
                if int(row["count"]) > 0:
                    self.sums[feat_name][(int(a), int(y))] += float(row["sum"])
                    self.counts[feat_name][(int(a), int(y))] += int(row["count"])
        else:
            raise ValueError("within_year must be 'sum' or 'mean'")

    def finalize(self) -> pd.DataFrame:
        rows = []
        for feat, sum_map in self.sums.items():
            kind = self.kinds[feat]
            per_area_vals = defaultdict(list)
            for (area, year), s in sum_map.items():
                if kind == "mean":
                    c = self.counts[feat].get((area, year), 0)
                    if c == 0:
                        continue
                    v = s / c
                else:
                    v = s
                per_area_vals[area].append(v)
            for area, vals in per_area_vals.items():
                if vals:
                    rows.append((area, feat, float(np.mean(vals))))

        if not rows:
            return pd.DataFrame(columns=["area_code"])

        out = pd.DataFrame(rows, columns=["area_code", "feature", "value"])
        out = out.pivot(index="area_code", columns="feature", values="value").reset_index()
        return out


# In[4]:


# ============================================================
# Feature builders: FBS, CP, QCL, TM (with fallbacks)
# ============================================================

def process_fbs(csv_path: Path) -> pd.DataFrame:
    acc = FeatureAccumulator()
    for chunk in build_chunk_reader(csv_path):
        d = standardize_chunk(chunk)
        if d.empty:
            continue

        rules = [
            ("fbs_food_supply_kgcapyr", [r"food supply quantity"]),
            ("fbs_food_supply_kcalcapday", [r"food supply.*kcal"]),
            ("fbs_protein_gcapday", [r"protein supply"]),
            ("fbs_fat_gcapday", [r"fat supply"]),
            ("fbs_losses", [r"\bloss"]),
            ("fbs_feed", [r"\bfeed\b"]),
            ("fbs_seed", [r"\bseed\b"]),
            ("fbs_processing", [r"process"]),
            ("fbs_stock_variation", [r"stock"]),
            ("fbs_residuals", [r"residual"]),
            ("fbs_domestic_supply_qty", [r"domestic supply quantity"]),
            ("fbs_import_qty", [r"import quantity"]),
            ("fbs_export_qty", [r"export quantity"]),
            ("fbs_tourist_consumption", [r"tourist"]),
            ("fbs_population", [r"total population"]),
        ]
        for feat, pats in rules:
            m = contains_any(d["element_l"], pats)
            if m.any():
                acc.add(d[m], feat, within_year="sum")

        # Optional commodity-group food supply
        m_food = d["element_l"].str.contains("food supply quantity", na=False)
        if m_food.any():
            dq = d[m_food]
            groups = {
                "cereals": [r"wheat", r"rice", r"maize|corn", r"barley", r"sorghum", r"millet", r"oats", r"rye", r"cereal"],
                "fruits": [r"fruit", r"banana", r"apple", r"orange", r"grape", r"mango", r"pineapple"],
                "vegetables": [r"vegetable", r"tomato", r"onion", r"pepper", r"cabbage", r"eggplant", r"okra"],
                "meat": [r"meat", r"bovine", r"pig", r"poultry", r"mutton", r"goat"],
                "milk_dairy": [r"milk", r"butter", r"cheese", r"cream"],
            }
            for gname, pats in groups.items():
                mg = contains_any(dq["item_l"], pats)
                if mg.any():
                    acc.add(dq[mg], f"fbs_{gname}_food_supply_kgcapyr", within_year="sum")

    return acc.finalize()

def process_cp(csv_path: Path) -> pd.DataFrame:
    acc = FeatureAccumulator()
    for chunk in build_chunk_reader(csv_path):
        d = standardize_chunk(chunk)
        if d.empty:
            continue

        series_l = d["item_l"]
        rules = [
            ("cp_food_cpi", [r"\bfood\b.*(indices|index|cpi)", r"food cpi"]),
            ("cp_general_cpi", [r"\bgeneral\b.*(indices|index|cpi)", r"general cpi", r"consumer price index"]),
            ("cp_food_inflation_avg", [r"food price inflation.*weighted average"]),
            ("cp_food_inflation_median", [r"food price inflation.*median"]),
            ("cp_general_inflation_avg", [r"inflation.*consumer prices.*weighted average"]),
            ("cp_general_inflation_median", [r"inflation.*consumer prices.*median"]),
        ]
        for feat, pats in rules:
            m = contains_any(series_l, pats)
            if m.any():
                acc.add(d[m], feat, within_year="mean")

    out = acc.finalize()
    if {"cp_food_cpi", "cp_general_cpi"}.issubset(out.columns):
        out["cp_food_general_gap"] = out["cp_food_cpi"] - out["cp_general_cpi"]
    return out

def process_qcl_detailed(csv_path: Path) -> pd.DataFrame:
    acc = FeatureAccumulator()
    for chunk in build_chunk_reader(csv_path):
        d = standardize_chunk(chunk)
        if d.empty:
            continue

        m_prod = contains_any(d["element_l"], [r"^production$", r"production quantity", r"\bproduction\b"])
        m_area = contains_any(d["element_l"], [r"area harvested", r"harvested area"])
        m_yld = contains_any(d["element_l"], [r"\byield\b"])

        if m_prod.any():
            acc.add(d[m_prod], "qcl_production_total", within_year="sum")
        if m_area.any():
            acc.add(d[m_area], "qcl_area_harvested_total", within_year="sum")
        if m_yld.any():
            acc.add(d[m_yld], "qcl_yield_avg", within_year="mean")

        if m_prod.any():
            dp = d[m_prod]
            groups = {
                "cereals": [r"wheat", r"rice", r"maize|corn", r"barley", r"sorghum", r"millet", r"oats", r"rye", r"cereal"],
                "fruits": [r"fruit", r"banana", r"apple", r"orange", r"grape", r"mango", r"pineapple"],
                "vegetables": [r"vegetable", r"tomato", r"onion", r"pepper", r"cabbage", r"eggplant", r"okra"],
                "roots_tubers": [r"potato", r"cassava", r"yam", r"sweet potato"],
                "meat": [r"meat", r"bovine", r"pig", r"poultry", r"mutton", r"goat"],
                "milk": [r"milk"],
            }
            for gname, pats in groups.items():
                mg = contains_any(dp["item_l"], pats)
                if mg.any():
                    acc.add(dp[mg], f"qcl_production_{gname}", within_year="sum")

    return acc.finalize()

def process_prod_indices(csv_path: Path) -> pd.DataFrame:
    acc = FeatureAccumulator()
    for chunk in build_chunk_reader(csv_path):
        d = standardize_chunk(chunk)
        if d.empty:
            continue
        m_gross = contains_any(d["element_l"], [r"gross production index"])
        m_pc = contains_any(d["element_l"], [r"per capita production index"])
        if m_gross.any():
            acc.add(d[m_gross], "pi_gross_prod_index_mean", within_year="mean")
        if m_pc.any():
            acc.add(d[m_pc], "pi_percap_prod_index_mean", within_year="mean")
    return acc.finalize()

def process_tm_detailed(csv_path: Path) -> pd.DataFrame:
    acc = FeatureAccumulator()
    for chunk in build_chunk_reader(csv_path):
        d = standardize_chunk(chunk)
        if d.empty:
            continue
        rules = [
            ("tm_import_qty", [r"import quantity"]),
            ("tm_export_qty", [r"export quantity"]),
            ("tm_import_val", [r"import value"]),
            ("tm_export_val", [r"export value"]),
        ]
        for feat, pats in rules:
            m = contains_any(d["element_l"], pats)
            if m.any():
                acc.add(d[m], feat, within_year="sum")

    out = acc.finalize()
    if {"tm_import_qty", "tm_export_qty"}.issubset(out.columns):
        out["tm_net_import_qty"] = out["tm_import_qty"] - out["tm_export_qty"]
    if {"tm_import_val", "tm_export_val"}.issubset(out.columns):
        out["tm_net_import_val"] = out["tm_import_val"] - out["tm_export_val"]
    return out

def process_trade_indices(csv_path: Path) -> pd.DataFrame:
    acc = FeatureAccumulator()
    for chunk in build_chunk_reader(csv_path, force_python=True):
        d = standardize_chunk(chunk)
        if d.empty:
            continue
        rules = [
            ("ti_import_qty_idx", [r"import quantity index"]),
            ("ti_export_qty_idx", [r"export quantity index"]),
            ("ti_import_val_idx", [r"import value index"]),
            ("ti_export_val_idx", [r"export value index"]),
            ("ti_import_uv_idx", [r"import unit.?value index"]),
            ("ti_export_uv_idx", [r"export unit.?value index"]),
        ]
        for feat, pats in rules:
            m = contains_any(d["element_l"], pats)
            if m.any():
                acc.add(d[m], feat, within_year="mean")

    out = acc.finalize()
    if {"ti_import_val_idx", "ti_export_val_idx"}.issubset(out.columns):
        out["ti_val_idx_gap"] = out["ti_import_val_idx"] - out["ti_export_val_idx"]
    return out


# In[5]:


# ============================================================
# Run merge pipeline (creates food_waste_faostat_model_input.csv)
# ============================================================

def run_merge_pipeline():
    target_path = find_file(DATA_DIR, ["food waste", "country"], exts=(".csv",))
    fbs_path = find_file(DATA_DIR, ["foodbalancesheets"])
    cp_path = find_file(DATA_DIR, ["consumerpriceindices"])

    qcl_path = find_file(DATA_DIR, ["production", "crops", "livestock"])
    tm_path = find_file(DATA_DIR, ["trade", "detailed", "matrix"])

    prod_idx_path = find_file(DATA_DIR, ["production_indices"])
    trade_idx_path = find_file(DATA_DIR, ["trade_indices"])

    if target_path is None:
        raise FileNotFoundError("Target CSV not found (food waste by country).")
    if fbs_path is None:
        raise FileNotFoundError("FoodBalanceSheets file not found.")
    if cp_path is None:
        raise FileNotFoundError("ConsumerPriceIndices file not found.")
    if qcl_path is None and prod_idx_path is None:
        raise FileNotFoundError("Neither QCL nor Production_Indices file found.")
    if tm_path is None and trade_idx_path is None:
        raise FileNotFoundError("Neither TM nor Trade_Indices file found.")

    print("Detected:")
    print("  Target:", target_path.name)
    print("  FBS   :", fbs_path.name)
    print("  CP    :", cp_path.name)
    print("  QCL   :", qcl_path.name if qcl_path else "(fallback)")
    print("  TM    :", tm_path.name if tm_path else "(fallback)")

    extract_dir = DATA_DIR / "faostat_extracted"
    fbs_csv = ensure_main_csv(fbs_path, extract_dir)
    cp_csv = ensure_main_csv(cp_path, extract_dir)
    qcl_csv = ensure_main_csv(qcl_path, extract_dir) if qcl_path else None
    tm_csv = ensure_main_csv(tm_path, extract_dir) if tm_path else None
    prod_idx_csv = ensure_main_csv(prod_idx_path, extract_dir) if prod_idx_path else None
    trade_idx_csv = ensure_main_csv(trade_idx_path, extract_dir) if trade_idx_path else None

    print("\nBuilding FBS...")
    fbs_feat = process_fbs(fbs_csv)
    print("FBS:", fbs_feat.shape)

    print("\nBuilding CP...")
    cp_feat = process_cp(cp_csv)
    print("CP:", cp_feat.shape)

    if qcl_csv:
        print("\nBuilding QCL...")
        prod_feat = process_qcl_detailed(qcl_csv)
    else:
        print("\nBuilding Production Indices (fallback)...")
        prod_feat = process_prod_indices(prod_idx_csv)
    print("Prod:", prod_feat.shape)

    if tm_csv:
        print("\nBuilding TM...")
        trade_feat = process_tm_detailed(tm_csv)
    else:
        print("\nBuilding Trade Indices (fallback)...")
        trade_feat = process_trade_indices(trade_idx_csv)
    print("Trade:", trade_feat.shape)

    features = fbs_feat.copy()
    for block in [cp_feat, prod_feat, trade_feat]:
        if not block.empty:
            features = features.merge(block, on="area_code", how="outer")
    features = features.rename(columns={"area_code": "M49 code"})

    target = pd.read_csv(target_path)
    target["M49 code"] = parse_m49(target["M49 code"])

    target_cols = [
        "Country", "M49 code", "Region", "Confidence in estimate",
        "combined figures (kg/capita/year)",
        "Household estimate (kg/capita/year)", "Retail estimate (kg/capita/year)", "Food service estimate (kg/capita/year)",
        "Household estimate (tonnes/year)", "Retail estimate (tonnes/year)", "Food service estimate (tonnes/year)",
    ]
    target_cols = [c for c in target_cols if c in target.columns]
    target = target[target_cols].copy()

    df = target.merge(features, on="M49 code", how="left")

    feat_cols = [c for c in df.columns if c.startswith(("fbs_", "cp_", "qcl_", "pi_", "tm_", "ti_"))]
    empty_cols = [c for c in feat_cols if df[c].isna().all()]
    if empty_cols:
        print("\nDropping all-NaN feature columns:", empty_cols)
        df = df.drop(columns=empty_cols)

    out_csv = DATA_DIR / "food_waste_faostat_model_input.csv"
    df.to_csv(out_csv, index=False)
    print("\nSaved:", out_csv)
    print("Shape:", df.shape)
    return df

merged_df = run_merge_pipeline()
merged_df.head()


# # B) EDA → CV → Tuning → Evaluation → Explainability → Risk Scoring)

# In[7]:


from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.compose import TransformedTargetRegressor

import joblib


# In[8]:


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def save_fig(name):
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    print('Saved:', out)

def plot_hist(series, title, fname, bins=30):
    s = pd.to_numeric(series, errors='coerce').dropna()
    plt.figure()
    plt.hist(s, bins=bins)
    plt.title(title)
    plt.xlabel(series.name if series.name else 'value')
    plt.ylabel('count')
    save_fig(fname)

def plot_box_by_category(df, y_col, cat_col, title, fname):
    plt.figure(figsize=(10, 6))
    groups, labels = [], []
    for c in sorted(df[cat_col].dropna().unique()):
        vals = pd.to_numeric(df.loc[df[cat_col] == c, y_col], errors='coerce').dropna()
        if len(vals) > 0:
            groups.append(vals.values)
            labels.append(str(c))
    if groups:
        plt.boxplot(groups, labels=labels, showfliers=False)
        plt.xticks(rotation=45, ha='right')
        plt.title(title)
        plt.ylabel(y_col)
        save_fig(fname)
    else:
        plt.close()

def plot_corr_heatmap(df, cols, title, fname):
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=90)
    plt.yticks(range(len(cols)), cols)
    plt.title(title)
    save_fig(fname)

def plot_pred_vs_actual(y_true, y_pred, title, fname):
    plt.figure()
    plt.scatter(y_true, y_pred, s=18)
    minv = np.nanmin([y_true.min(), y_pred.min()])
    maxv = np.nanmax([y_true.max(), y_pred.max()])
    plt.plot([minv, maxv], [minv, maxv])
    plt.title(title)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    save_fig(fname)

def plot_residuals(y_true, y_pred, title, fname):
    resid = y_true - y_pred
    plt.figure()
    plt.scatter(y_pred, resid, s=18)
    plt.axhline(0)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Residual (Actual - Pred)')
    save_fig(fname)

def plot_error_hist(y_true, y_pred, title, fname, bins=30):
    resid = (y_true - y_pred)
    plt.figure()
    plt.hist(resid, bins=bins)
    plt.title(title)
    plt.xlabel('Residual')
    plt.ylabel('Count')
    save_fig(fname)


# In[9]:


INPUT_MERGED = DATA_DIR / 'food_waste_faostat_model_input.csv'
df = pd.read_csv(INPUT_MERGED)
df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')
if 'M49 code' in df.columns:
    df = df.drop_duplicates(subset=['M49 code'], keep='first')

df_model = df[df[TARGET_COL].notna()].copy()

faostat_prefixes = ('fbs_', 'cp_', 'qcl_', 'tm_', 'pi_', 'ti_')
faostat_cols = [c for c in df_model.columns if c.startswith(faostat_prefixes)]
cat_cols = [c for c in ['Region', 'Confidence in estimate'] if c in df_model.columns]

X = df_model[faostat_cols + cat_cols].copy()
y = df_model[TARGET_COL].astype(float)

# drop all-NaN and constant features
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print('Dropping all-NaN features:', all_nan_cols)
    X = X.drop(columns=all_nan_cols)

constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
if constant_cols:
    print('Dropping constant features:', constant_cols)
    X = X.drop(columns=constant_cols)

cat_cols = [c for c in ['Region', 'Confidence in estimate'] if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

print('Rows:', len(df_model))
print('Numeric features:', len(num_cols))
print('Categorical features:', len(cat_cols))
X.shape


# In[10]:


plot_hist(df_model[TARGET_COL], 'Target Distribution: Combined Food Waste (kg/capita/year)', '01_target_distribution.png')

if 'Region' in df_model.columns:
    plot_box_by_category(df_model, TARGET_COL, 'Region', 'Combined Food Waste by Region', '02_boxplot_by_region.png')

if 'Confidence in estimate' in df_model.columns:
    plot_box_by_category(df_model, TARGET_COL, 'Confidence in estimate', 'Combined Food Waste by Confidence Level', '03_boxplot_by_confidence.png')

waste_cols = [c for c in df_model.columns if 'estimate (kg/capita/year)' in c.lower() or c.lower().startswith('combined figures')]
waste_cols = [c for c in waste_cols if c in df_model.columns]
if len(waste_cols) >= 3:
    plot_corr_heatmap(df_model, waste_cols, 'Correlation: Waste Metrics (kg/capita/year)', '04_corr_waste_metrics.png')

if 'Country' in df_model.columns:
    top = df_model[['Country', TARGET_COL]].dropna().sort_values(TARGET_COL, ascending=False).head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top['Country'][::-1], top[TARGET_COL][::-1])
    plt.title('Top 15 Countries by Actual Combined Food Waste')
    plt.xlabel(TARGET_COL)
    save_fig('05_top15_actual_countries.png')


# In[11]:


numeric = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])

categorical = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

pre = ColumnTransformer(transformers=[('num', numeric, num_cols), ('cat', categorical, cat_cols)], remainder='drop')


# In[12]:


models = {
    'Ridge': Ridge(random_state=RANDOM_STATE),
    'Lasso': Lasso(random_state=RANDOM_STATE, max_iter=50_000),
    'ElasticNet': ElasticNet(random_state=RANDOM_STATE, max_iter=50_000),
    'RandomForest': RandomForestRegressor(random_state=RANDOM_STATE, n_estimators=500, n_jobs=-1),
    'ExtraTrees': ExtraTreesRegressor(random_state=RANDOM_STATE, n_estimators=800, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=RANDOM_STATE),
    'HistGradientBoosting': HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
results = []

for name, model in models.items():
    reg = TransformedTargetRegressor(
        regressor=Pipeline(steps=[('pre', pre), ('model', model)]),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    scores = cross_validate(
        reg, X, y, cv=cv,
        scoring={'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'},
        return_train_score=False
    )
    results.append({
        'model': name,
        'cv_rmse_mean': -scores['test_rmse'].mean(),
        'cv_rmse_std': scores['test_rmse'].std(),
        'cv_mae_mean': -scores['test_mae'].mean(),
        'cv_r2_mean': scores['test_r2'].mean(),
    })

baseline = pd.DataFrame(results).sort_values('cv_rmse_mean')
baseline.to_csv(OUTPUT_DIR / 'model_results.csv', index=False)
baseline


# In[13]:


best_name = baseline.iloc[0]['model']
print('Best baseline:', best_name)

if best_name in ['ExtraTrees', 'RandomForest']:
    base = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1) if best_name == 'ExtraTrees' else RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(steps=[('pre', pre), ('model', base)])
    reg = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    param_dist = {
        'regressor__model__n_estimators': [400, 800, 1200],
        'regressor__model__max_depth': [None, 6, 10, 16, 24],
        'regressor__model__min_samples_split': [2, 5, 10],
        'regressor__model__min_samples_leaf': [1, 2, 4],
        'regressor__model__max_features': ['sqrt', 'log2', 0.5, 0.8],
    }
elif best_name == 'HistGradientBoosting':
    base = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[('pre', pre), ('model', base)])
    reg = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    param_dist = {
        'regressor__model__learning_rate': [0.03, 0.05, 0.08, 0.12],
        'regressor__model__max_depth': [None, 3, 5, 7],
        'regressor__model__max_leaf_nodes': [15, 31, 63],
        'regressor__model__min_samples_leaf': [10, 20, 30],
    }
else:
    base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline(steps=[('pre', pre), ('model', base)])
    reg = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1, check_inverse=False)
    param_dist = {
        'regressor__model__n_estimators': [400, 800, 1200],
        'regressor__model__max_depth': [None, 6, 10, 16, 24],
        'regressor__model__min_samples_split': [2, 5, 10],
        'regressor__model__min_samples_leaf': [1, 2, 4],
        'regressor__model__max_features': ['sqrt', 'log2', 0.5, 0.8],
    }

search = RandomizedSearchCV(
    reg,
    param_distributions=param_dist,
    n_iter=30,
    scoring='neg_root_mean_squared_error',
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
)

search.fit(X, y)
best_model = search.best_estimator_
print('Best CV RMSE:', -search.best_score_)
print('Best params:', search.best_params_)

joblib.dump(best_model, OUTPUT_DIR / 'best_model.pkl')
print('Saved:', OUTPUT_DIR / 'best_model.pkl')


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

metrics = {
    'RMSE': rmse(y_test, pred),
    'MAE': float(mean_absolute_error(y_test, pred)),
    'R2': float(r2_score(y_test, pred)),
    'N_test': int(len(y_test)),
}
metrics


# In[15]:


plot_pred_vs_actual(y_test, pred, 'Predicted vs Actual (Holdout)', '10_pred_vs_actual.png')
plot_residuals(y_test, pred, 'Residuals vs Predicted (Holdout)', '11_residuals_vs_pred.png')
plot_error_hist(y_test, pred, 'Residual Distribution (Holdout)', '12_residual_hist.png')

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X, y, cv=cv,
    scoring='neg_root_mean_squared_error',
    train_sizes=np.linspace(0.2, 1.0, 6),
    n_jobs=-1,
)
plt.figure()
plt.plot(train_sizes, -train_scores.mean(axis=1), label='train RMSE')
plt.plot(train_sizes, -test_scores.mean(axis=1), label='cv RMSE')
plt.title('Learning Curve (RMSE)')
plt.xlabel('Training set size')
plt.ylabel('RMSE')
plt.legend()
save_fig('13_learning_curve_rmse.png')


# In[16]:


best_model.fit(X, y)

r = permutation_importance(
    best_model, X, y,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
)
imp = pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)
imp.to_csv(OUTPUT_DIR / 'permutation_importance.csv', header=['importance'])
imp.head(20)


# In[17]:


top_k = 12
top = imp.head(top_k)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(top.index, top.values)
plt.title(f'Top {top_k} Drivers (Permutation Importance)')
plt.xlabel('Importance (Δ RMSE, higher = more important)')
save_fig('20_permutation_importance_top.png')

top_numeric = [c for c in imp.index if c in num_cols][:6]
if top_numeric:
    PartialDependenceDisplay.from_estimator(best_model, X, features=top_numeric)
    plt.suptitle('Partial Dependence (Top Numeric Features)')
    save_fig('21_partial_dependence_top_numeric.png')


# In[18]:


best_model.fit(X, y)
pred_all = best_model.predict(X)

scored = df_model.copy()
scored['predicted_combined_waste_kgcapyr'] = pred_all

threshold = np.nanpercentile(pred_all, 75)
scored['risk_flag_high'] = (scored['predicted_combined_waste_kgcapyr'] >= threshold).astype(int)
scored['pred_rank'] = scored['predicted_combined_waste_kgcapyr'].rank(ascending=False, method='dense').astype(int)

pred_path = OUTPUT_DIR / 'predictions_with_risk.csv'
scored.to_csv(pred_path, index=False)

dash_cols = [c for c in ['Country', 'M49 code', 'Region', 'Confidence in estimate'] if c in scored.columns]
dash_cols += ['predicted_combined_waste_kgcapyr', 'risk_flag_high', 'pred_rank']
dash = scored[dash_cols].copy()
dash_path = OUTPUT_DIR / 'dashboard_country_scores.csv'
dash.to_csv(dash_path, index=False)

print('Saved:', pred_path)
print('Saved:', dash_path)

dash.sort_values('pred_rank').head(20)


# In[19]:


if 'Country' in scored.columns:
    top = scored[['Country', 'predicted_combined_waste_kgcapyr']].dropna().sort_values(
        'predicted_combined_waste_kgcapyr', ascending=False
    ).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(top['Country'][::-1], top['predicted_combined_waste_kgcapyr'][::-1])
    plt.title('Top 15 Countries by Predicted Combined Food Waste')
    plt.xlabel('Predicted (kg/capita/year)')
    save_fig('30_top15_predicted_countries.png')

print('\nDONE ✅')
print('- Merged dataset:', (DATA_DIR / 'food_waste_faostat_model_input.csv').resolve())
print('- Outputs folder :', OUTPUT_DIR.resolve())
print('- Figures folder :', FIG_DIR.resolve())


# In[ ]:





# In[ ]:




