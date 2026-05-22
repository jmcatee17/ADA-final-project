import os

# Prevent XGBoost + PyTorch OpenMP collision on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify

from utils.usaspending_api import USASpendingAPI
from utils.inference import ContractInference

app = Flask(__name__)

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH       = os.path.join(BASE_DIR, "data", "contracts_cache.csv")
CACHE_META_PATH  = os.path.join(BASE_DIR, "data", "cache_meta.json")
CACHE_START_DATE = "2023-10-01"

NAICS_DESCRIPTIONS = {
    "518210": "Data Processing, Hosting & Related Services",
    "541511": "Custom Computer Programming Services",
    "541512": "Computer Systems Design Services",
    "541519": "Other Computer Related Services",
    "513210": "Software Publishers",
    "511210": "Software Publishers (Legacy)",
    "334111": "Electronic Computer Manufacturing",
    "UNKNOWN": "Unknown / Not Specified",
}

ALL_AGENCY_NAMES = [
    "Commodity Futures Trading Commission",
    "Department of Agriculture",
    "Department of Commerce",
    "Department of Defense",
    "Department of Energy",
    "Department of Health and Human Services",
    "Department of Homeland Security",
    "Department of Justice",
    "Department of State",
    "Department of Transportation",
    "Department of the Interior",
    "Department of the Treasury",
    "Securities and Exchange Commission",
]

MODEL_PATHS = {
    "regressor":  os.path.join(BASE_DIR, "models", "xgb_amount_model.joblib"),
    "classifier": os.path.join(BASE_DIR, "models", "xgb_modification_model.joblib"),
}

usa_api   = USASpendingAPI()
predictor = ContractInference(
    reg_path=MODEL_PATHS["regressor"],
    clf_path=MODEL_PATHS["classifier"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flatten_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten every column that the USASpending API may return as a nested dict.
    Keeps a separate PSC_description column for hover tooltips.
    """
    if "PSC" in df.columns:
        df["PSC_description"] = df["PSC"].apply(
            lambda x: x.get("description", "") if isinstance(x, dict) else ""
        )
        df["PSC"] = df["PSC"].apply(
            lambda x: x.get("code", "UNKNOWN") if isinstance(x, dict) else str(x)
        )

    for col in ("Awarding Agency", "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.get("name", "UNKNOWN") if isinstance(x, dict) else x
            )

    if "Recipient Name" in df.columns:
        df["Recipient Name"] = df["Recipient Name"].apply(
            lambda x: x.get("name", "UNKNOWN") if isinstance(x, dict) else x
        )

    return df


def _to_numeric_amount(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.replace(r"[$,\s]", "", regex=True)
        .replace({"nan": "0", "None": "0", "": "0"})
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Flatten nested dict columns first
    df = _flatten_api_columns(df)

    if "Award Amount" in df.columns:
        df["Award Amount"] = _to_numeric_amount(df["Award Amount"])
    else:
        df["Award Amount"] = 0.0

    if "PSC" in df.columns:
        df["PSC"] = (
            df["PSC"].fillna("UNKNOWN").astype(str).str.strip()
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
        )

    if "PSC_description" not in df.columns:
        df["PSC_description"] = ""

    if "NAICS Code" in df.columns and "naics_code" not in df.columns:
        df["naics_code"] = df["NAICS Code"].fillna("UNKNOWN").astype(str).str.strip()
    elif "naics_code" not in df.columns:
        df["naics_code"] = "UNKNOWN"

    if "Recipient Name" in df.columns:
        df["Recipient Name"] = df["Recipient Name"].fillna("UNKNOWN").astype(str).str.strip()

    df["Start Date"] = pd.to_datetime(df["Start Date"], format="mixed", errors="coerce")
    df = df.dropna(subset=["Start Date"])
    return df


def _load_and_prepare_cache() -> pd.DataFrame:
    """Read cache CSV and ensure key columns are correctly typed."""
    df = pd.read_csv(CACHE_PATH, low_memory=False)
    df["Start Date"] = pd.to_datetime(df["Start Date"], format="mixed", errors="coerce")
    if "Award Amount" in df.columns:
        df["Award Amount"] = _to_numeric_amount(df["Award Amount"])
    if "PSC_description" not in df.columns:
        df["PSC_description"] = ""
    if "PSC" not in df.columns:
        df["PSC"] = "UNKNOWN"
    return df


def _append_to_cache(new_df: pd.DataFrame) -> None:
    """Deduplicate on Award ID and persist to cache CSV."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if os.path.exists(CACHE_PATH):
        try:
            existing = pd.read_csv(CACHE_PATH)
            combined = pd.concat([existing, new_df], ignore_index=True)
            if "Award ID" in combined.columns:
                combined = combined.drop_duplicates(subset=["Award ID"], keep="last")
            combined.to_csv(CACHE_PATH, index=False)
        except Exception as e:
            print(f"Cache append error: {e}")
            new_df.to_csv(CACHE_PATH, index=False)
    else:
        new_df.to_csv(CACHE_PATH, index=False)


def _load_meta() -> dict:
    if os.path.exists(CACHE_META_PATH):
        try:
            with open(CACHE_META_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_meta(meta: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_META_PATH), exist_ok=True)
    with open(CACHE_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


def _empty_fig(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[dict(
            text="No data available", showarrow=False,
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=16),
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/sync", methods=["POST"])
def sync_cache():
    """
    Check staleness and fetch only the missing date range for every agency.
    Called by the frontend on every 'Sync & Analyze' click; returns immediately
    when the cache is already current (same calendar day).
    """
    meta      = _load_meta()
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    today_dt  = pd.Timestamp(today_str)

    last_updated = meta.get("last_updated")

    if last_updated and last_updated >= today_str:
        return jsonify({
            "status":       "current",
            "last_updated": last_updated,
            "message":      "Cache is already up to date.",
        })

    fetch_start = (
        (pd.Timestamp(last_updated) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        if last_updated
        else CACHE_START_DATE
    )

    total_new = 0
    for agency in ALL_AGENCY_NAMES:
        try:
            raw    = usa_api.scrape_contracts(fetch_start, today_str, [agency])
            new_df = clean_data(raw)
            if not new_df.empty:
                new_df = new_df[
                    (new_df["Start Date"] >= pd.Timestamp(fetch_start)) &
                    (new_df["Start Date"] <= today_dt)
                ]
                if not new_df.empty:
                    _append_to_cache(new_df)
                    total_new += len(new_df)
        except Exception as e:
            print(f"Sync error for {agency}: {e}")

    meta["last_updated"]   = today_str
    meta["covered_start"]  = CACHE_START_DATE
    _save_meta(meta)

    return jsonify({
        "status":       "updated",
        "last_updated": today_str,
        "new_records":  total_new,
    })


@app.route("/api/market-data", methods=["POST"])
def market_data():
    filters         = request.json or {}
    target_agencies = filters.get("agencies", ["Department of Homeland Security"])
    start_date      = filters.get("start_date", CACHE_START_DATE)
    end_date        = filters.get("end_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    naics_filter    = filters.get("naics", "").strip()

    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    # ------------------------------------------------------------------
    # Always serve from cache — /api/sync handles all API interaction.
    # ------------------------------------------------------------------
    if not os.path.exists(CACHE_PATH):
        return jsonify({"error": "Cache is empty. Click 'Sync & Analyze' to initialize data."}), 404

    try:
        cached = _load_and_prepare_cache().dropna(subset=["Start Date"])
    except Exception as e:
        return jsonify({"error": f"Cache read error: {e}"}), 500

    if "Awarding Agency" in cached.columns:
        df = cached[
            cached["Awarding Agency"].isin(target_agencies) &
            (cached["Start Date"] >= start_dt) &
            (cached["Start Date"] <= end_dt)
        ].copy()
    else:
        df = cached[
            (cached["Start Date"] >= start_dt) &
            (cached["Start Date"] <= end_dt)
        ].copy()

    if df.empty:
        return jsonify({
            "error": (
                f"No cached data for {target_agencies[0]} between {start_date} and {end_date}. "
                "Click 'Sync & Analyze' to fetch it."
            )
        }), 404

    df["Award Amount"] = _to_numeric_amount(df["Award Amount"]) if "Award Amount" in df.columns else 0.0

    if "PSC_description" not in df.columns:
        df["PSC_description"] = ""
    if "PSC" not in df.columns:
        df["PSC"] = "UNKNOWN"

    # Optional keyword / NAICS filter (applied after cache hit too)
    if naics_filter:
        for col in ("PSC", "naics_code", "Description"):
            if col in df.columns:
                m = df[col].astype(str).str.contains(naics_filter, case=False, na=False)
                if m.any():
                    df = df[m]
                    break

    # ------------------------------------------------------------------
    # Chart 1 — PSC activity (count per code, hover shows description)
    # ------------------------------------------------------------------
    psc_df = df

    psc_hover = (
        psc_df.groupby("PSC")
        .agg(Count=("PSC", "count"), Description=("PSC_description", "first"))
        .reset_index()
        .sort_values("Count", ascending=False)
        .head(15)
        .sort_values("Count", ascending=True)  # largest renders at top in horizontal bar
    )
    if psc_hover.empty:
        fig_psc = _empty_fig("Market Activity by Category (PSC Code)")
    else:
        counts_list = psc_hover["Count"].tolist()
        psc_list    = psc_hover["PSC"].tolist()
        desc_list   = psc_hover["Description"].tolist()
        fig_psc = go.Figure(go.Bar(
            x=counts_list,
            y=psc_list,
            orientation="h",
            customdata=[[d] for d in desc_list],
            hovertemplate="<b>%{y}</b><br>Count: %{x}<br>%{customdata[0]}<extra></extra>",
            marker_color="#636efa",
        ))
        fig_psc.update_layout(
            title="Market Activity by Category (PSC Code)",
            xaxis_title="Count",
            yaxis_title="PSC",
            yaxis=dict(type="category"),
        )

    # ------------------------------------------------------------------
    # Chart 2 — Top 10 contractors by total dollar spend
    # ------------------------------------------------------------------
    top_con = (
        df[df["Award Amount"] > 0]
        .groupby("Recipient Name", as_index=False)["Award Amount"]
        .sum()
        .sort_values("Award Amount")
        .tail(10)
    )
    if top_con.empty:
        fig_contractor = _empty_fig("Top 10 Contractors by Total Spend ($)")
    else:
        fig_contractor = go.Figure(go.Bar(
            x=top_con["Award Amount"].tolist(),
            y=top_con["Recipient Name"].tolist(),
            orientation="h",
            marker_color="#636efa",
            hovertemplate="<b>%{y}</b><br>Total Spend: $%{x:,.0f}<extra></extra>",
        ))
        fig_contractor.update_layout(
            title="Top 10 Contractors by Total Spend ($)",
            xaxis_title="Total Award ($)",
            yaxis=dict(type="category"),
            xaxis_tickprefix="$",
            xaxis_tickformat=",",
            xaxis_exponentformat="none",
        )

    # ------------------------------------------------------------------
    # Chart 3 — Award value distribution
    # Bins are pre-computed in Python with np.histogram so Plotly.js
    # receives a plain bar trace — avoids histogram binning/rendering
    # quirks when the figure is serialised to JSON and re-hydrated.
    # ------------------------------------------------------------------
    dist_df = df[df["Award Amount"] > 0].copy()

    if dist_df.empty:
        fig_dist = _empty_fig("Award Value Distribution")
    else:
        log_vals   = np.log10(dist_df["Award Amount"].clip(lower=1))
        bin_edges  = [0, 3, 4, 5, 6, 7, 8, 9, 10]
        bin_labels = ["<$1K", "$1K–$10K", "$10K–$100K", "$100K–$1M",
                      "$1M–$10M", "$10M–$100M", "$100M–$1B", ">$1B"]
        counts, _  = np.histogram(log_vals, bins=bin_edges)
        fig_dist   = go.Figure(go.Bar(
            x=bin_labels,
            y=counts.tolist(),
            marker_color="#636efa",
            hovertemplate="<b>%{x}</b><br># Contracts: %{y}<extra></extra>",
        ))
        fig_dist.update_layout(
            title="Award Value Distribution",
            xaxis_title="Award Amount ($)",
            yaxis_title="# Contracts",
            bargap=0.15,
        )

    # ------------------------------------------------------------------
    # Chart 4 — Monthly spending trend
    # Filter to positive amounts before resampling so $0-valued rows
    # don't pollute the monthly totals.
    # ------------------------------------------------------------------
    df_time = (
        df[df["Award Amount"] > 0]
        .set_index("Start Date")
        .resample("ME")["Award Amount"]
        .sum()
        .reset_index()
    )

    if df_time.empty or df_time["Award Amount"].sum() == 0:
        fig_time = _empty_fig("Monthly Spending Trend")
    else:
        dates_iso  = df_time["Start Date"].dt.strftime("%Y-%m-%d").tolist()
        amounts    = df_time["Award Amount"].tolist()
        fig_time   = go.Figure(go.Scatter(
            x=dates_iso,
            y=amounts,
            mode="lines+markers",
            hovertemplate="<b>%{x}</b><br>Spend: $%{y:,.0f}<extra></extra>",
        ))
        fig_time.update_layout(
            title="Monthly Spending Trend",
            xaxis_title="Month",
            yaxis_title="Total Spend ($)",
            yaxis_tickprefix="$",
            yaxis_tickformat=",.2s",
        )

    return jsonify({
        "psc_chart":        json.loads(fig_psc.to_json()),
        "contractor_chart": json.loads(fig_contractor.to_json()),
        "dist_chart":       json.loads(fig_dist.to_json()),
        "time_chart":       json.loads(fig_time.to_json()),
        "from_cache":       True,
    })


@app.route("/api/predict-amount", methods=["POST"])
def predict_amount():
    data   = request.json or {}
    result = predictor.predict_all(data)
    return jsonify(result)


@app.route("/api/predict-mod-risk", methods=["POST"])
def predict_mod_risk():
    if not os.path.exists(CACHE_PATH):
        return jsonify({"error": "No cached data. Click 'Sync & Analyze' first."}), 400

    try:
        df_c = _load_and_prepare_cache()
    except Exception as e:
        return jsonify({"error": f"Cache read failed: {e}"}), 500

    df_c = df_c.dropna(subset=["Start Date"])

    if df_c.empty:
        return jsonify({"error": "Cache is empty. Click 'Sync & Analyze' first."}), 400

    six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)
    df = df_c[df_c["Start Date"] >= six_months_ago].copy()

    if df.empty:
        return jsonify({"error": "No contracts found in the last 6 months."}), 400

    # 2. Run predictions on the full dataset
    processed = predictor.engineer.prepare_for_inference(df)
    try:
        raw_scores = predictor.risk_model.predict_proba(processed)[:, 1]
    except AttributeError:
        raw_scores = predictor.risk_model.predict(processed)
    except Exception:
        raw_scores = predictor.risk_model.predict(processed)
    raw_scores = np.array(raw_scores, dtype=float)
    raw_min, raw_max = float(raw_scores.min()), float(raw_scores.max())
    if raw_max > 1.0 or raw_min < 0.0:
        if raw_max > raw_min:
            raw_scores = (raw_scores - raw_min) / (raw_max - raw_min)
        else:
            raw_scores = np.zeros_like(raw_scores)
    df["risk_score"] = raw_scores

    if "Award ID" not in df.columns:
        df["Award ID"] = [f"Contract {i}" for i in range(len(df))]
    else:
        df["Award ID"] = df["Award ID"].fillna("Unknown").astype(str)

    df["Short Description"] = df["Description"].fillna("").str[:40] + "..."

    # 3. Sort by highest risk FIRST, then take the top 20
    df_sorted = (
        df.sort_values("risk_score", ascending=False)
        .head(20)
        .sort_values("risk_score", ascending=True)  # largest renders at top in horizontal bar
        .reset_index(drop=True)
    )
    risk_vals  = df_sorted["risk_score"].tolist()
    desc_vals  = df_sorted["Short Description"].tolist()
    award_ids  = df_sorted["Award ID"].tolist()

    # Use go.Bar directly so we control the hovertemplate precisely.
    # px.bar with color=x causes Plotly to emit %{marker.color} in the
    # template, which never formats correctly.
    fig_risk = go.Figure(go.Bar(
        x=risk_vals,
        y=award_ids,
        orientation="h",
        marker=dict(
            color=risk_vals,
            colorscale="RdYlGn_r",
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(title="Risk", tickformat=".0%"),
        ),
        customdata=list(zip(desc_vals)),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Risk Probability: %{x:.1%}<br>"
            "Desc: %{customdata[0]}"
            "<extra></extra>"
        ),
    ))
    six_mo_label = six_months_ago.strftime("%b %Y")
    fig_risk.update_layout(
        title=f"Top 20 Contracts by Modification Risk · Last 6 Months ({six_mo_label} – Present)",
        xaxis=dict(title="Risk Probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(type="category"),
    )

    # ------------------------------------------------------------------
    # NAICS grouped bar chart — predicted modifications vs non-modifications
    # ------------------------------------------------------------------
    df["predicted_mod"] = (df["risk_score"] >= 0.5).astype(int)

    if "naics_code" not in df.columns:
        df["naics_code"] = "UNKNOWN"
    else:
        df["naics_code"] = df["naics_code"].fillna("UNKNOWN").astype(str).str.strip()
        df["naics_code"] = df["naics_code"].replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})

    naics_grp = (
        df.groupby("naics_code")["predicted_mod"]
        .agg(Modifications="sum", Total="count")
        .reset_index()
    )
    naics_grp["Non_Modifications"] = naics_grp["Total"] - naics_grp["Modifications"]
    naics_grp = naics_grp.sort_values("Total", ascending=False).head(12)

    naics_codes  = naics_grp["naics_code"].tolist()
    mod_counts   = naics_grp["Modifications"].tolist()
    nomod_counts = naics_grp["Non_Modifications"].tolist()
    hover_descs  = [NAICS_DESCRIPTIONS.get(c, c) for c in naics_codes]

    fig_naics = go.Figure()
    fig_naics.add_trace(go.Bar(
        name="Predicted Modification",
        x=naics_codes,
        y=mod_counts,
        marker_color="#ef4444",
        customdata=[[d] for d in hover_descs],
        hovertemplate="<b>NAICS %{x}</b><br>%{customdata[0]}<br>Predicted Modifications: %{y}<extra></extra>",
    ))
    fig_naics.add_trace(go.Bar(
        name="Predicted No Modification",
        x=naics_codes,
        y=nomod_counts,
        marker_color="#22c55e",
        customdata=[[d] for d in hover_descs],
        hovertemplate="<b>NAICS %{x}</b><br>%{customdata[0]}<br>Predicted No Modification: %{y}<extra></extra>",
    ))
    fig_naics.update_layout(
        title=f"Modification Predictions by NAICS Code · Last 6 Months ({six_mo_label} – Present)",
        barmode="group",
        xaxis_title="NAICS Code",
        yaxis_title="Contract Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return jsonify({
        "risk_bar":   json.loads(fig_risk.to_json()),
        "naics_chart": json.loads(fig_naics.to_json()),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)