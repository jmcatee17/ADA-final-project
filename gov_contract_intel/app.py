import os
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, "data", "contracts_cache.csv")

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

    if "Recipient Name" in df.columns:
        df["Recipient Name"] = df["Recipient Name"].fillna("UNKNOWN").astype(str).str.strip()

    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df = df.dropna(subset=["Start Date"])
    return df


def _load_and_prepare_cache() -> pd.DataFrame:
    """Read cache CSV and ensure key columns are correctly typed."""
    df = pd.read_csv(CACHE_PATH)
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
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


@app.route("/api/market-data", methods=["POST"])
def market_data():
    filters         = request.json or {}
    target_agencies = filters.get("agencies", ["Department of Homeland Security"])
    start_date      = filters.get("start_date", "2023-10-01")
    # Default end date is today so the UI always reflects the current date
    end_date        = filters.get("end_date", pd.Timestamp.now().strftime("%Y-%m-%d"))
    naics_filter    = filters.get("naics", "").strip()

    start_dt = pd.Timestamp(start_date)
    end_dt   = pd.Timestamp(end_date)

    # ------------------------------------------------------------------
    # 1. Cache-first: serve cached rows + identify date gaps to fetch
    # ------------------------------------------------------------------
    df           = pd.DataFrame()
    from_cache   = False
    fetch_ranges = []  # list of (fetch_start_str, fetch_end_str) gaps to pull from API

    if os.path.exists(CACHE_PATH):
        try:
            cached = _load_and_prepare_cache().dropna(subset=["Start Date"])

            agency_cached = (
                cached[cached["Awarding Agency"].isin(target_agencies)]
                if "Awarding Agency" in cached.columns
                else cached
            )

            if not agency_cached.empty:
                cached_min = agency_cached["Start Date"].min()
                cached_max = agency_cached["Start Date"].max()

                # Serve whatever is already cached within the requested window
                hit = agency_cached[
                    (agency_cached["Start Date"] >= start_dt) &
                    (agency_cached["Start Date"] <= end_dt)
                ]
                if not hit.empty:
                    df         = hit.copy()
                    from_cache = True

                # Identify gaps not yet covered by the cache
                if end_dt > cached_max:
                    # Need newer data: from day after cached max → requested end
                    gap_start = (cached_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    fetch_ranges.append((gap_start, end_date))
                if start_dt < cached_min:
                    # Need older data: from requested start → day before cached min
                    gap_end = (cached_min - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    fetch_ranges.append((start_date, gap_end))
                # If start_dt >= cached_min and end_dt <= cached_max: fully covered
            else:
                # No cached data for this agency at all
                fetch_ranges.append((start_date, end_date))

        except Exception as e:
            print(f"Cache read error: {e}")
            fetch_ranges.append((start_date, end_date))
    else:
        fetch_ranges.append((start_date, end_date))

    # ------------------------------------------------------------------
    # 2. Fetch only the missing date ranges; persist and merge with cache hit
    # ------------------------------------------------------------------
    for fetch_start, fetch_end in fetch_ranges:
        raw    = usa_api.scrape_contracts(fetch_start, fetch_end, target_agencies)
        new_df = clean_data(raw)

        if not new_df.empty:
            fs_dt  = pd.Timestamp(fetch_start)
            fe_dt  = pd.Timestamp(fetch_end)
            new_df = new_df[(new_df["Start Date"] >= fs_dt) & (new_df["Start Date"] <= fe_dt)]

            if not new_df.empty:
                _append_to_cache(new_df)

                # Include any portion that falls within the requested display window
                portion = new_df[
                    (new_df["Start Date"] >= start_dt) &
                    (new_df["Start Date"] <= end_dt)
                ]
                if not portion.empty:
                    df = (
                        pd.concat([df, portion], ignore_index=True)
                        if not df.empty else portion.copy()
                    )
                    from_cache = False

    if df.empty:
        return jsonify({"error": "No data returned for this selection."}), 404

    # Re-parse after potential CSV round-trip
    # df["Start Date"]  = pd.to_datetime(df["Start Date"], errors="coerce")
    # df                = df.dropna(subset=["Start Date"])
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
    )

    if psc_hover.empty:
        fig_psc = _empty_fig("Market Activity by Category (PSC Code)")
    else:
        fig_psc = px.bar(
            psc_hover,
            x="Count", y="PSC", orientation="h",
            hover_data={"Description": True, "Count": True},
            title="Market Activity by Category (PSC Code)",
        )
        fig_psc.update_layout(yaxis=dict(type="category"))

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
        fig_contractor = px.bar(
            top_con,
            x="Award Amount", y="Recipient Name", orientation="h",
            title="Top 10 Contractors by Total Spend ($)",
            labels={"Award Amount": "Total Award ($)"},
        )
        fig_contractor.update_traces(
            hovertemplate="<b>%{y}</b><br>Total Spend: $%{x:,.0f}<extra></extra>"
        )
        fig_contractor.update_layout(
            yaxis=dict(type="category"),
            xaxis_tickprefix="$",
            xaxis_tickformat=",",
            xaxis_exponentformat="none"
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
        fig_time = px.line(
            df_time, x="Start Date", y="Award Amount",
            title="Monthly Spending Trend",
            labels={"Award Amount": "Total Spend ($)", "Start Date": "Month"},
            markers=True,
        )
        fig_time.update_traces(
            hovertemplate="<b>%{x|%b %Y}</b><br>Spend: $%{y:,.0f}<extra></extra>"
        )
        fig_time.update_layout(
            yaxis_tickprefix="$",
            yaxis_tickformat=",.2s",
        )

    return jsonify({
        "psc_chart":        json.loads(fig_psc.to_json()),
        "contractor_chart": json.loads(fig_contractor.to_json()),
        "dist_chart":       json.loads(fig_dist.to_json()),
        "time_chart":       json.loads(fig_time.to_json()),
        "from_cache":       from_cache,
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

    # Use the most-recent 500 contracts regardless of age so that older
    # bulk-ingest batches are still included when recent data is sparse.
    df = df_c.sort_values("Start Date", ascending=False).head(500).copy()

    # 2. Run predictions on the full dataset
    processed = predictor.engineer.prepare_for_inference(df)
    try:
        raw_scores = predictor.risk_model.predict_proba(processed)[:, 1]
    except AttributeError:
        # Fallback if model is a regressor without predict_proba
        raw_scores = predictor.risk_model.predict(processed)
    df["risk_score"] = np.clip(raw_scores, 0.0, 1.0)

    if "Award ID" not in df.columns:
        df["Award ID"] = [f"Contract {i}" for i in range(len(df))]
    else:
        df["Award ID"] = df["Award ID"].fillna("Unknown").astype(str)

    df["Short Description"] = df["Description"].fillna("").str[:40] + "..."

    # 3. Sort by highest risk FIRST, then take the top 20
    df_sorted = df.sort_values("risk_score", ascending=False).head(20).reset_index(drop=True)
    risk_vals  = df_sorted["risk_score"].values
    desc_vals  = df_sorted["Short Description"].values

    # Use go.Bar directly so we control the hovertemplate precisely.
    # px.bar with color=x causes Plotly to emit %{marker.color} in the
    # template, which never formats correctly.
    fig_risk = go.Figure(go.Bar(
        x=risk_vals,
        y=df_sorted["Award ID"].values,
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
    fig_risk.update_layout(
        title="Modification Risk Forecast",
        xaxis=dict(title="Risk Probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(type="category"),
    )

    return json.dumps({"risk_bar": fig_risk}, cls=plotly.utils.PlotlyJSONEncoder)


if __name__ == "__main__":
    app.run(debug=True, port=5001)