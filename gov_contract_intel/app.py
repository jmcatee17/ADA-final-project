import os
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify

from utils.usaspending_api import USASpendingAPI
from utils.inference import ContractInference

app = Flask(__name__)

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
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

def _extract_field(val, key="name"):
    """
    USASpending returns some fields as nested dicts, e.g.:
      PSC          → {'code': 'S206', 'description': 'HOUSEKEEPING- GUARD'}
      Agency fields→ {'id': 123, 'name': 'Dept of Defense', ...}
    This safely pulls out the requested key, or str(val) as fallback.
    """
    if isinstance(val, dict):
        return val.get(key) or val.get("name") or val.get("code") or str(val)
    return val


def _flatten_api_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten every column that the USASpending API may return as a nested dict.
    Keeps a separate *_description column for PSC hover tooltips.
    """
    # PSC: keep code as the display value, store description for hover
    if "PSC" in df.columns:
        df["PSC_description"] = df["PSC"].apply(
            lambda x: x.get("description", "") if isinstance(x, dict) else ""
        )
        df["PSC"] = df["PSC"].apply(
            lambda x: x.get("code", "UNKNOWN") if isinstance(x, dict) else str(x)
        )

    # Agency / sub-agency columns
    for col in ("Awarding Agency", "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.get("name", "UNKNOWN") if isinstance(x, dict) else x
            )

    # Recipient Name
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

    # Flatten nested dict columns FIRST
    df = _flatten_api_columns(df)

    # Award Amount
    if "Award Amount" in df.columns:
        df["Award Amount"] = _to_numeric_amount(df["Award Amount"])
    else:
        df["Award Amount"] = 0.0

    # PSC nulls
    if "PSC" in df.columns:
        df["PSC"] = (
            df["PSC"].fillna("UNKNOWN").astype(str).str.strip()
            .replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
        )

    # Recipient Name nulls
    if "Recipient Name" in df.columns:
        df["Recipient Name"] = df["Recipient Name"].fillna("UNKNOWN").astype(str).str.strip()

    # Dates
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df = df.dropna(subset=["Start Date"])
    return df


def _empty_fig(title: str):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[dict(text="No data available", showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          font=dict(size=16))],
    )
    return fig


def _dollar_axis(fig):
    """Apply $X.XM / $X.XB tick formatting to the x-axis."""
    fig.update_layout(
        xaxis_tickprefix="$",
        xaxis_tickformat=",.2s",   # e.g. 1.2B, 450M
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
    filters        = request.json or {}
    target_agencies = filters.get("agencies", ["Department of Homeland Security"])
    start_date     = filters.get("start_date", "2023-10-01")
    end_date       = filters.get("end_date",   "2024-09-30")

    df = usa_api.scrape_contracts(start_date, end_date, target_agencies)
    df = clean_data(df)

    if df.empty:
        return jsonify({"error": "No data returned for this selection."}), 404

    # Filter strictly to the requested date window (avoids stale cache bleed)
    mask = (df["Start Date"] >= start_date) & (df["Start Date"] <= end_date)
    df   = df[mask]
    if df.empty:
        return jsonify({"error": "No data in selected date range."}), 404

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if os.path.exists(CACHE_PATH):
        existing = pd.read_csv(CACHE_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        if "Award ID" in combined.columns:
            combined = combined.drop_duplicates(subset=["Award ID"], keep="last")
        df = combined
    df.to_csv(CACHE_PATH, index=False)

    # Re-parse after concat — CSV round-trip turns Timestamps back into strings
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
    df = df.dropna(subset=["Start Date"])

    # ------------------------------------------------------------------
    # Chart 1 — PSC activity (code label + description on hover)
    # ------------------------------------------------------------------
    psc_df = df[df["PSC"] != "UNKNOWN"].copy() if (df["PSC"] != "UNKNOWN").any() else df.copy()

    psc_hover = (
        psc_df.groupby("PSC")
        .agg(Count=("PSC", "count"), Description=("PSC_description", "first"))
        .reset_index()
        .sort_values("Count", ascending=False)
        .head(15)
        .sort_values("Count")           # ascending for horizontal bar
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
    # Chart 2 — Top 10 contractors by dollar spend
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
        fig_contractor.update_layout(
            yaxis=dict(type="category"),
            xaxis_tickprefix="$",
            xaxis_tickformat=",.2s",
        )

    # ------------------------------------------------------------------
    # Chart 3 — Award value distribution (histogram handles sparse data)
    # ------------------------------------------------------------------
    dist_df = df[df["Award Amount"] > 0].copy()

    if dist_df.empty:
        fig_dist = _empty_fig("Award Value Distribution")
    else:
        fig_dist = px.histogram(
            dist_df,
            x="Award Amount",
            nbins=30,
            log_x=True,
            title="Award Value Distribution",
            labels={"Award Amount": "Award Amount ($)"},
        )
        fig_dist.update_layout(
            xaxis_tickprefix="$",
            xaxis_tickformat=",.2s",
        )

    # ------------------------------------------------------------------
    # Chart 4 — Monthly spending trend (scoped to selected date window)
    # ------------------------------------------------------------------
    df_time = (
        df.set_index("Start Date")
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
        )
        fig_time.update_layout(
            yaxis_tickprefix="$",
            yaxis_tickformat=",.2s",
        )

    return json.dumps(
        {
            "psc_chart":        fig_psc,
            "contractor_chart": fig_contractor,
            "dist_chart":       fig_dist,
            "time_chart":       fig_time,
        },
        cls=plotly.utils.PlotlyJSONEncoder,
    )


@app.route("/api/predict-amount", methods=["POST"])
def predict_amount():
    data   = request.json or {}
    result = predictor.predict_all(data)
    return jsonify(result)


@app.route("/api/predict-mod-risk", methods=["POST"])
def predict_mod_risk():
    if not os.path.exists(CACHE_PATH):
        return jsonify({"error": "No cached data. Click 'Sync & Analyze' first."}), 400

    df_c = pd.read_csv(CACHE_PATH)
    df_c["Start Date"] = pd.to_datetime(df_c["Start Date"])
    
    # Get the current dynamic timestamp normalized to midnight
    current_date = pd.Timestamp.now().floor('D') 
    one_year = pd.Timedelta(days=365)
    
    # FIX: Added missing closing parenthesis/bracket for the conditional mask
    df_filter = df_c[(df_c["Start Date"] >= current_date - one_year)]
        
    df = df_filter.head(20)

    if "Award Amount" in df.columns:
        df["Award Amount"] = _to_numeric_amount(df["Award Amount"])

    processed        = predictor.engineer.prepare_for_inference(df)
    df["risk_score"] = predictor.risk_model.predict_proba(processed)[:, 1]

    df["Award ID"] = (
        df["Award ID"].fillna("Unknown").astype(str)
        if "Award ID" in df.columns
        else pd.Series([f"Contract {i}" for i in range(len(df))])
    )

    # Create a quick, truncated version of the description for the hover text
    df["Short Description"] = df["Description"].str[:30] + "..."
    
    fig_risk = px.bar(
        df.sort_values("risk_score"),
        x="risk_score", 
        y="Award ID", 
        orientation="h",
        color="risk_score", 
        color_continuous_scale="Reds",
        title="Modification Risk Forecast",
        labels={
            "risk_score": "Risk Probability",
            "Short Description": "Description"  # Renames the hover label cleanly
        },
        hover_data={
            "risk_score": ":.2f",
            "Award ID": True,
            "Short Description": True  # Pulls the 20-character version into the hover
        }
    )
    fig_risk.update_layout(xaxis_range=[0, 1], yaxis=dict(type="category"))

    return json.dumps({"risk_bar": fig_risk}, cls=plotly.utils.PlotlyJSONEncoder)


if __name__ == "__main__":
    app.run(debug=True, port=5001)