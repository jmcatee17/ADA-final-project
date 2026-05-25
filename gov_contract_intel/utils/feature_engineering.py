import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

_DEMOCRAT_YEARS = {
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
    2021, 2022, 2023, 2024,
}


class ContractFeatureEngineer:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_model_path = os.path.join(base_dir, "models", "all-MiniLM-L6-v2")

        if os.path.exists(local_model_path):
            print("--- Loading Embedding Model from Local Disk ---")
            self.embedder = SentenceTransformer(local_model_path)
        else:
            print("--- Local Model Not Found: Downloading from Web ---")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.embedding_col_names = [f"v_{i}" for i in range(384)]

        # All columns the pipeline's ColumnTransformer may reference.
        # We must guarantee every one exists — the pipeline selects internally.
        self.required_cols = (
            self.embedding_col_names
            + [
                "Award Amount",
                "Awarding Agency",
                "Awarding Sub Agency",
                "Funding Agency",
                "Funding Sub Agency",
                "duration_days",
                "start_year",
                "is_democrat",
                "naics_code",
                "market_share",
                "Place of Performance Country Code",
                "Place of Performance State Code",
                # Derived boolean flags
                "award_not_fund_agency",
                "award_not_fund_sub_agency",
            ]
        )

    def prepare_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ------------------------------------------------------------------
        # Description → 384-dim embeddings
        # ------------------------------------------------------------------
        descriptions = df["Description"].fillna("No description provided").tolist()
        embeddings   = self.embedder.encode(descriptions)
        emb_df       = pd.DataFrame(embeddings, columns=self.embedding_col_names, index=df.index)
        df           = pd.concat([df, emb_df], axis=1)

        # ------------------------------------------------------------------
        # Award Amount
        # ------------------------------------------------------------------
        if "Award Amount" in df.columns:
            df["Award Amount"] = (
                df["Award Amount"].astype(str)
                .str.replace(r"[$,]", "", regex=True)
                .replace({"nan": "0", "None": "0", "": "0"})
                .pipe(pd.to_numeric, errors="coerce")
                .fillna(0.0)
            )
        else:
            df["Award Amount"] = 0.0

        # ------------------------------------------------------------------
        # Date-derived: start_year, is_democrat, duration_days
        # ------------------------------------------------------------------
        if "Start Date" in df.columns:
            start = pd.to_datetime(df["Start Date"], format="mixed", errors="coerce")
        else:
            start = pd.Series([pd.Timestamp.now()] * len(df), index=df.index)

        df["start_year"]  = start.dt.year.fillna(2024).astype(int)
        df["is_democrat"] = df["start_year"].apply(lambda y: int(y in _DEMOCRAT_YEARS))

        if "duration" in df.columns:
            df["duration_days"] = pd.to_numeric(df["duration"], errors="coerce").fillna(365)
        elif "End Date" in df.columns:
            end = pd.to_datetime(df["End Date"], format="mixed", errors="coerce")
            df["duration_days"] = (end - start).dt.days.fillna(365)
        else:
            df["duration_days"] = 365

        # ------------------------------------------------------------------
        # Agency columns
        # ------------------------------------------------------------------
        for col in ("Awarding Agency", "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency"):
            if col not in df.columns:
                df[col] = "UNKNOWN"
            else:
                df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()

        # ------------------------------------------------------------------
        # Derived boolean flags: does awarding agency differ from funding?
        #    These were engineered during training and must be reproduced here.
        # ------------------------------------------------------------------
        df["award_not_fund_agency"] = (
            df["Awarding Agency"].str.upper() != df["Funding Agency"].str.upper()
        ).astype(int)

        df["award_not_fund_sub_agency"] = (
            df["Awarding Sub Agency"].str.upper() != df["Funding Sub Agency"].str.upper()
        ).astype(int)

        # ------------------------------------------------------------------
        # NAICS + place of performance + market share
        # ------------------------------------------------------------------
        if "NAICS Code" in df.columns and "naics_code" not in df.columns:
            df["naics_code"] = df["NAICS Code"]

        defaults = {
            "naics_code": "UNKNOWN",
            "market_share": 0.0,
            "Place of Performance Country Code": "USA",
            "Place of Performance State Code": "DC",
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        # ------------------------------------------------------------------
        # Guarantee every required column exists
        # ------------------------------------------------------------------
        numeric_defaults = {
            "start_year", "is_democrat", "duration_days",
            "market_share", "Award Amount",
            "award_not_fund_agency", "award_not_fund_sub_agency",
        }
        for col in self.required_cols:
            if col not in df.columns:
                df[col] = 0 if col in numeric_defaults else "UNKNOWN"

        return df