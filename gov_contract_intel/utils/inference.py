import joblib
import pandas as pd

import numpy as np

# Absolute import - works correctly inside the utils/ package
from utils.feature_engineering import ContractFeatureEngineer


class ContractInference:
    def __init__(self, reg_path: str, clf_path: str):
        self.amount_model = joblib.load(reg_path)
        self.risk_model = joblib.load(clf_path)
        self.engineer = ContractFeatureEngineer()

    def predict_all(self, data_dict: dict) -> dict:
        """
        Accepts a single dict from the UI form and returns amount + risk predictions.
        Expected keys: PSC, duration (int, days), Description
        """
        df = pd.DataFrame([data_dict])
        processed_df = self.engineer.prepare_for_inference(df)

        # Regression: predicted contract value
        y_pred_log = float(self.amount_model.predict(processed_df)[0])
        pred_amount = np.expm1(y_pred_log)

        # Classification: modification risk probability
        risk_prob = float(self.risk_model.predict_proba(processed_df)[0][1])
        if risk_prob > 0.9:
            risk_label = "Very High"
        elif risk_prob > .7:
            risk_label = "High"
        elif risk_prob > .4:
            risk_label = "Medium"
        else:
            risk_label = "Low"

        return {
            "amount": pred_amount,
            "risk_score": risk_prob,
            "risk_label": risk_label,
        }