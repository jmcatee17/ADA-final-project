import requests
import pandas as pd


class USASpendingAPI:
    BASE_URL = "https://api.usaspending.gov/api/v2/"

    def __init__(self):
        self.search_url = f"{self.BASE_URL}search/spending_by_award/"

    def scrape_contracts(self, start_date: str, end_date: str, agencies: list) -> pd.DataFrame:
        """
        Fetches up to 200 contract records from USASpending for the given
        date range and awarding agency names.
        """
        payload = {
            "filters": {
                "time_period": [{"start_date": start_date, "end_date": end_date}],
                "agencies": [
                    {"type": "awarding", "tier": "toptier", "name": a} for a in agencies
                ],
                "award_type_codes": ["A", "B", "C", "D"],  # Contracts only
            },
            "fields": [
                "Award ID",
                "Recipient Name",
                "Start Date",
                "End Date",
                "Award Amount",
                "Awarding Agency",
                "Awarding Sub Agency",
                "PSC",
                "Description",
            ],
            "limit": 100,
            "page": 1,
            "sort": "Award Amount",
            "order": "desc",
        }

        all_results = []
        try:
            for page in range(1, 3):  # 2 pages = up to 200 records
                payload["page"] = page
                response = requests.post(self.search_url, json=payload, timeout=30)
                if response.status_code == 200:
                    data = response.json().get("results", [])
                    if not data:
                        break
                    all_results.extend(data)
                else:
                    print(f"API Error on page {page}: {response.status_code} — {response.text[:200]}")
                    break
        except requests.exceptions.RequestException as e:
            print(f"Connection failed: {e}")

        df = pd.DataFrame(all_results)
        if not df.empty:
            df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")
        return df