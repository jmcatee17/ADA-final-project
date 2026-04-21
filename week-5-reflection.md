# Week 5 Project Reflection
### What you have achieved so far?

#### Data Access Accomplishments
I have written code to access columns

["Award ID", "generated_internal_id", "Recipient Name", "Recipient DUNS Number", "recipient_id",
"Award Amount", "Awarding Agency", "Awarding Sub Agency", "Funding Agency", "Funding Sub Agency",
"Place of Performance City Code", "Place of Performance State Code", "Place of Performance Country Code",
"Place of Performance Zip5", "Last Modified Date", "Base Obligation Date",
"Start Date", "End Date", "Description"]

From the USA Spending (Spending by Award) API Endpoint: https://api.usaspending.gov/api/v2/search/spending_by_award/
[Documentation](https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/contracts/v2/search/spending_by_award.md)

As well as columns

["Award ID", "internal_id", "generated_internal_id", "Recipient Name", "Action Date",
"Award Type", "Mod", "Transaction Amount", "Transaction Description", 
"Awarding Sub Agency", "Awarding Agency", "PSC", "NAICS", "Action Date"]

from the USA Spending (Spending by Transaction) API Endpoint: https://api.usaspending.gov/api/v2/search/spending_by_transaction/

[Documentation]https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/contracts/v2/search/spending_by_transaction.md

I set initializations such that I am filtering the API query for the past 5 years of data for all Cabinets and several independent agencies. Additionally. I am looking specifically at technology spend which is specified by the specific NAICS codes below.

```python
# Fiscal Year 2021 (10/01/2020) through 04/20/2026
start = '2020-10-01'
end = '2026-04-20'

# Define all cabinets plus several independent agencies
target_departments = [
    "Department of Homeland Security", "Department of Justice",
    "Department of Defense",
    "Department of State", "Department of the Treasury",
    "Department of Energy", "Department of Commerce",
    "Department of Health and Human Services", "Department of Agriculture",
    "Department of the Interior", "Department of Transportation",
    "Securities and Exchange Commission", "Commodity Futures Trading Commission"
]

award_type_codes = ["A", "B", "C", "D"]

# Technology Only NAICS codes of interest
naics_codes = ["518210", "541511", "541519", "541512", "513210", "511210", "541512", "334111"]

```

#### Feature Engineering
Additionally, I wrote pandas functions to inspect and ensure the quality of the data pulled, as well as code to merge the tables together. I also added a column for the incumbent administration.

I am continuing to work on getting the modification data and calculating the binary on whether or not the contract was modified within the first 365 days.

#### Data Management Considerations
- The `data_ingestion` folder will house both awards and transactions baseline csvs, as well as the code to Scrape and Preprocess the data.
- There will be a third preprocessed csv saved in the folder, which will include proprocessing, cleaning, and feature engineering.
- A model will be trained on this data in a new folder called `model_training`. A series of models will be trained and saved as pickle files or joblib files in this repository.
- For inference, and the live dashboard, there will be a separate folder hosting the code for the streamlit app. This app will make predictions on data pulled directly from the USA Spending API.
### What you are happy with, from your project work so far?
I am happy with my problem solving skills, and ability to consolidate so much data although I had several hurdles. Additionally, I put a lot of investment and effort up front into organizing my repository, and jupyter notebook code. I believe that this will make it much easier for collaboration, as well as when it is time to translate the code from a notebook to production environment. This is a skillset I learned as a technical consultant, and I believe it will make my project much more complete and industry-grade.

### What you are struggling with, or what challenges you are facing next?
A major challenge I faced was with the sheer volume of contracting data that exists. My initial scope was to
pull 5 years worth of data, which I am attempting to do. I am finding that this equates to hundreds of thousands
of federal contracts. Since I am pinging an open API, I initially ran into rate limiting and pagination issues.

I leveraged online sources and Generative AI to help solve this bottleneck. Still, I have not figured out a way
to parallelize these queries. As a result, I am able to obtain the information, but the query takes a long time.
To save time, I am storing the pulled API data in a csv.

### Anything else you’d specifically like the course staff to focus on in giving you feedback or advice?
I am interested to see if there are better ways of accessing and optimizing the API queries. Specifically, can this be parallelized from my Macbook.