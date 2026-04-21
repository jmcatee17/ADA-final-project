# Final Project Proposal
## Prompt
I am interested in how the federal government spends its money on technology. USA Spending (https://www.usaspending.gov/) has an incredibly robust website and API of unclassified contracts that the federal government spends its money on. I would like to evaluate trends in spending across technologies, departments, and agencies across different administrations.

### Execution Plan:
#### By Week 5 Homework Deadline:
- Pull 5 years of transactions data from the usa spending api across all departments, agencies, and NAICS codes. I plan to access these API's using pythons requests library.
- Create a ground truth table of historical spending in a baseline table. Ideate on exactly where this data will live and implement (i.e. csv versus Database).
- Create a binary column (overspend) if a contract will cost more than the original base award (total award amount > base award amount).
- Modification (what is the probability that contract will be modified within first year)
    - Create a binary column (1/0) for if modified within 365 days
    - Way to measure scope creep
- Create a feature to show party of incumbent administration.
- Ideate on model implementation and process (i.e. XGBoost, Random Forest, Regression, Logistic Regression)

#### By Week 7 Homework Deadline: 
- Conduct feature engineering. Create proxies for things like Market Power, Complexity, and Stability of contract and contractor. I will also use adminstration factor to examine in Republican vs. Democrat has an affect on the awards.
- Apply and train a natural language processing tool to the description field of the awards/
    - This would go into different technology towers (Zero Trust, Generative AI, Quantum Computing, Graph Computing, etc.)
- I will also create a model to identifying relationships between certain keywords and concepts in the contracts and administrations.
- Train a model and Predict if contract will cost more than the original base award
- Train a model and predict total amount of spend. The award amount will be a predictor variable. Compare results with binary prediction.
- Predict Modification likelihood within first 365 days
- Create a dashboard that allows the user to see largest contracts, incumbent, and if they are expected to win the follow on,
- Allow for user to filter for different contracts, incumbents, performance periods, PSC codes, etc.

#### By the final due date: 
- Update V1 of models to ensure they meet industry standards. Create a process for continuously updating and refining models such that they do not become stale (MLOps).
- Integrate new models into dashboard
- Ensure that dashboard is running continuously.
- Test dashboard works on different environments (outside of my computer)
- On Thursday morning in week 9, I plan to film my video.
- Finish touch ups to the dashboard to make sure it has a great user experience.
