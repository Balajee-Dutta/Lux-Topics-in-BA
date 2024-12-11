Tobacco Affordability & Taxation Trends in Luxembourg
Project Overview
This project analyzes how tobacco taxation impacts affordability in Luxembourg from 2012-2020. Using Python and Power BI, we explore trends, build predictive models, and offer policy recommendations to support public health. Key models forecast affordability trends from 2022-2030, providing valuable insights for policymakers.

Objectives
Trend Analysis: Assess how taxation affected affordability (2012-2020).
Forecasting: Predict future affordability using Polynomial Regression, STL, and Exponential Smoothing models.

Policy Support: Offer data-driven recommendations to reduce smoking rates.

Project Structure

[data/             # Raw and processed datasets]
[notebooks/        # Jupyter notebooks for analysis]
[scripts/          # Python scripts for data cleaning, visualization, and modeling]
[visualizations/   # Visuals (heatmaps, charts, etc.)]
[README.md         # Project overview and instructions, Required libraries]

How to Run

Run Scripts:
Run the main file for data cleaning, visualizations, and forecasting:
python Lux-Topics_in_BA\Lux-main_file.py

Tools & Techniques
Languages: Python (Pandas, NumPy, Scikit-Learn, Statsmodels)
Visuals: Power BI, Matplotlib, Seaborn
Models: Polynomial Regression, STL, and Exponential Smoothing

Data Sources
WHO (World Health Organization)
Eurostat
Tobacco Atlas
World Bank
IHME (Institute for Health Metrics and Evaluation)

All data complies with GDPR, and privacy measures like k-anonymity and data aggregation were applied.

Predictive Models & Key Insights:

Polynomial Regression: Captures non-linear trends but less accurate long-term.
STL with Linear Regression: Best model for tracking seasonal and long-term trends.
Exponential Smoothing: Tracks short-term fluctuations but weaker for long-term forecasts.

Key Insight
There is a perfect negative correlation (-1.0) between affordability and tax rate. This shows that higher taxes make tobacco less affordable, which can reduce smoking.

Visualizations
Heatmaps: Correlation between affordability, taxes, and other factors.
Line Charts: Affordability trends and predictions for 2022-2030.
Bar Graphs: Smoking rates by age, gender, and country (via Power BI).

Key Takeaways

Taxation is Key: Higher taxes reduce affordability and lower smoking rates.
Best Model: STL with Linear Regression is the most effective for forecasting.
Policy Advice: Increase tobacco taxes, introduce dynamic pricing, and support public health campaigns.

Usage Instructions:

Clone the repository.

Install dependencies.

\Run the main file:

python Lux-Topics_in_BA\Lux-main_file.py

Acknowledgments
Data was sourced from WHO, Eurostat, Tobacco Atlas, World Bank, and IHME. Special thanks to these organizations for their valuable datasets.

License
This project is under the MIT License. See the LICENSE file for more details.
