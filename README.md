# 🥗 NutriTrack Behavior

**Which daily behaviors distinguish women's weight categories — and what does it mean for nutrition coaching apps?**

🔗 **[Live interactive dashboard](https://nutritrack-behavior.streamlit.app)**

---

## Business question

Nutrition coaching apps (e.g., Noom) personalize their programs around users' daily habits: vegetable intake, meal frequency, water, physical activity, screen time. But how much do these habits actually tell us about a woman's weight status, beyond basic demographics like age?

This project analyzes the behavioral profile of **6,373 women** across four weight categories to answer two questions:

1. Which daily behaviors differ most between weight categories?
2. Do behavioral variables add meaningful predictive power beyond age alone?

## Key findings

**1. Age carries most of the predictive signal; daily habits add a modest increment.**
Two multinomial logistic regressions were trained on the same 80/20 stratified split:

| Model | Features | Accuracy | Macro F1 |
|---|---|---|---|
| Random baseline | — | 25.0% | — |
| Model A | Age only | 42.4% | 0.42 |
| Model B | Age + 5 lifestyle behaviors | 48.1% | 0.48 |

Age alone accounts for roughly **three-quarters of the model's predictive power above chance**. Adding five lifestyle behaviors improves accuracy by only **+5.7 points**.

**2. Lifestyle effects shrank after removing synthetic-data artifacts; age's effect did not.**
Effect sizes (eta²) for most lifestyle variables dropped substantially once data-quality issues were cleaned, while age remained stable (~0.21) before and after cleaning. This suggests age is a genuine factor, while part of the apparent lifestyle signal in the raw data was noise.

**3. The model understands the ordinal structure of weight.**
Nearly all misclassifications happen between *adjacent* categories (e.g., Normal_Weight confused with Insufficient_Weight or Overweight). Only 3.9% of Normal_Weight cases were misclassified as Obesity_Type_I.

**4. Vegetable consumption showed the strongest behavioral association.**
FCVC had the largest coefficients in both directions: positively associated with Insufficient_Weight and negatively with Obesity_Type_I. Some associations were counterintuitive (e.g., higher water intake associated with higher weight categories), a reminder that these are correlations, not causal effects.

## Product implications

For a nutrition coaching app, these results suggest that **cross-sectional, self-reported habit data adds limited predictive value beyond basic demographics**. Habit-based personalization would likely need:

- **Longitudinal tracking** (how behaviors change over time, not a single snapshot)
- **More granular measurement** (actual logged meals and activity instead of 1–3 scales)
- **Outcome data tied to behavior change** (weight trajectory, adherence, retention)

## Data

**Source:** [Obesity Levels Based on Eating Habits & Physical Condition](https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset) (Kaggle / UCI) — 20,758 rows, 18 columns.

### Cleaning decisions

| Decision | Reason |
|---|---|
| Filtered to women only | Project focus on women's health |
| Excluded **Obesity_Type_II** | Almost exclusively male (8 of 3,248 cases were women), likely a synthetic-generation bias |
| Excluded **Obesity_Type_III** | Synthetic-data artifacts: zero or near-zero variance in several variables (FCVC, NCP, SCC, FAVC, SMOKE, Height) |
| Merged **Overweight Level I + II** | BMI recalculated from Weight/Height matched these labels only 64% and 46% of the time; merging raised overall label consistency to **85.3%** |

**Final sample:** 6,373 women across 4 categories: Insufficient_Weight, Normal_Weight, Overweight, Obesity_Type_I.

### Variables analyzed

| Variable | Description |
|---|---|
| Age | Age in years |
| FCVC | Frequency of vegetable consumption (1–3) |
| NCP | Number of main meals per day (1–4) |
| CH2O | Daily water intake (1–3) |
| FAF | Physical activity frequency (0–3) |
| TUE | Time using technology devices (0–2) |

## Methods

- **Data quality audit:** label consistency check via recalculated BMI, variance checks per category
- **Effect sizes:** eta² to quantify how much each variable separates weight categories
- **Modeling:** multinomial logistic regression (scikit-learn) with standardized features, comparing an age-only baseline against a full behavioral model
- **Evaluation:** accuracy, macro F1, confusion matrix analysis

## Limitations

- **Cross-sectional data:** shows which behaviors *co-occur* with each weight category, not what causes weight change.
- **Partly synthetic dataset:** artifacts were detected and removed, but residual synthetic patterns may remain.
- **Self-reported, coarse scales:** limits measurement precision.
- **Associations, not causation:** coefficients should not be read as recommendations.

## Next steps

- **Ordinal logistic regression**, which explicitly models the natural order of weight categories and would likely outperform the multinomial approach
- Apply the same framework to a **longitudinal** dataset to study adherence and behavior change over time

## Tech stack

Python (pandas, NumPy, scikit-learn, SciPy) · SQL · Plotly · Streamlit · Jupyter · Git/GitHub

## Repository structure

## Repository structure

```
nutritrack-behavior/
├── data/           # Raw dataset
├── sql/            # SQL queries
├── notebooks/      # Exploration, data quality, statistics, modeling
├── dashboard/      # Streamlit app (app.py)
├── requirements.txt
└── README.md
```

## Run locally

```bash
git clone https://github.com/martalazaroquiles-lang/nutritrack-behavior.git
cd nutritrack-behavior
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Related project

This is the second project in my women's health analytics portfolio, following **FemCycle Insights** (menstrual cycle pattern analysis).

---

**Author:** Marta Lázaro Quiles · [GitHub](https://github.com/martalazaroquiles-lang)