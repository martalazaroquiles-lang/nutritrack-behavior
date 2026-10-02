import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---------------- CONFIG ----------------
st.set_page_config(page_title="NutriTrack Behavior", page_icon="🥗", layout="wide")

WEIGHT_ORDER = ['Insufficient_Weight', 'Normal_Weight', 'Overweight', 'Obesity_Type_I']
COLOR_MAP = {
    'Insufficient_Weight': '#2E86AB',
    'Normal_Weight': '#A23B72',
    'Overweight': '#F18F01',
    'Obesity_Type_I': '#C73E1D',
}
FEATURES = ['Age', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
LABELS = {
    'Age': 'Age (years)',
    'FCVC': 'Vegetable consumption (1-3)',
    'NCP': 'Main meals per day (1-4)',
    'CH2O': 'Daily water intake (1-3)',
    'FAF': 'Physical activity frequency (0-3)',
    'TUE': 'Screen time (0-2)',
}

# Path relative to this file -> works locally and on Streamlit Cloud
DATA_PATH = Path(__file__).resolve().parent.parent / 'data' / 'obesity_level.csv'


# ---------------- DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={'0be1dad': 'NObeyesdad'})
    df['NObeyesdad'] = df['NObeyesdad'].replace('0rmal_Weight', 'Normal_Weight')
    df['CAEC'] = df['CAEC'].replace('0', 'No')
    df['CALC'] = df['CALC'].replace('0', 'No')

    women = df[
        (df['Gender'] == 'Female')
        & (~df['NObeyesdad'].isin(['Obesity_Type_II', 'Obesity_Type_III']))
    ].copy()
    women['category'] = women['NObeyesdad'].replace({
        'Overweight_Level_I': 'Overweight',
        'Overweight_Level_II': 'Overweight',
    })
    return women


def eta_squared(data, var):
    grand_mean = data[var].mean()
    ss_between = sum(
        len(g) * (g[var].mean() - grand_mean) ** 2
        for _, g in data.groupby('category')
    )
    ss_total = ((data[var] - grand_mean) ** 2).sum()
    return ss_between / ss_total if ss_total > 0 else 0


@st.cache_data
def train_models(data):
    y = data['category']
    results = {}
    for name, feats in {'A': ['Age'], 'B': FEATURES}.items():
        X = data[feats]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        lr = model.named_steps['logisticregression']
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'cm': confusion_matrix(y_test, y_pred, labels=WEIGHT_ORDER),
            'coef': pd.DataFrame(lr.coef_, index=lr.classes_, columns=feats).loc[WEIGHT_ORDER],
        }
    return results


df = load_data()

# ---------------- HEADER ----------------
st.title("🥗 NutriTrack Behavior")
st.markdown(
    "**Which daily behaviors distinguish women's weight categories — "
    "and what does it mean for nutrition coaching apps?**"
)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Filters")
age_min, age_max = int(df['Age'].min()), int(df['Age'].max())
age_range = st.sidebar.slider("Age range", age_min, age_max, (age_min, age_max))
filtered = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
st.sidebar.caption("Filters apply to the Overview and Behaviors tabs. The model always uses the full sample.")

if filtered.empty:
    st.warning("No data for this age range. Widen the filter.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🍽️ Behaviors", "🤖 Model", "⚠️ Methodology & Limits"])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Women analyzed", f"{len(filtered):,}")
    c2.metric("Average age", f"{filtered['Age'].mean():.1f}")
    c3.metric("Normal weight", f"{(filtered['category'] == 'Normal_Weight').mean():.0%}")
    c4.metric(
        "Overweight or obesity",
        f"{filtered['category'].isin(['Overweight', 'Obesity_Type_I']).mean():.0%}",
    )

    dist = filtered['category'].value_counts().reindex(WEIGHT_ORDER).fillna(0).reset_index()
    dist.columns = ['category', 'count']
    fig = px.bar(
        dist, x='category', y='count', color='category',
        color_discrete_map=COLOR_MAP, text='count',
        title="Weight category distribution",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Women")
    st.plotly_chart(fig, use_container_width=True)

    fig_age = px.box(
        filtered, x='category', y='Age', color='category',
        category_orders={'category': WEIGHT_ORDER}, color_discrete_map=COLOR_MAP,
        title="Age by weight category",
    )
    fig_age.update_layout(showlegend=False, xaxis_title="")
    st.plotly_chart(fig_age, use_container_width=True)

# ---------------- TAB 2: BEHAVIORS ----------------
with tab2:
    st.subheader("How much does each variable separate the weight categories?")
    eta = pd.DataFrame({
        'variable': [LABELS[f] for f in FEATURES],
        'eta_squared': [eta_squared(filtered, f) for f in FEATURES],
    }).sort_values('eta_squared')
    fig_eta = px.bar(
        eta, x='eta_squared', y='variable', orientation='h', text_auto='.3f',
        title="Effect size (eta²) — share of variance explained by weight category",
    )
    fig_eta.update_layout(xaxis_title="eta²", yaxis_title="")
    st.plotly_chart(fig_eta, use_container_width=True)
    st.caption("Rule of thumb: 0.01 = small, 0.06 = medium, 0.14 = large.")

    st.subheader("Explore a behavior")
    var = st.selectbox("Variable", FEATURES, format_func=lambda x: LABELS[x])
    fig_box = px.box(
        filtered, x='category', y=var, color='category',
        category_orders={'category': WEIGHT_ORDER}, color_discrete_map=COLOR_MAP,
        title=f"{LABELS[var]} by weight category",
    )
    fig_box.update_layout(showlegend=False, xaxis_title="", yaxis_title=LABELS[var])
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Average values by category")
    means = (
        filtered.groupby('category')[FEATURES].mean()
        .reindex(WEIGHT_ORDER).round(2).rename(columns=LABELS)
    )
    st.dataframe(means, use_container_width=True)

# ---------------- TAB 3: MODEL ----------------
with tab3:
    st.subheader("Do daily habits add predictive power beyond age?")
    st.markdown(
        "Two multinomial logistic regressions, trained on the same 80/20 stratified split:  \n"
        "**Model A** uses only Age. **Model B** uses Age + 5 lifestyle variables."
    )
    res = train_models(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model A accuracy", f"{res['A']['accuracy']:.1%}")
    m2.metric(
        "Model B accuracy", f"{res['B']['accuracy']:.1%}",
        delta=f"{(res['B']['accuracy'] - res['A']['accuracy']) * 100:+.1f} pts vs A",
    )
    m3.metric("Model A macro F1", f"{res['A']['f1_macro']:.2f}")
    m4.metric(
        "Model B macro F1", f"{res['B']['f1_macro']:.2f}",
        delta=f"{res['B']['f1_macro'] - res['A']['f1_macro']:+.2f} vs A",
    )
    st.caption("Random-chance baseline with 4 categories ≈ 25% accuracy.")

    col_a, col_b = st.columns(2)
    with col_a:
        cm = pd.DataFrame(res['B']['cm'], index=WEIGHT_ORDER, columns=WEIGHT_ORDER)
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Women"),
            title="Confusion matrix — Model B",
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Most errors fall between adjacent categories, showing the model captures the ordinal structure of weight.")
    with col_b:
        coef = res['B']['coef'].T.rename(index=LABELS)
        fig_coef = px.imshow(
            coef, text_auto='.2f', color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0, aspect='auto',
            title="Coefficients — Model B (standardized)",
        )
        st.plotly_chart(fig_coef, use_container_width=True)
        st.caption("Red = higher values make the category more likely. Associations, not causation.")

# ---------------- TAB 4: METHODOLOGY ----------------
with tab4:
    st.subheader("Data cleaning decisions")
    st.markdown("""
- **Source:** *Obesity Levels Based on Eating Habits & Physical Condition* (Kaggle/UCI), filtered to women only.
- **Excluded Obesity_Type_II:** almost exclusively male (8 of 3,248 cases were women), likely a bias from synthetic data generation.
- **Excluded Obesity_Type_III:** synthetic-data artifacts (zero or near-zero variance in several variables).
- **Merged Overweight Level I + II:** recalculated BMI matched these labels only 64% and 46% of the time; merging raised overall label consistency to 85.3%.
- **Final sample:** 6,373 women across 4 weight categories.
    """)
    st.subheader("Key limitations")
    st.markdown("""
- **Cross-sectional data:** this shows which behaviors *co-occur* with each weight category, not what causes weight change or predicts long-term adherence.
- **Partly synthetic dataset:** after removing artifacts, most lifestyle effects shrank substantially, while Age's effect stayed stable (~0.21). Lifestyle findings should be read with caution.
- **Self-reported behaviors** on coarse scales (1-3, 0-2), which limits precision.
- **Next step:** an ordinal logistic regression would respect the natural order of the weight categories.
    """)

st.divider()
st.caption("NutriTrack Behavior · Portfolio project by Marta Lázaro Quiles · Data: Kaggle / UCI")