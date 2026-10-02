import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

st.set_page_config(page_title="NutriTrack Behavior", layout="wide")

px.defaults.color_discrete_sequence = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

# Load data
df = pd.read_csv('../data/obesity_level.csv')

# Reapply the same cleaning steps from the notebook
df = df.rename(columns={'0be1dad': 'NObeyesdad'})
df['NObeyesdad'] = df['NObeyesdad'].replace('0rmal_Weight', 'Normal_Weight')
df['CAEC'] = df['CAEC'].replace('0', 'No')
df['CALC'] = df['CALC'].replace('0', 'No')

df_women = df[(df['Gender'] == 'Female') & (df['NObeyesdad'] != 'Obesity_Type_II') & (df['NObeyesdad'] != 'Obesity_Type_III')].copy()
df_women['NObeyesdad_merged'] = df_women['NObeyesdad'].replace({
    'Overweight_Level_I': 'Overweight',
    'Overweight_Level_II': 'Overweight'
})

weight_order_final = ['Insufficient_Weight', 'Normal_Weight', 'Overweight', 'Obesity_Type_I']

# Title
st.title("🥗 NutriTrack Behavior")
st.markdown("**Which daily behaviors distinguish women's weight categories — and what does it mean for nutrition coaching apps?**")