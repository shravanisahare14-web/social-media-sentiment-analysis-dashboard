import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from src.predict import predict_sentiment

# ----------------------
# CONFIG
# ----------------------
st.set_page_config(page_title="Sentiment Analytics", layout="wide")

st.title("🚀 Social Media Sentiment Analytics Dashboard")

# ----------------------
# LOAD DATA
# ----------------------
df = pd.read_csv("data/dataset.csv")

# FAKE TIME COLUMN FOR TREND
df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

# ----------------------
# USER INPUT
# ----------------------
st.subheader("🔍 Real-Time Prediction")

user_input = st.text_area("Enter text")

if st.button("Analyze"):
    result = predict_sentiment(user_input)
    st.success(f"Predicted Sentiment: {result}")

# ----------------------
# KPI METRICS
# ----------------------
total = len(df)
pos = (df['sentiment'] == 'positive').sum()
neg = (df['sentiment'] == 'negative').sum()
neu = (df['sentiment'] == 'neutral').sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total", total)
col2.metric("Positive", pos)
col3.metric("Negative", neg)
col4.metric("Neutral", neu)

# ----------------------
# PIE + BAR
# ----------------------
st.subheader("📊 Distribution")

col1, col2 = st.columns(2)

with col1:
    fig_pie = px.pie(df, names='sentiment', hole=0.5)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    counts = df['sentiment'].value_counts().reset_index()
    counts.columns = ['sentiment', 'count']
    fig_bar = px.bar(counts, x='sentiment', y='count', color='sentiment', text='count')
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------------
# TREND ANALYSIS (IMPORTANT)
# ----------------------
st.subheader("📈 Sentiment Trend Over Time")

trend = df.groupby(['date', 'sentiment']).size().reset_index(name='count')

fig_trend = px.line(
    trend,
    x='date',
    y='count',
    color='sentiment',
    markers=True
)

st.plotly_chart(fig_trend, use_container_width=True)

# ----------------------
# WORD CLOUD
# ----------------------
st.subheader("🧠 Word Analysis")

text = " ".join(df['text'])
wc = WordCloud(background_color='black').generate(text)

fig, ax = plt.subplots()
ax.imshow(wc)
ax.axis("off")

st.pyplot(fig)

# ----------------------
# CONFUSION MATRIX DISPLAY
# ----------------------
st.subheader("📉 Model Evaluation")

st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")

# ----------------------
# FILTER + DATA TABLE
# ----------------------
st.subheader("📂 Explore Data")

option = st.selectbox("Filter", ["All", "positive", "negative", "neutral"])

if option != "All":
    st.dataframe(df[df['sentiment'] == option])
else:
    st.dataframe(df)