# 🚀 Social Media Sentiment Analysis Dashboard

## 📌 Overview

This project is an **end-to-end Machine Learning + NLP dashboard** that analyzes textual data and classifies sentiment into **Positive, Negative, and Neutral** categories.

It includes a **Streamlit-based interactive dashboard** to visualize sentiment insights, making it easy to understand patterns in textual data.

---

## 🎯 Problem Statement

Understanding sentiment from large volumes of text data (such as comments, feedback, or reviews) is challenging and time-consuming.

This project solves the problem by:

* Automatically classifying sentiment using Machine Learning
* Providing visual insights through an interactive dashboard
* Enabling quick interpretation of text data

---

## 🧠 Project Workflow

```
Raw Text Data
   ↓
Text Cleaning
   ↓
Preprocessing (Stopwords Removal)
   ↓
Feature Extraction (TF-IDF)
   ↓
Machine Learning Model (Logistic Regression)
   ↓
Sentiment Prediction
   ↓
Streamlit Dashboard Visualization
```

---

## 🛠️ Tech Stack

**Language**

* Python

**Libraries**

* Pandas, NumPy
* NLTK
* Scikit-learn
* TF-IDF Vectorizer
* Matplotlib
* Seaborn
* Plotly
* WordCloud

**Frontend**

* Streamlit

---

## 🏗️ Project Architecture

```
Data (CSV)
   ↓
Preprocessing Module (clean_text, stopwords)
   ↓
Feature Engineering (TF-IDF)
   ↓
ML Model (Logistic Regression)
   ↓
Prediction Module
   ↓
Dashboard (Streamlit UI)
   ↓
Charts & Insights
```

---

## 📂 Folder Structure

```
Social-Media-Sentiment-Analysis-Dashboard/
│
├── data/               # Dataset
├── src/                # Core ML logic
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│
├── models/             # Saved model & vectorizer
├── app/                # Streamlit dashboard
├── outputs/            # Confusion matrix
├── images/             # Screenshots
├── .streamlit/         # Theme configuration
│
├── requirements.txt
└── README.md
```

---

## ✨ Features

### 🔍 Sentiment Prediction

* Input any text and get real-time sentiment classification

### 📊 Interactive Dashboard

* Pie chart (sentiment distribution)
* Bar chart (sentiment comparison)
* Line chart (trend analysis)

### 📈 Data Visualization

* Interactive graphs using Plotly
* Word Cloud for text insights

### 📉 Model Evaluation

* Accuracy score
* Classification report
* Confusion matrix

### 📂 Data Exploration

* Dataset preview
* Filter by sentiment

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/YOUR_USERNAME/social-media-sentiment-analysis-dashboard.git
cd social-media-sentiment-analysis-dashboard
```

---

### 2️⃣ Create virtual environment

**Windows**

```
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux**

```
python3 -m venv venv
source venv/bin/activate
```

---

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Train the model

```
python src/train.py
```

---

### 5️⃣ Run the dashboard

```
streamlit run app/dashboard.py
```

---

## 📊 Sample Prediction

**Input:**

```
"This product is really good and useful"
```

**Output:**

```
Positive
```

---

## 📈 Results

* Successfully classified sentiment into three categories
* Built a complete ML pipeline from preprocessing to deployment
* Created an interactive dashboard for better visualization

---

## 🎓 Learning Outcomes

* Natural Language Processing (NLP) fundamentals
* Text preprocessing techniques
* Feature extraction using TF-IDF
* Machine Learning model building
* Model evaluation techniques
* Dashboard development using Streamlit

---

## 🚀 Future Improvements

* Use larger and more diverse datasets
* Improve model performance with advanced algorithms
* Add real-time data integration
* Deploy the dashboard online


If you found this project useful, consider giving it a ⭐ on GitHub!
