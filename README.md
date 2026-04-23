# 🚔 Crime Pattern Analysis in India using Machine Learning

This project focuses on analyzing and predicting crime patterns in India using Machine Learning techniques. It helps identify trends, visualize crime distribution, and assist in decision-making for safety and governance. By leveraging historical crime data, the model identifies trends, patterns, and key factors contributing to criminal activities. The goal is to assist in better decision-making, crime prevention strategies, and resource allocation.

---

## 🎯 Objectives
- Analyze crime data across different regions
- Identify patterns and trends
- Build a predictive model for crime classification
- Develop an interactive UI for visualization

---

## 🧠 Features
- 📊 Data preprocessing and cleaning
- 📈 Exploratory Data Analysis (EDA)
- 🤖 Machine Learning model training
- 📉 Model evaluation (accuracy, classification report)
- 💻 Interactive UI using Streamlit

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Streamlit
- Jupyter
---


## 📊 Dataset

**File:** `data/crime_dataset_india.csv`

| Column | Description |
|---|---|
| Report Number | Unique ID for each crime report |
| Date Reported | Date the crime was reported |
| Date of Occurrence | Actual date of the crime |
| Time of Occurrence | Time the crime occurred |
| City | City where the crime took place |
| Crime Code | Numeric code for crime type |
| Crime Description | Detailed description of the crime |
| Victim Age | Age of the victim |
| Victim Gender | Gender of the victim |
| Weapon Used | Weapon involved (if any) |
| Crime Domain | Category/domain of the crime (Target Variable) |
| Police Deployed | Number of police personnel deployed |
| Case Closed | Whether the case was closed |
| Date Case Closed | Date the case was closed |

---

## 🧠 ML Pipeline

### 1. Data Preprocessing
- Removed duplicates
- Filled missing values (median for age, "Unknown" for weapons)
- Parsed and extracted date/time features

### 2. Exploratory Data Analysis (EDA)
- Top cities by crime count
- Crime domain distribution
- Yearly crime trends

### 3. Classification (Random Forest)
- **Target:** `Crime Domain`
- **Model:** `RandomForestClassifier` (100 estimators)
- **Split:** 80% train / 20% test
- **Metrics:** Accuracy, Classification Report

### 4. Clustering (K-Means)
- Grouped cities into 5 clusters based on crime count
- Visualized hotspot clusters on a scatter plot

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/aayu04shi/Crime-Anaylis-in-India.git
cd crime-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook notebooks/ML_Model_CrimeAnalysis.ipynb
```
---

## 📦 Requirements

See `requirements.txt` for the full list. Key libraries:
- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Visualization
- `scikit-learn` – ML models (Random Forest, K-Means)
- `jupyter` – Notebook environment

---

## 📈 Results

| Metric | Value |
|---|---|
| Model | Random Forest Classifier |
| Number of Clusters | 5 (K-Means) |
| Target Variable | Crime Domain |

---

## 📁 Repository Structure

```
Crime-Analysis-in-India/
│
├── app.py                         # Main Streamlit app
├── leaderboard.csv                # Auto-generated leaderboard
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── .gitignore                     # Ignore unnecessary files
│
├── data/
│   └── crime_dataset_india.csv    # Dataset
│
├── models/
│   └── .gitkeep
│  
│
├── src/
│   ├── __init__.py
│   ├── train_model.py             # Training logic
│   ├── data_preprocessing.py      # Cleaning logic
│
├── submissions/                   # ⭐ USER SUBMISSIONS (VERY IMPORTANT)
│   ├── user1_1712345678.csv
│   ├── user2_1712348901.csv
│
├── pages/
│   └── leaderboard.py             # Streamlit leaderboard page
│
├── .github/
│   └── workflows/
│       └── update_leaderboard.yml # Auto leaderboard update
│
├── update_leaderboard_auto.py     # Reads submissions → updates leaderboard
├── update_readme.py               # Updates README leaderboard
│
└── notebooks/ 
    └── analysis.ipynb
```

---
## 📸 Screenshots [User Interface]

### 🔹 Prediction Output
![Prediction](outputs/prediction.png)

### 🔹 Dataset Preview
![Dataset_Preview](outputs/dataset_preview.png)

### 🔹 Crime Analysis Graph
![Graph](outputs/graph.png)


---


## 👥 Contributors

| Name |
|---|
| Sreeya S. S. |
|---|
| Aayushi P. Naik |
|---|
| Saksham S. Lohote |



---

## 📌 How to Participate

Follow these simple steps to contribute to the leaderboard:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/aayu04shi/Crime-Analysis-in-India.git
cd Crime-Analysis-in-India
```

---

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

---

### 4️⃣ Train the Model

* Enter your **GitHub username** in the app
* Click **"Train / Retrain Model"**

✅ The system will:

* Train the ML model
* Calculate accuracy
* Save your score to the leaderboard

---

### 5️⃣ Update the Global Leaderboard

After training, push your score:

```bash
git add submissions/
git commit -m "My model submission"
git push
```


---


## 🏆 Leaderboard

👉 Click here to view full leaderboard:
 [View Full Leaderboard](http://localhost:8505/leaderboard)


| 🏅 | Rank | GitHub | Model | Accuracy |
|----|------|--------|-------|----------|
<!-- LEADERBOARD START -->
| 🥇 | 1 | Aayushi | RandomForest | 85.00% |
| 🥈 | 2 | user2 | XGBoost | 78.12% |
| 🥉 | 3 | user3 | GradientBoost | 75.45% |
<!-- LEADERBOARD END -->

---



