Air-Quality-Analysis-and-Prediction--ML
Air Quality Analysis and Prediction using Python &amp; Machine Learning | EDA, Data Visualization, Regression and Classification on CPCB Tamil Nadu Air Pollution Dataset (2015).
 🌍 Air Quality Analysis and Prediction Using Machine Learning

 A Case Study of Tamil Nadu, India (2015)

 📌 Project Overview

This project focuses on analysing historical air pollution data collected by the Central Pollution Control Board (CPCB) across different locations in Tamil Nadu during 2015.

The objective is to explore pollution patterns, identify relationships between atmospheric pollutants, and develop machine learning models to predict PM10 concentration and classify air quality levels.

 🎯 Objectives

- Analyse historical air pollution data.
- Perform data cleaning and preprocessing.
- Conduct Exploratory Data Analysis (EDA).
- Visualize PM10 distribution and pollution trends.
- Identify correlations between SO2, NO2 and PM10.
- Build regression models for PM10 prediction.
- Build classification models for pollution categorization.
- Compare model performance using evaluation metrics.

 📂 Dataset

**Source:** Central Pollution Control Board (CPCB)

**Dataset:** Daily Air Quality Monitoring Data – Tamil Nadu (2015)

Main features:
- SO2 – Sulphur Dioxide
- NO2 – Nitrogen Dioxide
- RSPM/PM10 – Respirable Suspended Particulate Matter
- Sampling Date
- City/Town/Village/Area

PM2.5 was excluded because of extensive missing values.

 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

 🔍 Exploratory Data Analysis

The project includes:

1. PM10 distribution histogram.
2. City-wise average PM10 comparison.
3. Monthly PM10 trend analysis.
4. Pollutant correlation heatmap.

Key EDA Findings

- PM10 distribution shows positive skewness.
- Thoothukudi records the highest average PM10 among the displayed cities.
- Coimbatore records the lowest average.
- July shows the highest monthly average.
- SO2 has a stronger positive correlation with PM10 (0.48) than NO2 (0.13).

🤖 Machine Learning Algorithms

1. Regression Models

Target variable: PM10 concentration.

| Algorithm | R² Score |
|---|---:|
| Simple Linear Regression | -0.025 |
| Multiple Linear Regression | 0.262 |
| Polynomial Regression | 0.257 |

2. Classification Models

Target variable: PM10 pollution category.

Categories:
- Good: 0–50
- Satisfactory: Above 50–100
- Moderate or Above: Above 100

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 56.70% |
| K-Nearest Neighbors (K=5) | 60.88% |
| Decision Tree | 58.20% |

📊 Results and Insights

- Multiple Linear Regression achieved the highest regression R² score of 0.262.
- KNN achieved the highest classification accuracy of 60.88%.
- SO2 showed a stronger linear relationship with PM10 than NO2.
- Pollution levels varied across cities and months.

📈 Model Evaluation Metrics

**Regression:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

**Classification:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

📁 Project Structure

```text
Air-Quality-Analysis-and-Prediction-ML/
│
├── Air pollution.py
├── cpcb_dly_aq_tamil_nadu-2015.csv
├── README.md
├── requirements.txt
│
└── outputs/
    ├── pm10_distribution.png
    ├── average_pm10_city.png
    ├── monthly_pm10_trend.png
    ├── correlation_heatmap.png
    ├── regression_comparison.png
    └── confusion_matrices.png
```

⚙️ How to Run

1. Clone the repository.

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Place the dataset in the project directory.

4. Run the Python file.

```bash
python "Air pollution.py"
```

🚀 Future Improvements

- Incorporate meteorological parameters such as temperature, rainfall, wind speed and humidity.
- Improve model performance through hyperparameter tuning.
- Explore Random Forest and other ensemble algorithms.
- Apply time-series forecasting techniques.
- Develop an interactive air quality monitoring dashboard.

👨‍💻 Author

**Damuluri Dinesh Kartik**

B.Tech – Computer Science and Engineering

Domain: Data Science

Lovely Professional University

---

*This project is an academic machine learning study based on historical CPCB observations. It is not a real-time air quality monitoring or forecasting service.*
