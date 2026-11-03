# ==========================================================
# AIR POLLUTION PREDICTION - TAMIL NADU (2015)
# Simple syllabus-based code for Spyder
# ==========================================================

# 1. Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)


# 2. Load Dataset
df = pd.read_csv("C:/Users/Lenovo/Downloads/cpcb_dly_aq_tamil_nadu-2015.csv")

print(df.head())
print(df.info())
print(df.describe())


# 3. Data Preprocessing
df['Sampling Date'] = pd.to_datetime(
    df['Sampling Date'], dayfirst=True, errors='coerce'
)

for col in ['SO2', 'NO2', 'RSPM/PM10']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['Sampling Date', 'SO2', 'NO2', 'RSPM/PM10'])
df = df.drop_duplicates()
df = df.drop(columns=['PM 2.5'])       # Excluded due to many missing values

df['Month'] = df['Sampling Date'].dt.month
df['Day'] = df['Sampling Date'].dt.day

print("\nCleaned Dataset Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())


# 4. Exploratory Data Analysis (EDA)

# PM10 distribution
sns.histplot(df['RSPM/PM10'], kde=True, color='steelblue')
plt.title("PM10 Distribution")
plt.xlabel("PM10 Concentration")
plt.show()

# City-wise average PM10
df.groupby('City/Town/Village/Area')['RSPM/PM10'].mean().sort_values().plot(
    kind='bar', title="Average PM10 by City"
)
plt.ylabel("Average PM10")
plt.tight_layout()
plt.show()

# Monthly PM10 trend
df.groupby('Month')['RSPM/PM10'].mean().plot(
    kind='line', marker='o', title="Monthly Average PM10"
)
plt.xlabel("Month")
plt.ylabel("Average PM10")
plt.grid()
plt.show()

# Correlation heatmap
sns.heatmap(df[['SO2', 'NO2', 'RSPM/PM10']].corr(),
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Pollutant Correlation")
plt.show()


# 5. Correlation Values
print("\nCorrelation Matrix")
print(df[['SO2', 'NO2', 'RSPM/PM10']].corr())


# ==========================================================
# 6. REGRESSION
# Target: PM10 concentration
# ==========================================================

print("\n========== REGRESSION MODELS ==========")

y = df['RSPM/PM10']

# Same train-test rows are used for all regression models
train_idx, test_idx = train_test_split(
    df.index, test_size=0.2, random_state=42
)
y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

# 7. Simple Linear Regression (NO2 -> PM10)
X_simple = df[['NO2']]
model1 = LinearRegression()
model1.fit(X_simple.loc[train_idx], y_train)
pred1 = model1.predict(X_simple.loc[test_idx])

# 8. Multiple Linear Regression (SO2, NO2, Month, Day -> PM10)
X_multi = df[['SO2', 'NO2', 'Month', 'Day']]
model2 = LinearRegression()
model2.fit(X_multi.loc[train_idx], y_train)
pred2 = model2.predict(X_multi.loc[test_idx])

# OLS estimation: LinearRegression estimates coefficients
# by minimizing the sum of squared residuals.
print("\nOLS Intercept:", model2.intercept_)
print("OLS Coefficients:", model2.coef_)

# 9. Polynomial Regression (degree 2)
X_poly = df[['SO2', 'NO2']]
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_poly.loc[train_idx])
X_test_poly = poly.transform(X_poly.loc[test_idx])

model3 = LinearRegression()
model3.fit(X_train_poly, y_train)
pred3 = model3.predict(X_test_poly)


# 10. Regression Model Evaluation
reg_predictions = [pred1, pred2, pred3]
reg_names = ['Simple Linear', 'Multiple Linear', 'Polynomial']

for name, pred in zip(reg_names, reg_predictions):
    print("\n", name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2 Score:", r2_score(y_test, pred))

# Regression comparison graph
pd.Series(
    [r2_score(y_test, p) for p in reg_predictions],
    index=reg_names
).plot(kind='bar', title="Regression Model Comparison (R2)")
plt.ylabel("R2 Score")
plt.tight_layout()
plt.show()


# ==========================================================
# 11. CLASSIFICATION
# Target: PM10 pollution category
# ==========================================================

print("\n========== CLASSIFICATION MODELS ==========")

# Good: 0-50; Satisfactory: >50-100; Moderate or Above: >100
df['PM10_Category'] = pd.cut(
    df['RSPM/PM10'],
    bins=[-np.inf, 50, 100, np.inf],
    labels=['Good', 'Satisfactory', 'Moderate or Above']
)
print("\nCategory Counts:\n", df['PM10_Category'].value_counts())

# PM10 is not an input feature (avoids target leakage)
X = df[['SO2', 'NO2', 'Month', 'Day']]
y_class = df['PM10_Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class
)

# Scale features for Logistic Regression and KNN
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 12. Logistic Regression
model4 = LogisticRegression(max_iter=1000)
model4.fit(X_train_scaled, y_train)
pred4 = model4.predict(X_test_scaled)

# 13. K-Nearest Neighbors
model5 = KNeighborsClassifier(n_neighbors=5)
model5.fit(X_train_scaled, y_train)
pred5 = model5.predict(X_test_scaled)

# 14. Decision Tree
model6 = DecisionTreeClassifier(random_state=42)
model6.fit(X_train, y_train)
pred6 = model6.predict(X_test)


# 15. Classification Evaluation
class_predictions = [pred4, pred5, pred6]
class_names = ['Logistic Regression', 'KNN', 'Decision Tree']
labels = ['Good', 'Satisfactory', 'Moderate or Above']

for name, pred in zip(class_names, class_predictions):
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred,
                                        average='weighted', zero_division=0))
    print("Recall:", recall_score(y_test, pred,
                                     average='weighted', zero_division=0))
    print("F1 Score:", f1_score(y_test, pred,
                                average='weighted', zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Classification accuracy comparison
pd.Series(
    [accuracy_score(y_test, p) for p in class_predictions],
    index=class_names
).plot(kind='bar', title="Classification Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# 16. Project Completion
print("\n========== PROJECT COMPLETED ==========")
