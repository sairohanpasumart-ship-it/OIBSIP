import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('Task 3/car data.csv')

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["Fuel_Type"] = df["Fuel_Type"].str.strip().str.title()
df["Selling_type"] = df["Selling_type"].str.strip().str.title()
df["Transmission"] = df["Transmission"].str.strip().str.title()

df["Car_Age"] = 2020 - df["Year"]
df["Brand"] = df["Car_Name"].str.split().str[0]

df.drop(columns=["Car_Name", "Year"], inplace=True)

sns.histplot(df["Selling_Price"], kde=True)
plt.title("Distribution of Selling Price")
plt.show()

sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price")
plt.title("Selling Price by Fuel Type")
plt.show()

sns.scatterplot(data=df, x="Car_Age", y="Selling_Price")
plt.title("Selling Price vs Car Age")
plt.show()

df = pd.get_dummies(
    df,
    columns=["Brand", "Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, lr_pred),
        mean_absolute_error(y_test, rf_pred)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, lr_pred)),
        np.sqrt(mean_squared_error(y_test, rf_pred))
    ],
    "R²": [
        r2_score(y_test, lr_pred),
        r2_score(y_test, rf_pred)
    ]
})

print(results)

importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(
    x=importance.values,
    y=importance.index
)
plt.title("Random Forest Feature Importance")
plt.show()