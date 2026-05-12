import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


#  Load Dataset

df = pd.read_csv("data.csv")

#  Basic Cleaning


# Remove zero prices
df = df[df['price'] > 0]

# Remove extreme outliers (top 1%)
upper_limit = df['price'].quantile(0.99)
df = df[df['price'] < upper_limit]

# Drop only unnecessary columns (NOT city/statezip)
df = df.drop(['date', 'street', 'country'], axis=1)

# Encode location columns
df = pd.get_dummies(df, columns=['city', 'statezip'], drop_first=True)

#  Feature Engineering

df['house_age'] = 2025 - df['yr_built']
df['renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)


#  Define Features & Target

X = df.drop('price', axis=1)


y = np.log1p(df['price'])


#  Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#  Train Model

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)


#  Prediction

y_pred_log = model.predict(X_test)

# Convert back to original scale
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)


#  Evaluation

print("\nModel Performance")
print("------------------")
print("MAE:", mean_absolute_error(y_test_actual, y_pred))
print("R2 Score:", r2_score(y_test_actual, y_pred))


#  Feature Importance

importance = pd.Series(model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

print("\nTop 5 Important Features:")
print(importance.head())


#  Visualizations


# Living Area vs Price
plt.figure()
plt.scatter(df['sqft_living'], df['price'])
plt.xlabel("Living Area")
plt.ylabel("Price")
plt.title("Living Area vs Price")
plt.show()

# Actual vs Predicted (after inverse transform)
plt.figure()
plt.scatter(y_test_actual, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

input("Press Enter to close graphs...")
print("Total columns after encoding:", df.shape[1])