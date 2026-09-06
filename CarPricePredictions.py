''' 
Car Price Prediction - CodeAlpha Internship 
Name: Siyolise Mbadu
Task: Train a regression model to predict used car selling prices 
'''

#--Imports--
import os 
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#--Creating output folders upfront so nothing fails later trying to save into them--
os.makedirs("images", exist_ok=True)
os.makedirs("results", exist_ok=True)


print("\n", "=" * 50)
print("1. DATA LOADING")
print("=" * 50)

#--Loading dataset from CSV
df = pd.read_csv("dataset/car data.csv")
print("Data loaded successfully")


print("\n", "=" * 50)
print("2. DATA EXPLORATION")
print("=" * 50)

#--Basic overview of the dataset: shape, stats, structure--
print("First 5 rows:")
print(df.head())
print("\n dataFrame shape:")
print(df.shape)
print("\n statistics:")
print(df.describe())
print("\n dataset info:")
print(df.info())


print("\n", "=" * 50)
print("3. DATA CLEANING")
print("=" * 50)

#--Checking for missing values--
missing_values = df.isnull().sum()
if missing_values.sum() > 0:
    print(f"Missing values found: {missing_values}")
    print("Dropping rows with missing values")
    df = df.dropna()
else:
    print("No missing values found")
        
#--Check for and remove duplicate rows--
duplicates_no = df.duplicated().sum()
if duplicates_no > 0: 
    print(f"Removing {duplicates_no} duplicate rows")
    df = df.drop_duplicates()
else: 
    print("No duplicate rows found")


print("\n", "=" * 50)
print("4. FEATURE ENGINEERING")
print("=" * 50)

#-- Turning 'Year' into 'Car_Age', Its more useful for predictions rather than raw year.--
#-- Using 2026 as the reference year since thats the current year for this project.--
df["Car_Age"] = 2026 - df["Year"]
df = df.drop("Year", axis = 1)

#--Droping Car_Name--
#--Because there are too many car models, and our data can also be having a large diverse number of car models --
#--Dropping them, I believe would be useful for this model because it has too many unique names--
#--An alternative would be extracting a brand from the name--
#--The present price and the brand-level features already tell us most of what the name would.--
df = df.drop("Car_Name", axis=1)

#--Encoding categorical columns (Fuel_type, Seller_Type, Transmission)--
#--Reason for this is, in ML we have to change each category into numeric values--
categorical_cols = df.select_dtypes(include="str").columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("\n Data after feature engineering: ")
print(df.head())


print("\n", "=" * 50)
print("5. DATA VISUALIZATION")
print("=" * 50)

Target_Column = "Selling_Price"

# --Distributing the target variable (selling price)--
plt.figure(figsize=(8, 6))
plt.hist(df[Target_Column], bins=30, edgecolor="black")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.title("Distribution of Car Selling Prices")
plt.savefig("images/price_distribution.png", dpi=150, bbox_inches="tight") #save BEFORE show
plt.show()

#--Present price vs selling price (expected to be strongly correlated)--
plt.figure(figsize=(8, 6))
plt.scatter(df["Present_Price"], df[Target_Column], alpha=0.6)
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Present Price vs Selling Price")
plt.savefig("images/present_vs_selling_price.png", dpi=150, bbox_inches="tight") #save BEFORE showing
plt.show()

#--Correlation heatmap of all numeric features--
plt.figure(figsize=(9, 7))
corre = df.corr(numeric_only=True)
im = plt.imshow(corre, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(corre.columns)), corre.columns, rotation=90)
plt.yticks(range(len(corre.columns)), corre.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("images/correlation_matrix.png", dpi=150, bbox_inches="tight") #always save 
plt.show()

#--Feature/Target split--
x = df.drop(Target_Column, axis=1)
y = df[Target_Column]

#--Trai/test split--
x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size=0.2, random_state=42
)

#--Feature Scaling--
#--Linear Regression and distance-based comparisons benefit from scaled features.--
#--Fitting the scaler on training data only, then I apply it to test data.--
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


print("\n", "=" * 50)
print("6. BASELINE MODEL (Linear Regression)")
print("=" * 50)

#--Training a simple Linear Regression model as a baseline--
baseline_model = LinearRegression()
baseline_model.fit(x_train_scaled, y_train)
baseline_predictions = baseline_model.predict(x_test_scaled)

baseline_average_error = mean_absolute_error(y_test, baseline_predictions)
baseline_root_error = np.sqrt(mean_squared_error(y_test, baseline_predictions))
baseline_accuracy_explanation = r2_score(y_test, baseline_predictions)

print(f"Baseline Mean Absolute Error: {baseline_average_error:.4f}")
print(f"Baseline Root Mean Squared Error: {baseline_root_error:.4f}")
print(f"Baseline R² Coe_of_determination: {baseline_accuracy_explanation:.4f}")


print("\n", "=" * 50)
print("7. FINDING THE BEST GRADIENT BOOSTING MODEL (Cross-Validation)")
print("=" * 50)

#--Testing a range of learning rates using 5-fold cross-validation--
#--Gradient Boosting builds trees one at a time, each correcting the errors of the previous ones.--
#--Meanwhile the learning_rate controls how much each new tree can adjust the prediction.--
learning_rate_range = [0.01, 0.05, 0.1, 0.2, 0.3]
cv_scores = []

for lr in learning_rate_range:
    gbr = GradientBoostingRegressor(learning_rate=lr, n_estimators=200, random_state=42)
    scores = cross_val_score(gbr, x, y, cv=5, scoring="r2")
    cv_scores.append(scores.mean())

best_learning_rate = learning_rate_range[np.argmax(cv_scores)]
print(f"Best learning rate: {best_learning_rate}, CV R2: {max(cv_scores):.4f}")

#--Plotting: R² Across tested learning rate values--
plt.figure(figsize=(8, 5))
plt.plot(learning_rate_range, cv_scores, marker="o")
plt.xlabel("Learning Rate")
plt.ylabel("Cross-Validated R²")
plt.title("Gradient Boosting: R² vs Learning Rate")
plt.grid(True)
plt.savefig("images/R2_vs_learning_rate.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n", "=" * 50)
print("8. FINAL MODEL (Gradient Boosting)")
print("=" * 50)

#--Training the final model using the best learning rate found above--
#--Gradient Boosting doesnt require scaled input, so we train it on the unscaled train/test 
# features directly (unlike the Linear Regression Baseline).--
final_model = GradientBoostingRegressor(
    learning_rate=best_learning_rate, n_estimators=200, random_state=42
)
final_model.fit(x_train, y_train)

#--Saving the trained model and scaler for reuse without retraining--
#--The scaler is still saved in case predictions are ever made using a 
#--scaled-input mode (for example: The Baseline) alongside this one.--
joblib.dump(final_model, "results/car_price_model.joblib")
joblib.dump(scaler, "results/scaler.joblib")
print("Model and scaler saved to results")

final_predictions = final_model.predict(x_test)

final_average_error = mean_absolute_error(y_test, final_predictions)
final_root_error = np.sqrt(mean_squared_error(y_test, final_predictions))
final_accuracy_explanation = r2_score(y_test, final_predictions)

print(f"Final Mean Absolute Error: {final_average_error:.4f}")
print(f"Final Root Mean Squared Error: {final_root_error:.4f}")
print(f"Final R² (Coe_of_determination): {final_accuracy_explanation:.4f}")


print("\n", "=" * 50)
print("9. MODEL EVALUATION")
print("=" * 50)

#--Residual plotting: how far off were the predictions, and in which direction--
residuals = y_test - final_predictions

fig, axes = plt.subplots(1,2, figsize=(12, 5))
axes[0].scatter(final_predictions, residuals, alpha=0.6)
axes[0].axhline (0, color="red", linestyle= "--")
axes[0].set_xlabel("Predicted Selling Price")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residuals vs Predicted")

axes[1].hist(residuals, bins=30, edgecolor="black")
axes[1].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("images/residuals.png", dpi = 150, bbox_inches= "tight")
plt.show()

# --Featuring importance: which features drove the predictions the most--
importances = pd.Series(final_model.feature_importances_, index=x.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(8,6))
importances.head(10).plot(kind="barh")
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("images/feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()


print("\n", "=" * 50)
print("10. SAVING RESULTS")
print("=" * 50)

#--Saing a plain text summary of results--
summary = (
    f"Baseline (Linear Regression) - Mean Average Error: {baseline_average_error:.4f}, "
    f"Root Error: {baseline_root_error:.4f}, Baseline R²: {baseline_accuracy_explanation:.4f}\n"
    f"Final (Gradient Boosting, learning_rate={best_learning_rate}) - "
    f"Mean Average Error: {final_average_error:.4f}, Final Root Error: {final_root_error:.4f}, Final R²: {final_accuracy_explanation:.4f} "
)
with open("results/model_summary.txt", "w") as f:
    f.write(summary)
print(summary)

#--Auto-generate a ready-to-paste Results section for the README--
# Pulls every value directly from the results above, so its always accurate.
results_block = f"""## Results
- **Model:** Gradient Boosting Regressor (learning_rate={best_learning_rate})
- **Baseline model:** Linear Regression

| Model | Mean Abs Error | Root Mean Squared Error | R² |
|-------|----------------|-------------------------|----|
| Linear Regression (baseline) | {baseline_average_error:.2f} | {baseline_root_error:.2f} | {baseline_accuracy_explanation:.2f} |
| Gradient Boosting (final) | {final_average_error:.2f} | {final_root_error:.2f} | {final_accuracy_explanation:.2f} |

Full summary: [results/model_summary.txt](results/model_summary.txt)
"""

with open("results/results_section.md", "w") as f:
    f.write(results_block)

print("\n Results section written to results_section.md - copy this into your README.")