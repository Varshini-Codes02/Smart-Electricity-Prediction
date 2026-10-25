import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/energydata_complete.csv")

data["date"] = pd.to_datetime(data["date"])

data["hour"] = data["date"].dt.hour
data["day"] = data["date"].dt.day
data["month"] = data["date"].dt.month

print(data[["date", "hour", "day", "month"]].head(10))

X = data.drop(columns=["Appliances", "date"])
y = data["Appliances"]

print("Features:", X.shape)
print("Target:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed!")

y_pred = model.predict(X_test)

print("Prediction completed!")
print(y_pred[:10])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = mean_squared_error(y_test, linear_pred) ** 0.5
linear_r2 = r2_score(y_test, linear_pred)

print("Linear Regression MAE:", linear_mae)
print("Linear Regression RMSE:", linear_rmse)
print("Linear Regression R2:", linear_r2)

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=100,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_pred = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_pred)
gb_rmse = mean_squared_error(y_test, gb_pred) ** 0.5
gb_r2 = r2_score(y_test, gb_pred)

print("Gradient Boosting MAE:", gb_mae)
print("Gradient Boosting RMSE:", gb_rmse)
print("Gradient Boosting R2:", gb_r2)

def get_usage_level(energy):
    if energy < 60:
        return "Low"
    elif energy < 150:
        return "Medium"
    else:
        return "High"


usage_level = get_usage_level(y_pred[0])

print("Predicted Consumption:", y_pred[0], "Wh")
print("Usage Level:", usage_level)

current_hour = X_test.iloc[0]["hour"]

if 6 <= current_hour < 18:
    time_period = "Day"
else:
    time_period = "Night"

print("Time Period:", time_period)

if usage_level == "High" and time_period == "Night":
    recommendation = "High energy usage at night. Switch off unnecessary appliances and reduce high-power appliance usage."

elif usage_level == "High":
    recommendation = "High energy usage detected. Reduce the use of high-power appliances."

elif usage_level == "Medium" and time_period == "Night":
    recommendation = "Moderate night-time usage. Switch off lights and appliances that are not required."

elif usage_level == "Medium":
    recommendation = "Moderate energy usage. Switch off unnecessary appliances."

else:
    recommendation = "Energy usage is low. Continue using electricity efficiently."

print("Recommendation:", recommendation)

plt.figure(figsize=(10, 5))

plt.hist(y, bins=30)

plt.title("Appliances Energy Consumption Distribution")
plt.xlabel("Energy Consumption (Wh)")
plt.ylabel("Frequency")

plt.show()

print("\nEnter the values for prediction:")

hour = int(input("Hour (0-23): "))
day = int(input("Day (1-31): "))
month = int(input("Month (1-12): "))

user_input = X_test.iloc[0].copy()

user_input["hour"] = hour
user_input["day"] = day
user_input["month"] = month

user_input = user_input.values.reshape(1, -1)

predicted_energy = model.predict(user_input)[0]

print("\nPredicted Energy Consumption:", predicted_energy, "Wh")

usage_level = get_usage_level(predicted_energy)

print("Usage Level:", usage_level)

if 6 <= hour < 18:
    time_period = "Day"
else:
    time_period = "Night"

print("Time Period:", time_period)

if usage_level == "High" and time_period == "Night":
    recommendation = "High energy usage at night. Switch off unnecessary appliances and reduce high-power appliance usage."

elif usage_level == "High":
    recommendation = "High energy usage detected. Reduce the use of high-power appliances."

elif usage_level == "Medium" and time_period == "Night":
    recommendation = "Moderate night-time usage. Switch off lights and appliances that are not required."

elif usage_level == "Medium":
    recommendation = "Moderate energy usage. Switch off unnecessary appliances."

else:
    recommendation = "Energy usage is low. Continue using electricity efficiently."

print("Recommendation:", recommendation)
