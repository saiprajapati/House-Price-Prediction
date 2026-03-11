import pandas as pd

#Load the dataset
data =  pd.read_csv('data/house_data.csv')

# Show first five rows of the dataset
print(data.head())

# Show dataset shape (rows, columns)
print("Shape:", data.shape)

#show dataset columns names
print("Columns:", data.columns)

print("\n--- Dataset Info ---")
print(data.info())

print("\n--- Statistical Summary ---")
print(data.describe(include='all'))

# Check for missing values
print(data.isnull().sum())
# Dataset has no missing values

binary_columns = [
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'prefarea'
]

for col in binary_columns:
    data[col] = data[col].map({'yes': 1, 'no': 0})

data = pd.get_dummies(
    data,
    columns=['furnishingstatus'],
    drop_first= True
)

print(data.head)
print(data.dtypes)

# Feature Scaling
X = data.drop('price', axis=1)
y = data['price']

# Train-test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size= 0.2, random_state=42
)

# Applying StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)

coefficients = pd.DataFrame(
    model.coef_,
    X.columns,
    columns=['Coefficient']
)

print(model.intercept_)

y_pred = model.predict(X_test_scaled)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2 ):", r2)