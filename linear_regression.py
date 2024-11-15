import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Temperature
x = np.array([51, 52, 67, 65, 70, 69, 72, 75, 73, 81, 78, 83]).reshape(-1, 1) # reshape to have 2 dimensional matrix

# Total number of ice cream sellers on the day
y = np.array([1, 0, 14, 14, 23, 20, 23, 26, 22, 30, 26, 36])

model = LinearRegression()

# Training
model.fit(x, y)

# Predict
y_predict = model.predict(x)

# Visualization
plt.scatter( x, y, color='blue', label='Original data')
plt.plot(x, y_predict, color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print(f"Coeficiente angular (b1): {model.coef_[0]}")
print(f"Intercepto (b0): {model.intercept_}")