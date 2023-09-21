# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Define the data
data = {
    'Hours': [2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8],
    'Scores': [21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['Hours']]
y = df['Scores']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict the score for 9.25 hours of study
predicted_score = model.predict([[9.25]])
print(f"Predicted score for 9.25 hours of study: {predicted_score[0]}")

# Plot the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel('Hours of Study')
plt.ylabel('Scores')
plt.title('Hours vs Scores')
plt.show()
