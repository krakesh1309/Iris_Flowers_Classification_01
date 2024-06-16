# Iris Flowers Classification_01

This project demonstrates the classification of Iris flowers based on their features using the K-Nearest Neighbors (KNN) algorithm. The dataset contains 150 samples of Iris flowers with four features: sepal length, sepal width, petal length, and petal width. The goal is to classify these samples into three species: Iris-setosa, Iris-versicolor, and Iris-virginica.

## Dataset

The dataset used is the famous Iris dataset, which contains the following features:
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

Each sample in the dataset is labeled with one of three species:
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

## Installation

To run this project, you need Python and several Python libraries. The easiest way to install these libraries is using `pip`.

1. **Create and activate a virtual environment** (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

## Libraries Used

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical computations
- **scikit-learn**: For machine learning algorithms and tools
- **matplotlib**: For plotting and visualization

## Usage

Here's an example of how to use the K-Nearest Neighbors classifier with the Iris dataset.

```python
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create an instance of KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Print the predictions
print(y_pred)

# Optional: Visualize the results
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Iris Classification')
plt.show()

## Results
The K-Nearest Neighbors algorithm is used to classify the Iris flowers into one of the three species. The performance of the classifier can be evaluated using various metrics such as accuracy, precision, recall, and F1-score.
