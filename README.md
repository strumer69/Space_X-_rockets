# SpaceX Rocket Landing Prediction

## Project Overview

This project aims to predict whether SpaceX rockets will land successfully or not. The target variable is a binary class where:
- **Class = 0**: The landing will fail.
- **Class = 1**: The landing will be successful.

## Dataset

The dataset used for this analysis is `dataset_falcon9.csv`. It contains various attributes related to each rocket launch, which we will explore to understand the relationship between different features and the landing outcome.

## Importing Libraries and Data

```python
import os
import pandas as pd
import numpy as np

# Set working directory
os.getcwd()

# Load the dataset
df = pd.read_csv('dataset_falcon9.csv')
```

### Initial Data Exploration

- Use the `.info()` method to get an overview of the dataset.
```python
df.info()
```
- The dataset consists of 90 rows (0 to 89) and 18 columns, with most columns containing non-null values. The `LandingPad` column has 26 missing values.

### Viewing the Data

- Check the first few rows:
```python
df.head()
```

- Check the last few rows:
```python
df.tail(3)
```

- Access a specific column:
```python
df['FlightNumber']
```

### Data Cleaning

- Drop a specific row or column:
```python
df.drop(2, axis=0)  # Drops the third row
df.drop('Date', axis=1)  # Drops the 'Date' column (temporary)
```

- Use the `inplace=True` parameter to make permanent changes:
```python
df.drop('Date', axis=1, inplace=True)
```

- Check the column names and shape of the dataset:
```python
df.columns
df.shape
```

### Adding a Row

To add a new row to the DataFrame, create a dictionary with the new data and use `pd.concat()`:

```python
new_row = {
    'FlightNumber': 11,
    'Date': '2024-01-01',
    'BoosterVersion': 'Falcon 9',
    'PayloadMass': 4500,
    'Orbit': 'LEO',
    'LaunchSite': 'CCAFS SLC 40',
    'Outcome': 'Success',
    'Flights': 10,
    'GridFins': 1,
    'Reused': 0,
    'Legs': 1,
    'LandingPad': '5e9e3032383ecb6bb234e7ca',
    'Block': 5,
    'ReusedCount': 1,
    'Serial': 'B1049',
    'Longitude': -80.577366,
    'Latitude': 28.561857,
    'Class': 1
}

df2 = pd.DataFrame(new_row, index=[0])
df2 = pd.concat([df, df2], ignore_index=True)
df2.tail()
```

### Data Analysis

- Analyze unique values in the `BoosterVersion` and `PayloadMass` columns:
```python
set(df['BoosterVersion'])
min(df['PayloadMass'])
max(df['PayloadMass'])
df['PayloadMass'].mean()
df['PayloadMass'].std()
```

- Visualize the distribution of `PayloadMass`:
```python
df['PayloadMass'].hist()
```

### Filtering Success and Failure DataFrames

Create two separate DataFrames for successful and failed landings:
```python
Success_df = df[df['Class'] == 1]
Fail_df = df[df['Class'] == 0]
```

### Data Visualization

Use Matplotlib and Seaborn to visualize relationships between variables:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.catplot(y="PayloadMass", x="FlightNumber", data=df, aspect=5)
plt.xlabel("Flight Number", fontsize=20)
plt.ylabel("Payload Mass (kg)", fontsize=20)
plt.show()
```

### Data Preprocessing

Convert categorical variables to numerical data and handle missing values:

```python
df['GridFins'] = df['GridFins'].astype(int)
df['Reused'] = df['Reused'].astype(int)
df['Legs'] = df['Legs'].astype(int)

# Convert categorical columns to dummy variables
df_dummy = pd.get_dummies(df[['Orbit', 'LaunchSite', 'Outcome', 'LandingPad']])
df = pd.concat([df, df_dummy], axis=1)

# Drop original categorical columns
df.drop(['Orbit', 'LaunchSite', 'Outcome', 'LandingPad', 'Date'], axis=1, inplace=True)
```

### Model Training with Logistic Regression

Train a logistic regression model on the preprocessed dataset:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

# Create and fit the model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

# Make predictions
prediction = logmodel.predict(X_test)
```

### Model Evaluation

Evaluate the model performance using confusion matrix and accuracy score:

```python
from sklearn.metrics import confusion_matrix, accuracy_score

confusion = confusion_matrix(y_test, prediction)
accuracy = accuracy_score(y_test, prediction, normalize=True)

print("Confusion Matrix:\n", confusion)
print("Accuracy:", accuracy)
```

## Conclusion

This project provides a comprehensive analysis of the dataset to predict the successful landing of SpaceX rockets using logistic regression. The exploratory data analysis, preprocessing, and model evaluation steps are critical for understanding and improving the model's performance.

### References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
