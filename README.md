# SpaceX Rocket Landing Prediction

## Objective
The goal of this project is to predict whether the rockets of SpaceX will land successfully or not. If the class is 0, it means the landing will fail; otherwise, if the class is 1, the landing will be successful.

## Dataset
The dataset used for this project is `dataset_falcon9.csv`. This dataset contains 90 rows and 18 columns.

## Data Exploration
### Import Libraries
```python
import os
import pandas as pd
import numpy as np
```

### Load Dataset
```python
df = pd.read_csv('dataset_falcon9.csv')
```

### Display Dataset
```python
df.head()  # Display the first 5 rows
df.tail(3)  # Display the last 3 rows
```

### Data Information
```python
df.info()
```
- The dataset has 90 rows and 18 columns.
- There are no missing values except for the `LandingPad` column, which has 26 missing values.

### Column Operations
- Display a specific column:
  ```python
  df['FlightNumber']
  ```
- Drop a row:
  ```python
  df.drop(2, axis=0)  # Temporarily drop the second row
  ```
- Drop a column:
  ```python
  df.drop('Date', axis=1)  # Temporarily drop the 'Date' column
  ```

### Column Names and Shape
```python
df.columns  # Display column names
df.shape  # Display dataset shape
```

## Data Manipulation
### Adding a Row
```python
new_row = {'FlightNumber': 11, 'Date': 2, 'BoosterVersion': 3, 'PayloadMass': 4, 'Orbit': 5,
           'LaunchSite': 6, 'Outcome': 7, 'Flights': 8, 'GridFins': 9, 'Reused': 10, 'Legs': 11,
           'LandingPad': 12, 'Block': 13, 'ReusedCount': 14, 'Serial': 15, 'Longitude': 16, 'Latitude': 17,
           'Class': 18}
df2 = pd.DataFrame(new_row, index=[0])
df = pd.concat([df, df2], ignore_index=True)
```

### Analyzing Specific Columns
- Unique values in `BoosterVersion`:
  ```python
  set(df['BoosterVersion'])
  ```
- Min and max of `PayloadMass`:
  ```python
  df['PayloadMass'].min()
  df['PayloadMass'].max()
  ```
- Mean and standard deviation of `PayloadMass`:
  ```python
  df['PayloadMass'].mean()
  df['PayloadMass'].std()
  ```

### Descriptive Statistics
```python
df['PayloadMass'].describe()
```

### Visualizing Data
- Histogram:
  ```python
  df['PayloadMass'].hist()
  ```
- Plot:
  ```python
  df['PayloadMass'].plot()
  ```

### Unique Values and Value Counts
- Unique values in `Orbit`:
  ```python
  df['Orbit'].unique()
  ```
- Value counts:
  ```python
  df['Orbit'].value_counts()
  ```

### Success and Fail DataFrames
```python
Success_df = df[df['Class'] == 1]
Fail_df = df[df['Class'] == 0]
```

### Analyzing Specific Columns in New DataFrames
```python
Success_df.info()
Fail_df['Class'].head()
```

## Data Cleaning
- Drop unnecessary columns:
  ```python
  df = df.drop(['BoosterVersion', 'Serial', 'Longitude', 'Latitude'], axis=1)
  df.info()  # Number of columns changed from 18 to 14
  ```

## Data Visualization
### Install Required Libraries
```python
!pip install matplotlib
!pip install seaborn
```
```python
import seaborn as sns
import matplotlib.pyplot as plt
```

### Visualize Relations
- PayloadMass vs FlightNumber:
  ```python
  sns.catplot(y="PayloadMass", x="FlightNumber", data=df, aspect=5)
  plt.xlabel("Flight Number", fontsize=20)
  plt.ylabel("Payload Mass (kg)", fontsize=20)
  plt.show()
  ```
- LaunchSite vs FlightNumber:
  ```python
  sns.catplot(y="LaunchSite", x="FlightNumber", hue="Class", data=df, aspect=5)
  plt.xlabel("Flight Number", fontsize=20)
  plt.ylabel("LaunchSite", fontsize=20)
  plt.show()
  ```
- LaunchSite vs PayloadMass:
  ```python
  sns.catplot(x="LaunchSite", y="PayloadMass", hue="Class", data=df, aspect=5)
  plt.xlabel("LaunchSite", fontsize=20)
  plt.ylabel("Payload Mass (kg)", fontsize=20)
  plt.show()
  ```

### Visualize Numerical and Categorical Data
- Distribution of `PayloadMass`:
  ```python
  sns.displot(df['PayloadMass'])
  ```
- Countplot of `LaunchSite`:
  ```python
  sns.countplot(x="LaunchSite", data=df)
  ```

## Note
- Use `sns.countplot` for categorical data.
- Use `sns.displot` for numerical data.

## Conclusion
This project involves exploring the SpaceX rocket launch dataset to predict the success of rocket landings. Data cleaning, manipulation, and visualization techniques were applied to understand and prepare the data for modeling.
