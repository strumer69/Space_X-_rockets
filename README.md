# Predicting SpaceX Rocket Landings

This Jupyter Notebook is designed to predict whether SpaceX rockets will land successfully or not. A successful landing is indicated by `class = 1`, while a failed landing is indicated by `class = 0`.

## Table of Contents
1. [Loading and Inspecting the Data](#loading-and-inspecting-the-data)
2. [Data Exploration](#data-exploration)
3. [Data Manipulation](#data-manipulation)
4. [Statistical Analysis](#statistical-analysis)
5. [Visualizing Data](#visualizing-data)
6. [Categorical Data Analysis](#categorical-data-analysis)
7. [Conclusion](#conclusion)

## Loading and Inspecting the Data
First, we load the dataset and perform initial inspections to understand its structure and content.

```python
import os
import pandas as pd
import numpy as np

df = pd.read_csv('dataset_falcon9.csv')
```

- Display the DataFrame:
    ```python
    df
    ```
- Get information about the DataFrame:
    ```python
    df.info()
    ```

## Data Exploration
- Preview the first and last few rows of the DataFrame:
    ```python
    df.head()
    df.tail(3)
    ```

- View a specific column:
    ```python
    df['FlightNumber']
    ```

## Data Manipulation
- Drop rows and columns:
    ```python
    df.drop(2, axis=0)  # Drop a row
    df.drop('Date', axis=1)  # Drop a column temporarily
    ```

- Get column names and DataFrame shape:
    ```python
    df.columns
    df.shape
    ```

- Add a new row to the DataFrame:
    ```python
    new_row = {'FlightNumber': 11, 'Date': 2, 'BoosterVersion': 3, 'PayloadMass': 4, 'Orbit': 5,
               'LaunchSite': 6, 'Outcome': 7, 'Flights': 8, 'GridFins': 9, 'Reused': 10, 'Legs': 11,
               'LandingPad': 12, 'Block': 13, 'ReusedCount': 14, 'Serial': 15, 'Longitude': 16, 'Latitude': 17,
               'Class': 18}
    df2 = pd.DataFrame(new_row, index=[0])
    df2 = pd.concat([df, df2], ignore_index=True)
    df2.tail()
    ```

## Statistical Analysis
- Analyzing numerical columns:
    ```python
    df['PayloadMass'].min()
    df['PayloadMass'].max()
    df['PayloadMass'].mean()
    df['PayloadMass'].std()
    df['PayloadMass'].describe()
    ```

## Visualizing Data
- Plot histograms and other visualizations:
    ```python
    df['PayloadMass'].hist()
    df['PayloadMass'].plot()
    ```

- Analyzing categorical columns:
    ```python
    set(df['Orbit'])
    df['Orbit'].unique()
    df['Orbit'].value_counts()
    df['Orbit'].hist()
    ```

## Categorical Data Analysis
- Create DataFrames for successful and failed landings:
    ```python
    Success_df = df[df['Class'] == 1]
    Fail_df = df[df['Class'] == 0]
    Success_df.info()
    Fail_df['Class'].head()
    ```

- Explore categorical columns:
    ```python
    print('LaunchSite:', set(df['LaunchSite']))
    print('Outcome:', set(df['Outcome']))
    print('Flights:', set(df['Flights']))
    
    for i in range(len(df.columns)):
        print(df.columns[i], ":")
        print(set(df[df.columns[i]]))
        print('---------')
    ```

- Outcome analysis for successful and failed landings:
    ```python
    print('Success:\n', Success_df['Outcome'].value_counts())
    print('-----------')
    print('Fail:\n', Fail_df['Outcome'].value_counts())
    ```

- Focus on categorical and boolean values:
    ```python
    print('------------LandingPad:\n', df['LandingPad'].value_counts())
    print('------------Block:\n', df['Block'].value_counts())
    print('------------ReusedCount:\n', df['ReusedCount'].value_counts())
    print('------------GridFins:\n', df['GridFins'].value_counts())
    print('------------Reused:\n', df['Reused'].value_counts())
    print('------------Legs:\n', df['Legs'].value_counts())
    ```

## Conclusion
This notebook provides a comprehensive analysis of SpaceX rocket landing data. By exploring and manipulating the data, we can gain insights into various factors that contribute to the success or failure of rocket landings. Further steps could include advanced modeling and predictive analysis to improve the accuracy of these predictions.
