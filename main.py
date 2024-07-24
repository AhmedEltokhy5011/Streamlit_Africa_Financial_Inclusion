### imoprt libraries
import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

### Load the dataset
## Prefix your string with r to make it a raw string, which treats backslashes as literal characters.
data_df = pd.read_csv(r"https://github.com/AhmedEltokhy5011/Streamlit_Africa_Financial_Inclusion/blob/main/Financial_inclusion_dataset.csv")

## Adjust display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 400)        # Adjust the width to fit the screen

### Display general information about the dataset
print(data_df.info(), '\n')
print(data_df.head(), '\n')
print(data_df.describe(), '\n')


### check for missing data
print(data_df.isnull().sum(), '\n')  # no missing data

### check for duplicates
print('sum of duplicates=',data_df.duplicated().sum(), '\n')  # no duplicates found


### Handle Outliers
## Visualize outliers using boxplots and save it
sns.boxplot(data=data_df[['age_of_respondent', 'household_size']])
plt.title('Boxplot of Age and Household Size')
plt.savefig('boxplot of Age and Household Size.png')
plt.close()  # Close the figure to free up memory


### Encoding
# get unique values
print(data_df['bank_account'].value_counts(), '\n')
print(data_df['location_type'].value_counts(), '\n')
print(data_df['cellphone_access'].value_counts(), '\n')
print(data_df['gender_of_respondent'].value_counts(), '\n')
print(data_df['relationship_with_head'].value_counts(), '\n')
print(data_df['marital_status'].value_counts(), '\n')
print(data_df['education_level'].value_counts(), '\n')
print(data_df['country'].value_counts(), '\n')

## Binary encoding for 'bank_account' and 'cellphone_access'
data_df['bank_account'] = data_df['bank_account'].map({'No': 0, 'Yes': 1})
data_df['cellphone_access'] = data_df['cellphone_access'].map({'No': 0, 'Yes': 1})

# Drop the ['uniqueid','year'] columns
data_df = data_df.drop(columns=['uniqueid','year'])

## One-hot encoding for other categorical variables
df_encoded = pd.get_dummies(data_df, columns=['country', 'location_type', 'gender_of_respondent', 'relationship_with_head',
                                              'marital_status', 'education_level', 'job_type'])


# Display the encoded dataframe
print(df_encoded.head(), '\n')

### Machine Learning MODEL

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define features and target
X = df_encoded.drop('bank_account', axis=1)
y = df_encoded['bank_account']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('\n', f"Accuracy: {accuracy_score(y_test, y_pred)}",'\n')
print(classification_report(y_test, y_pred))

## Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

## Save the encoded dataframe
with open('df_encoded.pkl', 'wb') as file:
    pickle.dump(df_encoded, file)
