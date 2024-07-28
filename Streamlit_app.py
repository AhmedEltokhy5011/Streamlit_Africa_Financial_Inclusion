# Streamlit app with data preprocessing and model training

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data_df = pd.read_csv(r"https://raw.githubusercontent.com/AhmedEltokhy5011/Streamlit_Africa_Financial_Inclusion/main/Financial_inclusion_dataset.csv")

# Adjust display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 400)

# Handle Outliers
sns.boxplot(data=data_df[['age_of_respondent', 'household_size']])
plt.title('Boxplot of Age and Household Size')
plt.savefig('boxplot of Age and Household Size.png')
plt.close()

# Encoding
data_df['bank_account'] = data_df['bank_account'].map({'No': 0, 'Yes': 1})
data_df['cellphone_access'] = data_df['cellphone_access'].map({'No': 0, 'Yes': 1})

data_df = data_df.drop(columns=['uniqueid', 'year'])
df_encoded = pd.get_dummies(data_df, columns=['country', 'location_type', 'gender_of_respondent', 'relationship_with_head',
                                              'marital_status', 'education_level', 'job_type'])

# Machine Learning Model
X = df_encoded.drop('bank_account', axis=1)
y = df_encoded['bank_account']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate the model
print('\n', f"Accuracy: {accuracy_score(y_test, y_pred)}", '\n')
print(classification_report(y_test, y_pred))

# Streamlit app
st.title("Africa Financial Inclusion Prediction")

# Add the image
image_path = r"C:\Users\ahmed\PycharmProjects\Africa_Financial_Inclusion\UNUZ4zR - Imgur.jpg"
st.image(image_path, caption="Financial Inclusion")

# Input fields
country = st.selectbox("Country", df_encoded['country'].unique())
location_type = st.selectbox("Location Type", df_encoded['location_type'].unique())
cellphone_access = st.selectbox("Cellphone Access", df_encoded['cellphone_access'].unique())
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=18, max_value=100, value=18)
gender_of_respondent = st.selectbox("Gender of Respondent", df_encoded['gender_of_respondent'].unique())
relationship_with_head = st.selectbox("Relationship with Head", df_encoded['relationship_with_head'].unique())
marital_status = st.selectbox("Marital Status", df_encoded['marital_status'].unique())
education_level = st.selectbox("Education Level", df_encoded['education_level'].unique())
job_type = st.selectbox("Job Type", df_encoded['job_type'].unique())

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'country': [country],
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    })

    # Apply the same encoding as during training
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=X.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data_encoded)
    st.write(f"The predicted class is: {'Yes' if prediction[0] == 1 else 'No'}")
