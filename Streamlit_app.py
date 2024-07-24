import streamlit as st
import pandas as pd
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the original dataframe for unique values
data_df = pd.read_csv(r"C:\Users\ahmed\PycharmProjects\Africa_Financial_Inclusion\Financial_inclusion_dataset.csv")

# Load the encoded dataframe
with open('df_encoded.pkl', 'rb') as file:
    df_encoded = pickle.load(file)

# Define features and target
X = df_encoded.drop('bank_account', axis=1)

# Define the Streamlit app
st.title("Africa Financial Inclusion Prediction")

# Add the image
image_path = r"C:\Users\ahmed\PycharmProjects\Africa_Financial_Inclusion\UNUZ4zR - Imgur.jpg"
st.image(image_path, caption="Financial Inclusion")

# Input fields
country = st.selectbox("Country", data_df['country'].unique())
year = st.selectbox("Year", data_df['year'].unique())
location_type = st.selectbox("Location Type", data_df['location_type'].unique())
cellphone_access = st.selectbox("Cellphone Access", data_df['cellphone_access'].unique())
household_size = st.number_input("Household Size", min_value=1, max_value=20, value=1)
age_of_respondent = st.number_input("Age of Respondent", min_value=18, max_value=100, value=18)
gender_of_respondent = st.selectbox("Gender of Respondent", data_df['gender_of_respondent'].unique())
relationship_with_head = st.selectbox("Relationship with Head", data_df['relationship_with_head'].unique())
marital_status = st.selectbox("Marital Status", data_df['marital_status'].unique())
education_level = st.selectbox("Education Level", data_df['education_level'].unique())
job_type = st.selectbox("Job Type", data_df['job_type'].unique())

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'country': [country],
        'year': [year],
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
