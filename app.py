import streamlit as st
import pandas as pd
import joblib

good_models = joblib.load('good_college_models.pkl')

st.title("College Admission Predictor")





agree = st.checkbox(
    "I understand that this prediction tool is NOT 100% accurate and should be used as a supplementary guide only."
)

if not agree:
    st.warning("Please check the box to acknowledge the disclaimer before using the app.")
    st.stop() 


st.markdown("Enter your academic details below to predict your admission outcome.")

gpa = st.number_input("Unweighted GPA", min_value=0.0, max_value=4.0, step=0.1)
ap_classes = st.number_input("Total number of AP/Dual Enrollment classes", min_value=0)

math_level = st.selectbox("Highest Math Level", [
    "Personal Finance", "Trigonometry", "Algebra II", "Business Math", "College Algebra",
    "Math Analysis", "Pre-Calculus", "Math AI SL", "Statistics", "Data Science",
    "Math AI HL", "Calculus AB", "Math AA", "Calculus BC", "College Calculus II",
    "Multivariable Calculus", "College Calculus III", "Linear Algebra", "Differential Equations"
])

math_mapping = {
    "Trigonometry": 0.5,
    "Personal Finance": 0.4,
    "Algebra II": 1,
    "Business Math": 1.2,
    "College Algebra": 1.5,
    "Pre-Calculus": 2,
    "Math AI SL": 2.4,
    "Statistics": 2.5,
    "Data Science": 2.5,
    "Math AI HL": 2.7,
    "Calculus AB": 3,
    "Math AA": 3.4,
    "Calculus BC": 4,
    "College Calculus II": 4.35,
    "Multivariable Calculus": 4.5,
    "College Calculus III": 4.5,
    "Linear Algebra": 5,
    "Differential Equations": 5.5
}

math_level_encoded = math_mapping[math_level]

race_options = {
    "White": 0,
    "Non-White": 1
}

schooltype_options = {
    "Private": 0,
    "Public": 1
}

race_label = st.selectbox("Race", list(race_options.keys()))
race = race_options[race_label]

schooltype_label = st.selectbox("School Type", list(schooltype_options.keys()))
schooltype = schooltype_options[schooltype_label]

college_choice = st.selectbox("Select College", list(good_models.keys()))

if st.button("Predict Admission"):
    # Use the corrected variable names here
    input_df = pd.DataFrame([[
        gpa,
        ap_classes,
        math_level_encoded,
        race,
        schooltype
    ]], columns=[
        'Unweighted GPA', 
        'Total # of AP/Dual courses or IB Diploma',
        'Highest level math (difficulty over the yr taken)', 
        '   Race/Gender', 
        'HS TYPE'
    ])

    model = good_models[college_choice]

    proba = model.predict_proba(input_df)[0][1]  # Probability of acceptance
    

    
    prediction = model.predict(input_df)[0]
    if prediction == 0:
        prediction = 'Denied'
    elif prediction == 1:
        prediction = 'Accepted'
    st.success(f"Prediction for {college_choice}: {prediction}")
    st.write(f"Chance of acceptance: {proba:.1%}")


