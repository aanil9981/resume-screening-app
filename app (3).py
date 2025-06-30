
import streamlit as st
import joblib

# Load the trained model
model = joblib.load("resume_screening_model.joblib")

st.title("📄 Resume Screening Tool")
st.write("This app predicts whether a resume should be shortlisted based on its content.")

# Resume input
resume_text = st.text_area("Paste the resume text here:")

if st.button("Predict"):
    if resume_text.strip() == "":
        st.warning("Please enter some resume text.")
    else:
        prediction = model.predict([resume_text])[0]
        proba = model.predict_proba([resume_text])[0][prediction]
        result = "✅ Shortlisted" if prediction == 1 else "❌ Rejected"
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence Score: {proba:.2f}")
