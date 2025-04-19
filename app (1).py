
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import openai

# Train simple ML model
X = np.array([[1], [2], [3.5], [4], [5], [5.5], [6], [6.5], [7], [8], [9]])
y = np.array([17, 25, 45, 50, 52, 55, 65, 70, 75, 85, 89])
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.set_page_config(page_title="Marks Predictor", page_icon="📊")
st.title("🎓 Student Marks Predictor")
st.write("Enter how many hours you studied and get your predicted score!")

hours = st.number_input("🕒 Hours Studied", min_value=0.0, max_value=24.0, value=1.0)

if st.button("Predict My Score"):
    pred = model.predict([[hours]])[0]
    st.success(f"📈 Predicted Score: **{round(pred, 2)} marks**")

    # ChatGPT API (optional)
    openai.api_key = st.secrets["OPENAI_API_KEY"]  # Secure API key
    prompt = f"I studied for {hours} hours. Give me 2 short tips to improve my marks."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You're a helpful study coach."},
                {"role": "user", "content": prompt}
            ]
        )
        advice = response['choices'][0]['message']['content']
        st.info("💡 AI Tips:\n" + advice)
    except Exception as e:
        st.error("⚠️ Error generating advice. Check your API key or try again.")
