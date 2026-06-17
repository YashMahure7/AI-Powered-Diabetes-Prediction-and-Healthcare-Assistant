# src/llm_explainer.py

import ollama

def explain_prediction(
    prediction,
    glucose,
    bmi,
    age,
    blood_pressure
):

    prompt = f"""
You are a healthcare assistant.

Prediction Result:
{prediction}

Patient Information:
Glucose: {glucose}
BMI: {bmi}
Age: {age}
Blood Pressure: {blood_pressure}

Explain:
1. What this prediction means
2. Possible risk factors
3. General lifestyle suggestions

Keep the explanation simple and easy to understand.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]