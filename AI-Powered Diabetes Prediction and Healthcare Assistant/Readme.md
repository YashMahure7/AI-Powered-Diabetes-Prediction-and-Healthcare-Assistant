# 🏥 AI-Powered Diabetes Prediction and Healthcare Assistant

## 📌 Overview

AI-Powered Diabetes Prediction and Healthcare Assistant is an end-to-end Machine Learning and Generative AI application that predicts diabetes risk and provides intelligent medical assistance through Retrieval-Augmented Generation (RAG).

The system combines traditional Machine Learning for diabetes prediction with a local Large Language Model (Llama 3 via Ollama) and a FAISS-powered knowledge retrieval system to deliver personalized explanations and healthcare guidance.

---

## 🚀 Features

### 🔮 Diabetes Prediction

* Predicts diabetes risk using patient medical information.
* Uses a trained Random Forest classifier.
* Displays prediction confidence score.

### 🧠 AI-Powered Explanation

* Generates personalized explanations for prediction results.
* Uses Llama 3 running locally through Ollama.
* Provides risk-factor analysis and lifestyle suggestions.

### 💬 Medical Chatbot

* Answers diabetes-related questions.
* Uses Retrieval-Augmented Generation (RAG).
* Retrieves relevant medical knowledge before generating responses.

### 📚 Medical Knowledge Base

* Diabetes Basics
* Symptoms
* Causes
* Prevention
* Treatment
* Diet Guidelines
* Exercise Recommendations

---

## 🏗️ Project Architecture

User Input

↓

Streamlit UI

↓

Machine Learning Model (Random Forest)

↓

Prediction Result

↓

Llama 3 Explanation

↓

RAG Retrieval (FAISS)

↓

Medical Knowledge Base

↓

AI Response

---

## 🛠️ Tech Stack

### Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

### Web Application

* Streamlit

### Generative AI

* Ollama
* Llama 3

### RAG Components

* LangChain
* FAISS
* Sentence Transformers

### Model Persistence

* Pickle

---

## 📂 Project Structure

```text
diabetes_prediction_project/
│
├── streamlit_app.py
├── build_vector_store.py
├── requirements.txt
├── README.md
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── diabetes_model.pkl
│   └── scaler.pkl
│
├── medical_docs/
│   ├── diabetes_basics.txt
│   ├── symptoms.txt
│   ├── causes.txt
│   ├── treatment.txt
│   ├── prevention.txt
│   ├── diet_guide.txt
│   └── exercise_guide.txt
│
├── vector_store/
│   ├── index.faiss
│   └── index.pkl
│
└── src/
    ├── preprocessing.py
    ├── train_model.py
    ├── evaluate.py
    ├── save_model.py
    ├── embeddings.py
    ├── rag_pipeline.py
    ├── chatbot.py
    └── llm_explainer.py
```

---

## 📊 Machine Learning Models Compared

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)

Random Forest achieved the best overall performance and was selected as the final model.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/YashMahure7/AI-Powered Diabetes Prediction and Healthcare Assistant.git
cd AI-Powered Diabetes Prediction and Healthcare Assistant
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install Ollama

Download and install Ollama:

https://ollama.com

### Download Llama 3

```bash
ollama pull llama3
```

### Build Vector Store

```bash
python build_vector_store.py
```

### Run Application

```bash
streamlit run streamlit_app.py
```

---

## 🎯 Sample Questions for Chatbot

* What are the symptoms of diabetes?
* How can diabetes be prevented?
* What foods should diabetic patients avoid?
* What exercises are recommended for diabetes?
* What are the risk factors for Type 2 diabetes?

---

## 👨‍💻 Author

Yash Mahure

GitHub: https://github.com/YashMahure7
