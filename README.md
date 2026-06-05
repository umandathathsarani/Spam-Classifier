# 📩 Spam Email Classifier

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.9.0-orange.svg)
![License](https://img.shields.io/badge/License-Proprietary-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

An end-to-end Machine Learning pipeline that leverages Natural Language Processing (NLP) to autonomously classify text messages and emails as either "Spam" or "Ham" (legitimate). 

**Developed by:** Mummullage Binuri Umanda Thathsarani  
**Academic Affiliation:** BSc (Hons) in Information Technology Specializing in Artificial Intelligence, SLIIT

---

## 📑 Table of Contents
1. [System Architecture](#-system-architecture)
2. [Project Structure](#-project-structure)
3. [Installation & Setup](#-installation--setup)
4. [Usage Guide](#-usage-guide)
5. [Model Performance](#-model-performance)
6. [Future Enhancements](#-future-enhancements)

---

## 🧠 System Architecture

This project is built on a standard NLP classification pipeline:

1. **Data Ingestion:** Automatically retrieves the open-source SMS Spam Collection dataset (5,500+ labeled messages).
2. **Text Preprocessing:** Utilizes `nltk` to normalize text. This includes converting to lowercase, stripping punctuation, and removing non-predictive English stop words.
3. **Feature Extraction:** Implements `CountVectorizer` to transform the raw text into a Bag-of-Words matrix, assigning mathematical weights to vocabulary frequency.
4. **Probabilistic Modeling:** Trains a **Multinomial Naive Bayes** algorithm. This specific algorithm is mathematically optimized for discrete features like word counts.

---

## 📁 Project Structure

```text
spam-classifier/
├── data/                   # Raw dataset storage (git-ignored)
├── model/                  # Serialized .pkl model files (git-ignored)
├── train.py                # Pipeline script: downloads data, trains, and serializes the model
├── predict.py              # CLI tool: loads the model for real-time user input testing
├── requirements.txt        # Environment dependencies
├── .gitignore              # Version control configuration
└── README.md               # Project documentation
```

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/umandathathsarani/Spam-Classifier.git](https://github.com/umandathathsarani/Spam-Classifier.git)
cd Spam-Classifier
```

**2. Isolate the environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

---

## 💻 Usage Guide

**Step 1: Train the Engine** Execute the training script to initialize the dataset, train the algorithm, and save the model states.
```bash
python train.py
```

**Step 2: Interactive Inference** Launch the interactive prediction CLI to test the model against custom inputs.
```bash
python predict.py
```

---

## 📊 Model Performance
* **Algorithm:** Multinomial Naive Bayes
* **Accuracy:** ~98.8%
* **Precision (Spam):** 97%
* **Recall (Spam):** 95%

---

## 🚀 Future Enhancements
- [ ] Implement TF-IDF Vectorization to penalize overly frequent words.
- [ ] Build a local web interface using Flask or Streamlit.
- [ ] Expand the dataset to include modern email spam templates.
