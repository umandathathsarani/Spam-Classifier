# 📩 Spam Email Classifier

An interactive Machine Learning pipeline that uses Natural Language Processing (NLP) to classify text messages and emails as either "Spam" or "Ham" (legitimate). 

Developed by Mummullage Binuri Umanda Thathsarani, Software Engineering Undergraduate at SLIIT.

## 🚀 Features

* **Automated Data Pipeline:** A training script that automatically fetches the open-source SMS Spam Collection dataset.
* **NLP Text Preprocessing:** Utilizes the Natural Language Toolkit (`nltk`) to clean text, remove punctuation, and filter out English stop words.
* **Vectorization:** Converts raw text into a mathematical matrix using Term Frequency-Inverse Document Frequency (via `CountVectorizer`).
* **Machine Learning Model:** Trains a Multinomial Naive Bayes algorithm using `scikit-learn`, achieving high-accuracy classification (approx. 98.8%).
* **Interactive CLI:** Includes a custom prediction script that allows users to type live messages into the terminal for real-time spam detection.

## 📁 Project Structure

```text
spam-classifier/
│
├── data/                   # Directory for the downloaded dataset (git-ignored)
├── model/                  # Directory for saved .pkl model files (git-ignored)
│
├── train.py                # Script to download data, train the model, and save it
├── predict.py              # Interactive script to test new messages
├── requirements.txt        # Project dependencies
├── .gitignore              # Version control configuration
└── README.md               # Project documentation
```

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/umandathathsarani/Spam-Classifier.git](https://github.com/umandathathsarani/Spam-Classifier.git)
cd Spam-Classifier
```

**2. Create a virtual environment**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download NLTK Stopwords**
```bash
python -c "import nltk; nltk.download('stopwords')"
```

## 💻 Usage

**Step 1: Train the Model**
Run the training script. This will automatically download the dataset to the `data/` folder, train the Naive Bayes algorithm, print the accuracy statistics, and save the compiled model files into the `model/` directory.
```bash
python train.py
```

**Step 2: Run the Interactive Predictor**
Once the model is trained, run the prediction script. You can type or paste any custom message into the terminal to see if the AI flags it as spam.
```bash
python predict.py
```

## ⚙️ Technologies Used
* **Python:** Core programming language.
* **Pandas:** Data manipulation and analysis.
* **Scikit-learn:** Machine learning framework.
* **NLTK:** Natural Language Toolkit for text preprocessing.
* **Joblib:** Model persistence (saving and loading trained models).

---
*This project was built to explore core concepts in Artificial Intelligence, dataset handling, and natural language processing.*