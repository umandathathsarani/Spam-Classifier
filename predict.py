import joblib

vectorizer = joblib.load('model/vectorizer.pkl')
classifier = joblib.load('model/classifier.pkl')

print("--- Spam Classifier ---")
print("Type 'exit' to quit.\n")

while True:
    user_input = input("Enter a message to check: ")
    
    if user_input.lower() == 'exit':
        break
        
    if not user_input.strip():
        continue

    input_dtm = vectorizer.transform([user_input])
    prediction = classifier.predict(input_dtm)
    
    if prediction[0] == 1:
        print("🚨 RESULT: SPAM\n")
    else:
        print("✅ RESULT: NOT SPAM\n")