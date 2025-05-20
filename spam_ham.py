import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib  # For saving models

# Load the dataset
file_path = r"C:\Users\pavith\Downloads\spam.csv"
data = pd.read_csv(file_path, encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']

# Map labels to binary values
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the model
y_pred = model.predict(X_test_tfidf)

# Classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Additional Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Save the vectorizer and model
joblib.dump(vectorizer, "spam_vectorizer.pkl")
joblib.dump(model, "spam_classifier.pkl")
print("Model and vectorizer saved successfully!")

# Function to classify messages
def predict_message(message):
    message_tfidf = vectorizer.transform([message])    
    prediction = model.predict(message_tfidf)[0]
    return "Spam" if prediction == 1 else "Ham"

# Interactive classification
while True:
    user_input = input("\nEnter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    
    result = predict_message(user_input)
    print(f"The message is classified as: {result}")

