import tkinter as tk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = joblib.load("Models/KNN/tfidf.pkl")
model = joblib.load("Models/KNN/KNN_model.pkl")

root = tk.Tk()
root.geometry("400x200")
root.title("Arabic News Classifier")

def predict_category():
    article = article_entry.get("1.0", tk.END)
    article = vectorizer.transform([article])
    prediction = model.predict(article)[0]
    confidence = model.predict_proba(article).max() * 100
    result_label.config(text=f"Predicted category: {prediction}\nConfidence: {confidence:.2f}%")

article_entry = tk.Text(root, height = 10,width=50)
article_entry.pack(pady=10)
predict_button = tk.Button(root, text="Predict", command=predict_category)
predict_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

root.mainloop()