
Sports vs Politics News Classification
Dataset: News Category Dataset (HuggingFace)

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import load_dataset
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset Loading And preparation

print("Loading dataset...")
dataset = load_dataset("heegyu/news-category-dataset")

data = pd.DataFrame(dataset["train"])
print("Total samples in dataset:", data.shape[0])

# Only sports and Politics category are sepaerated
data = data[data["category"].isin(["POLITICS", "SPORTS"])].copy()

# labels are mapping to integer
data["label"] = data["category"].map({
    "SPORTS": 0,
    "POLITICS": 1
})

print("Filtered dataset size:", data.shape[0])
print("\nCategory distribution:\n", data["category"].value_counts())

#Data Cleaning

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

data["clean_text"] = data["headline"].apply(preprocess_text)

# Spliting dateset into Training and Testing data

X_train, X_test, y_train, y_test = train_test_split(
    data["clean_text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Feature Extraction done using TF-IDF


vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models chosen to comapre 

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

#  Training & Evaluation

results = {}

print("\nModel Evaluation:\n")

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train_vec, y_train)
    predictions = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"{name} Test Accuracy: {round(accuracy,4)}")
    print(classification_report(y_test, predictions))

    # Creating Confusion Matrix for all ML models
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Sports", "Politics"],
                yticklabels=["Sports", "Politics"])
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    print("-"*60)

#  5 fold Cross Validation results

print("\n5-Fold Cross Validation Results:\n")

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5)
    print(f"{name} CV Accuracy: {round(cv_scores.mean(),4)}")

# Bar Graph To Compare Accuracy


plt.figure(figsize=(7,5))
plt.bar(results.keys(), results.values())
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.ylim(0.8, 1.0)
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")
