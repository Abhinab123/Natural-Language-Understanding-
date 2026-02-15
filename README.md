# 📰 Sport vs Politics News Classification

### CSL7640 – Natural Language Understanding

**Indian Institute of Technology, Jodhpur**

---

## 📌 Project Overview

This assignment implements a binary text classification system to distinguish between **Sports** and **Politics** news headlines using classical machine learning techniques.

The objective was not only to build a classifier but also to conduct a comparative analysis of multiple models and evaluate their performance using proper validation techniques.

This assignment was developed as part of the course **CSL7640 – Natural Language Understanding** at **IIT Jodhpur**.

---

## 📂 Dataset

**Source:** HuggingFace – `heegyu/news-category-dataset`

From the complete dataset, only the following categories were selected:

* `SPORTS`
* `POLITICS`

### Why This Dataset?

Several datasets were evaluated before final selection. The chosen dataset was preferred because:

* It contains real-world news headlines
* It presents vocabulary overlap between categories
* It avoids artificially perfect classification
* It provides a realistic evaluation setting

---

## ⚙️ Preprocessing Pipeline

The following preprocessing steps were applied:

* Lowercasing all text
* Removing punctuation
* Removing numeric characters
* Whitespace normalization

### Feature Representation

* TF-IDF (Term Frequency – Inverse Document Frequency)
* Unigrams and bigrams
* Maximum 10,000 features

---

## 🤖 Models Implemented

Four classical machine learning algorithms were implemented and compared:

1. **Naive Bayes**
2. **Logistic Regression**
3. **Linear Support Vector Machine (SVM)**
4. **Random Forest**

---

## 📊 Evaluation Strategy

* 80–20 Stratified Train-Test Split
* 5-Fold Cross Validation
* Accuracy as primary metric
* Confusion Matrix Analysis
* Classification Reports (Precision, Recall, F1-score)

---

## 📈 5-Fold Cross Validation Results

| Model               | Mean CV Accuracy |
| ------------------- | ---------------- |
| Naive Bayes         | 0.9516           |
| Logistic Regression | 0.9465           |
| Linear SVM          | 0.9671           |
| Random Forest       | 0.9609           |

**Linear SVM achieved the highest overall performance.**

---

## 📉 Visualizations Included

* Accuracy comparison bar graph
* Confusion matrix heatmaps
* Classification reports for each model

These visual tools help interpret model behavior beyond numerical scores.

---

## 🔍 Key Observations

* Linear classifiers perform exceptionally well in high-dimensional sparse TF-IDF feature spaces.
* SVM provides the best margin separation and generalization.
* Random Forest performs strongly but slightly below SVM.
* Most misclassifications occur in headlines with overlapping themes (e.g., sports policy discussions).

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install datasets pandas scikit-learn matplotlib seaborn
```

### 2️⃣ Run the Script

```bash
python M25CSE001_prob4.py
```

---

## 📁 Project Structure

```
.
├── M25CSE001_prob4.py
├── README.md
├── accuracy_bar_graph.png
├── confusion_matrix_svm.png
└── report.pdf
```

---

## 🧠 Learning Outcomes

This project demonstrates:

* Practical implementation of text preprocessing
* Feature engineering using TF-IDF
* Model comparison and evaluation
* Cross-validation for robust assessment
* Interpretation of graphical results
* Understanding of classical NLP classification techniques

---

## 👤 Author

**Abhinab Bezbaruah**
Department of Computer Science and Engineering
Indian Institute of Technology, Jodhpur
Course: CSL7640 – Natural Language Understanding

---

