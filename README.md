# Sarcasm Detection — NLP, Deep Learning, Transformers & LoRA

An end-to-end **NLP sarcasm detection project** comparing Traditional Machine Learning, Deep Learning, Transformer-based models, and **LoRA/PEFT** for parameter-efficient fine-tuning.

🚀 **[Live Streamlit Demo](https://sarcasm-detection-with-traditional-nlp-deep-learning-and-trans.streamlit.app/)**  
📂 **[GitHub Repository](https://github.com/ARCHITTOMAR15/Sarcasm-Detection-using-Tradational-NLP-Deep-Learning-Transformer-Models-LoRA-PEFT-)**

---

## ⭐ Key Results

| Model | Approach | Accuracy | F1 Score | Trainable Parameters |
|---|---|---:|---:|---:|
| Tuned TF-IDF + Linear SVM | Traditional NLP | **79.39%** | — | — |
| Tuned Bi-LSTM | Deep Learning | **85.95%** | — | — |
| Frozen DistilBERT | Transformer | **81.69%** | **80.17%** | Classifier Head |
| **DistilBERT + LoRA** | **PEFT** | **85.93%** | **85.13%** | **739,586 (1.09%)** |
| ⭐ Pre-fine-tuned Model* | Transformer | **94.00%** | **93.52%** | **0** |

> *The 94% model is an externally pre-fine-tuned sarcasm classifier evaluated on the project test set; it was **not trained from scratch in this project**.

### Transformer Comparison

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Frozen DistilBERT + Classifier | 81.69% | 82.68% | 77.80% | 80.17% |
| ⭐ Pre-fine-tuned Model | **94.00%** | **96.18%** | **91.00%** | **93.52%** |
| DistilBERT + LoRA | 85.93% | 85.61% | 84.66% | 85.13% |

---

## 🧠 What This Project Covers

```text
                 Sarcasm Detection
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
 Traditional NLP   Deep Learning    Transformers
        │               │               │
   TF-IDF + ML      RNN/LSTM/GRU    DistilBERT
        │               │               │
        │          Bi-directional       │
        │             Models            │
        │               │               ↓
        │               │            LoRA / PEFT
        └───────────────┴───────────────┘
                        ↓
                 Model Comparison
                        ↓
                  Streamlit Demo
```

### Traditional NLP
- Text preprocessing
- TF-IDF
- Unigrams + Bigrams
- Logistic Regression
- Naive Bayes
- Linear SVM
- XGBoost
- Hyperparameter tuning

### Deep Learning
- Simple RNN
- LSTM
- GRU
- Bi-RNN
- Bi-LSTM
- Bi-GRU
- Tuned Bi-LSTM

### Transformers
- DistilBERT as a frozen feature extractor
- Pre-fine-tuned sarcasm classifier evaluation
- DistilBERT + **LoRA (PEFT)**

---

## 📊 Dataset

**Sarcasm Headlines Dataset v2**

- **28,619** raw samples
- **28,503** unique headlines
- Binary classification: `sarcastic` / `not sarcastic`
- Train: **22,802**
- Validation: **2,850**
- Test: **2,851**

Duplicate headlines were handled during preprocessing.

---

## 🔬 Best Project Results

### Traditional NLP
**Tuned TF-IDF + Linear SVM**

- Test Accuracy: **79.39%**
- Recall: **77.28%**

### Deep Learning
**Tuned Bi-LSTM**

- Test Accuracy: **85.95%**
- Recall: **83.33%**

### Transformer
**DistilBERT + LoRA**

- Test Accuracy: **85.93%**
- Precision: **85.61%**
- Recall: **84.66%**
- F1: **85.13%**
- Trainable Parameters: **739,586**
- Trainable Percentage: **1.09%**

This demonstrates **parameter-efficient adaptation of a pretrained Transformer** without updating the full model.

---

## 🛠️ Tech Stack

**Python** · **Pandas** · **NumPy** · **NLTK** · **Scikit-learn** · **XGBoost** · **TensorFlow/Keras** · **PyTorch** · **Hugging Face Transformers** · **PEFT/LoRA** · **Jupyter** · **Google Colab** · **Streamlit** · **Git/GitHub**

## 🎯 Project Goal

The goal is to compare **different generations of NLP techniques** and understand the trade-offs between:

**Performance · Model Complexity · Trainable Parameters · Fine-tuning Strategy**

The project progresses from classical NLP to modern **Transformer and parameter-efficient fine-tuning techniques**.

---

## 👨‍💻 Author

**Archit Tomar**

AI/ML · NLP · Deep Learning · Transformers · Generative AI

**[GitHub](https://github.com/ARCHITTOMAR15)**

---

⭐ If you find this project useful, consider giving the repository a star.
