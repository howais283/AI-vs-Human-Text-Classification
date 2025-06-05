# AI-vs-Human-Text-Classification

**Binary text classification project to distinguish between AI-generated and human-written text using both machine learning and deep learning models.**  
Models such as GRU, LSTM, Logistic Regression, and Naive Bayes achieved classification accuracies exceeding **99%**, highlighting the potential of NLP in detecting AI-generated content.

---

## 🔍 Project Overview

This project addresses a growing concern in the age of generative AI: how to reliably distinguish between human-authored and AI-generated writing. Using a large, labeled dataset of 100,000 essays (balanced from an original dataset of over 480,000), we developed and compared a variety of classification models.

The project includes traditional models like Logistic Regression and Naive Bayes, as well as deep learning architectures like GRU and LSTM. Each model was carefully tuned using cross-validation and hyperparameter optimization to ensure generalization and high accuracy.

---

## 📈 Key Highlights

- **Balanced 100K dataset** curated from over 480K entries to handle class imbalance and memory limitations
- Achieved **99.25% accuracy** with Logistic Regression and **99.13%** with GRU after tuning
- Applied extensive **text preprocessing**, tokenization, vectorization (TF-IDF), and padding
- Integrated **model evaluation techniques** including confusion matrices, F1-scores, and k-fold cross-validation
- Highlights the potential for **ethical AI oversight**, academic integrity monitoring, and digital content verification

---

## 🎯 Use Case & Relevance

The project has real-world relevance in:
- **Education:** detecting AI-written essays to maintain academic integrity
- **Media:** verifying article authenticity to prevent misinformation
- **Cybersecurity:** identifying AI-written phishing or manipulation attempts
- **AI Ethics:** understanding and monitoring the spread of AI-generated content

---

## 📂 Contents

- `notebooks/`: Jupyter notebooks with preprocessing, model training, evaluation
- `dataset/`: Dataset reference with download instructions
- `reports/`: Project report with analysis, results, and methodology

---

## 📊 Dataset

The dataset used is publicly available on Kaggle:  
🔗 [AI vs Human Text Dataset – Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text)

- Original size: 487,235 essays
- Project subset: 100,000 entries (50K human-written, 50K AI-generated)

For more details, see [`dataset/dataset-info.md`](./Dataset/dataset-info.md)

---

## 🧠 Models & Techniques

- **Traditional ML Models:** Logistic Regression, Naive Bayes  
- **Deep Learning Models:** LSTM, GRU  
- **Preprocessing:** Stopword removal, lowercasing, symbol stripping, tokenization, TF-IDF  
- **Evaluation:** Accuracy, precision, recall, F1-score, confusion matrices, 5-fold cross-validation  
- **Optimization:** Hyperparameter tuning (regularization, dropout, learning rate, batch size)

---

## 🛠 Tools & Technologies

- Python (NumPy, Pandas, Scikit-learn, TensorFlow, Keras)
- Jupyter Notebooks
- Matplotlib, Seaborn
- Kaggle Datasets

---

## 🧪 Results Snapshot

| Model              | Accuracy | F1-Score |
|-------------------|----------|----------|
| Logistic Regression (Tuned) | 99.25%   | 0.99     |
| Naive Bayes (Tuned)         | 94.35%   | 0.94     |
| LSTM (Tuned)                | 99.04%   | 0.99     |
| GRU (Tuned)                 | 99.13%   | 0.99     |

---

## 📘 License

This project is for academic and educational purposes. Refer to the dataset’s license on [Kaggle](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) for usage terms.

