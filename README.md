# 🌾 Crop Recognition Using Machine Learning

> Identify crops from image data using classical machine learning, engineered image features, and ensemble modeling.  
> Achieved **98% validation accuracy** with a 70× speedup over fata with just flattened features.

---

## 📌 Overview

This project classifies five crop types — **Wheat, Rice, Maize, Sugarcane, and Jowar** — from RGB images using traditional ML approaches instead of CNNs.

Why?  
To build a **faster, interpretable, and efficient solution** that performs well even with limited data.

---

## 🧠 Key Highlights

- ✅ Used **5 hand-crafted image features** (Color Histogram, Haralick, LBP, HOG, Fourier)
- ✅ Feature fusion with PCA (from 150k → 512 dims)
- ✅ Tried 6+ ML models with hyperparameter tuning
- ✅ Created two ensemble layers for final prediction
- ✅ Achieved **98% validation accuracy**
- ✅ Training time reduced by **70×** compared to CNNs

---

## 📂 Dataset

- ✅ 1000 RGB images of crops  
- 🔄 Balanced across all 5 classes  
- 📐 Resized to 224×224  
- 📁 Public dataset from Kaggle

---

## 📊 Feature Extraction

| Feature Type       | Description                          |
|--------------------|--------------------------------------|
| Color Histogram    | Captures dominant RGB distributions |
| Haralick           | Texture info via co-occurrence matrix |
| LBP (Local Binary Patterns) | Captures local grayscale structure |
| HOG (Histogram of Oriented Gradients) | Shape + edge features |
| Fourier Transforms | Frequency domain features |

Final feature vector size: **512**

---

## 🧠 Models Used

### Base Models:
- SVM, Decision Tree, Logistic Regression, Naïve Bayes, KNN

### Ensembles:
- ✅ **Voting Classifier 1**: Base models  
- ✅ **Voting Classifier 2**: Bagging (Tree, SVM, RF) + Boosting (Ada, XGBoost, GradBoost)

### Best Model:
- **Voting Classifier 2**  
- 📈 Accuracy: **98% (val/test)**  
- 🕒 Train time: Reduced by 70× using PCA and feature optimization

---

## 🧪 Evaluation Metrics

- Accuracy, Confusion Matrix, F1-score (0.98)
- Validated on test and validation splits

---


- Deploy as a web app (Streamlit or FastAPI)
- Real-time crop classification from webcam / drone input

---

## 🧑‍💻 Author

Darshan Shirsat  
M.Tech AI & DS, K. J. Somaiya College of Engineering  
[LinkedIn](https://linkedin.com/in/darshan-shirsat) | [GitHub](https://github.com/dss-28)

---

## 📣 Contributions Welcome

If you'd like to extend this with deep learning models or deployment options, feel free to fork and submit a PR!

