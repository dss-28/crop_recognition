# 🌾 Crop Recognition Using Machine Learning

Identify crops from image data using classical machine learning, engineered image features, and ensemble modeling.  
Achieved **98% validation accuracy** with a **70× speedup** over CNNs using flattened and interpretable features.

---

## 📌 Overview

This project classifies **five crop types** — Wheat, Rice, Maize, Sugarcane, and Jowar — from RGB images using traditional ML approaches instead of CNNs.

### 🧠 Why?
To build a **faster, interpretable, and efficient** solution that performs well even with limited data.

---

## 🧠 Key Highlights

- ✅ Used **5 hand-crafted image features**: Color Histogram, Haralick, LBP, HOG, Fourier
- ✅ Feature fusion with **PCA** (from 150,000 → 512 dims)
- ✅ Tried **6+ ML models** with hyperparameter tuning
- ✅ Built **two ensemble layers** for final prediction
- ✅ Achieved **98% validation accuracy**
- ✅ Reduced training time by **70×** compared to CNNs

---

## 📂 Dataset

- 📸 1000 RGB crop images  
- 🔄 Balanced across 5 classes: Wheat, Rice, Maize, Sugarcane, Jowar  
- 📐 Resized to 224×224  
- 📁 Public dataset sourced from Kaggle  

---

## 🔬 Feature Extraction

| Feature Type      | Description                                |
|-------------------|--------------------------------------------|
| Color Histogram   | Captures dominant RGB distributions        |
| Haralick          | Texture info via co-occurrence matrix      |
| LBP               | Captures local grayscale structure         |
| HOG               | Edge and shape features                    |
| Fourier Transform | Frequency domain texture representation    |

> Final feature vector size: **512** (after PCA)

---

## 🤖 Models Used

### Base Models:
- SVM  
- Decision Tree  
- Logistic Regression  
- Naïve Bayes  
- KNN

### Ensemble Models:
- ✅ **Voting Classifier 1**: Base models  
- ✅ **Voting Classifier 2**:  
   - Bagging (Tree, SVM, RF)  
   - Boosting (AdaBoost, XGBoost, Gradient Boost)

### Best Model:
- **Voting Classifier 2**  
- 📈 Accuracy: 98% (val/test)  
- 🕒 Training time: 70× faster using PCA + feature optimization

---

## 🧪 Evaluation Metrics

- Accuracy: **98%**
- F1-score: **0.98**
- Confusion Matrix (in `Results/`)
- Validated on both test and validation splits

---

## 🚀 Future Work (Planned)

- Deploy as a web app using **Streamlit** or **FastAPI**
- Add **webcam/drone input** for real-time crop classification
- Integrate Docker for deployment

---

