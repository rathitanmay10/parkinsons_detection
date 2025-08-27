# Parkinson’s Disease Detection (with Data Augmentation)

## 📌 Project Overview
- Predict Parkinson’s disease using **voice measurements**.  
- Dataset: 22 biomedical voice features (jitter, shimmer, HNR, RPDE, DFA, etc.).  
- Challenge: dataset is **small and imbalanced**.  
- Solution: use **cGAN (Conditional GAN)** to generate synthetic patient data → improve training.

---

## ⚙️ Workflow
1. **Data Preprocessing**  
   - Removed non-numeric cols (name, status).  
   - Scaled features with `StandardScaler`.

2. **Data Augmentation**  
   - Trained **cGAN** for 500 epochs.  
   - Generated **synthetic samples** to balance classes.

3. **Classifier Training**  
   - Used **Random Forest** (after augmentation).  
   - Achieved **~92% accuracy** on test set.

4. **Outputs**  
   - Trained models saved in `models/`.  
   - Confusion matrix → `outputs/confusion_matrix.png`.  
   - Synthetic samples → `outputs/sample_synthetic_rows.csv`.  
   - Predictions from CLI → `predict_cli.py`.

---

## 📊 Results
- **Accuracy:** ~92%  
- **Precision (Parkinson’s):** ~0.93  
- **Recall (Parkinson’s):** ~0.97  
- **F1-score (Parkinson’s):** ~0.95  
- ✅ High recall = very few false negatives (important in medical detection).

---

## 🎯 Viva Key Points
- Why GANs? → To generate realistic synthetic data instead of random oversampling.  
- Why Random Forest? → Robust, handles small datasets, interpretable.  
- Key Features: jitter, shimmer, RPDE, DFA, spread1/2 → strong indicators of Parkinson’s.  
- Limitation: dataset is small; clinical validation needed.  
- Future Work: try XGBoost, cross-validation, and visualization with PCA/t-SNE.

---

## 🚀 How to Run
python train_full.py                       # Train GAN + classifier
python predict_cli.py --input sample_fixed.csv   # Predict from CSV
