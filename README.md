# 🧠 Emotion Detection — NLP Project

## ✅ Works on VS Code (local) AND Streamlit Cloud

**No pre-saved model files needed.** Models train automatically on first launch.

---

## 📁 Project Structure (simple — no models folder!)

```
emotion_detection/
├── app.py              ← Streamlit app (trains models on first run)
├── train_and_save.ipynb← Full notebook for submission/reference
├── requirements.txt
└── README.md
```

---

## 🚀 Run Locally (VS Code)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch app
streamlit run app.py
```

First launch trains the models (~2–3 min), then caches them. Instant on reload.

---

## ☁️ Deploy on Streamlit Cloud

1. Push all files to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Click **Deploy** ✅

Models train automatically in the cloud on first visit.

---

## 📊 Model Results

| Model | Type | Accuracy |
|---|---|---|
| **Bidirectional LSTM** | Deep Learning | **~89.7%** |
| CatBoost | ML | ~86.75% |
| Logistic Regression | ML | ~84.70% |
| XGBoost | ML | ~84.43% |

---

## 📓 About the Notebook

`train_and_save.ipynb` covers the full assignment:
- EDA & Visualizations
- Text Preprocessing (stopwords, stemming, lemmatization)
- Feature Engineering (BoW, TF-IDF, Word2Vec)
- ML Models + GridSearchCV Hyperparameter Tuning
- DL Models (RNN, LSTM, GRU, BiLSTM, Stacked LSTM)
- Final comparison table
