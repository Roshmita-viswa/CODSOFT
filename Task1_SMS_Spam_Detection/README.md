# Spam SMS Detection AI

A beautiful Streamlit-based Machine Learning project that classifies SMS messages as Spam or Ham.

## Technologies Used
- Python
- pandas
- scikit-learn
- Streamlit
- joblib

## Setup

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset
Place `spam.csv` (downloaded from Kaggle) in this folder.

## Project Structure
- `train_model.py` - trains the model and saves artifacts
- `app.py` - interactive pastel cyber UI
- `spam_model.pkl` - trained model
- `vectorizer.pkl` - TF-IDF vectorizer
