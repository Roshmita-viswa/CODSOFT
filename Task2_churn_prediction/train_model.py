import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("Loading dataset...")

# ---------------- LOAD DATASET ----------------
df = pd.read_csv("customer_churn.csv")

# Remove unnecessary columns
df.drop(
    columns=["RowNumber", "CustomerId", "Surname"],
    inplace=True,
    errors="ignore"
)

# ---------------- ENCODE CATEGORICAL DATA ----------------
encoders = {}

for column in df.select_dtypes(include="object").columns:

    encoder = LabelEncoder()

    df[column] = encoder.fit_transform(df[column])

    encoders[column] = encoder

# ---------------- FEATURES & TARGET ----------------
X = df.drop("Exited", axis=1)

y = df["Exited"]

# ---------------- SAVE FEATURE COLUMNS ----------------
feature_columns = X.columns.tolist()

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- SCALING ----------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    min_samples_split=5,

    min_samples_leaf=2,

    class_weight="balanced",

    random_state=42
)

# ---------------- TRAIN ----------------
model.fit(X_train_scaled, y_train)

# ---------------- PREDICT ----------------
predictions = model.predict(X_test_scaled)

# ---------------- RESULTS ----------------
accuracy = accuracy_score(y_test, predictions)

print(f"\nModel Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

# ---------------- SAVE FILES ----------------
joblib.dump(model, "churn_model.pkl")

joblib.dump(scaler, "scaler.pkl")

joblib.dump(feature_columns, "feature_columns.pkl")

print("\nModel saved successfully.")