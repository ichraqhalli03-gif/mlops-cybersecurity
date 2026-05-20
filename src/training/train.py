import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# =========================
# Chargement des données
# =========================

columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes",
    "land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files",
    "num_outbound_cmds","is_host_login","is_guest_login","count",
    "srv_count","serror_rate","srv_serror_rate","rerror_rate",
    "srv_rerror_rate","same_srv_rate","diff_srv_rate",
    "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate",
    "label","difficulty"
]

df = pd.read_csv("data/NSL-KDD-Dataset-master/NSL-KDD-Dataset-master/KDDTrain+.txt", names=columns)

# =========================
# Création target
# =========================

df["target"] = df["label"].apply(
    lambda x: 0 if x == "normal" else 1
)

# =========================
# Features / Labels
# =========================

X = df.drop(["label", "difficulty", "target"], axis=1)
y = df["target"]

# Encodage
X = pd.get_dummies(X)

# Sauvegarde des colonnes
joblib.dump(X.columns.tolist(), "models/features.pkl")

# =========================
# Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Modèles
# =========================

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=50,
        random_state=42
    ),

"SVM": SVC(
    kernel="linear",
    random_state=42,
    max_iter=1000
),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42
    )
}

# =========================
# Entraînement + MLflow
# =========================

mlflow.set_experiment("IDS_MLOps_Comparison")

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Logs MLflow
        mlflow.log_param("model", model_name)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Save model
        model_path = f"models/{model_name}.pkl"

        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(
            model,
            model_name
        )

        print(f"{model_name} terminé")
        print(f"Accuracy: {accuracy}")
        print("-" * 50)