import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from dataset import AudioFeatureDataset

# ======= 配置路径 =======
ROOT_DIR = '/data/chenmingzhi/dataset/Audio/WAV'
FEATURE_ROOT = '/data/chenmingzhi/dataset/Audio/AudioFeature/10s'
INFO_PATH = '/data/chenmingzhi/dataset/Audio/subject_info_with_viq.npz'
EVENT_CLASSES = [0, 1, 2, 3]
AGE_THRESHOLD = None
AGE_OPTION = None
GENDER = None
DURATION = 10
OUT_DIR = '/data/chenmingzhi/dataset/Audio/segment/Result/All'
os.makedirs(OUT_DIR, exist_ok=True)
EXCEL_PATH = os.path.join(OUT_DIR, 'ml_metrics_5fold_all.xlsx')

def get_models():
    return {
        "SVM": SVC(probability=True, kernel='rbf', C=1.0, gamma='scale'),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    }

# ======= 5 fold =======
all_results = {}
for model_name, model in get_models().items():
    print(f"\n============ running: {model_name} ============")
    metrics_rows = []
    for fold_n in range(5):
        # === data loading ===
        dataset = AudioFeatureDataset(
            root_dir=ROOT_DIR, feature_root=FEATURE_ROOT, info_path=INFO_PATH,
            event_classes=EVENT_CLASSES, age_threshold=AGE_THRESHOLD,
            age_option=AGE_OPTION, gender=GENDER, duration=DURATION,
            fold_index=fold_n, mode='train'
        )
        test_dataset = AudioFeatureDataset(
            root_dir=ROOT_DIR, feature_root=FEATURE_ROOT, info_path=INFO_PATH,
            event_classes=EVENT_CLASSES, age_threshold=AGE_THRESHOLD,
            age_option=AGE_OPTION, gender=GENDER, duration=DURATION,
            fold_index=fold_n, mode='test'
        )
        X_train, y_train = np.array(dataset.X), np.array(dataset.y)
        X_test, y_test = np.array(test_dataset.X), np.array(test_dataset.y)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros_like(y_pred)

        # === evaluate ===
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        try:
            auroc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auroc = np.nan  # binary only

        print(f"Fold {fold_n} - Acc: {acc:.4f}, BAcc: {bacc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
        metrics_rows.append({
            "fold": fold_n,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1,
            "auroc": auroc
        })

    # save result
    all_results[model_name] = pd.DataFrame(metrics_rows)

# ======= output Excel 文件 =======
with pd.ExcelWriter(EXCEL_PATH) as writer:
    for model_name, df in all_results.items():
        df.to_excel(writer, sheet_name=f'{model_name}_5fold', index=False)

