import shap
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
from dataset import AudioFeatureDataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score
)

fold_n = 0
dataset = AudioFeatureDataset(
    root_dir='/data/dataset/Audio/segment/Audio_Children',
    feature_root='/data/dataset/Audio/segment/AudioFeature_Children/10s',
    info_path='/data/dataset/Audio/subject_info_with_viq.npz',
    event_classes=[0, 1, 2, 3],
    age_threshold=None,
    age_option=None,
    gender=None,
    duration=10,
    fold_index=fold_n,
    mode='train'
)

test_dataset = AudioFeatureDataset(
    root_dir='/data/dataset/Audio/segment/Audio_Children',
    feature_root='/data/dataset/Audio/segment/AudioFeature_Children/10s',
    info_path='/data/dataset/Audio/subject_info_with_viq.npz',
    event_classes=[0, 1, 2, 3],
    age_threshold=None,
    age_option=None,
    gender=None,
    duration=10,
    fold_index=fold_n,
    mode='test'
)


X_train = np.array(dataset.X)
y_train = np.array(dataset.y)

X_test = np.array(test_dataset.X)
y_test = np.array(test_dataset.y)



# === Step 2 ===
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# === Step 3 ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# === Step 4: SHAP ===
explainer = shap.Explainer(model)
shap_values = explainer(X_test)

# === Step 5 ===
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)


acc = accuracy_score(y_test, y_pred)


bacc = balanced_accuracy_score(y_test, y_pred)


f1 = f1_score(y_test, y_pred, average='macro')


y_proba = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_proba)

print(f"Accuracy: {acc:.4f}")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"F1 Score (macro): {f1:.4f}")
print(f"AUROC: {auroc:.4f}")