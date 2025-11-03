# train_ai_transformer_health.py
# Honors Contract: AI Transformer Health Monitor
# Generates synthetic data, trains a model, and saves artifacts.

import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import joblib

np.random.seed(42)

# -----------------------------
# Synthetic signal utilities
# -----------------------------
def synth_waveform_V(t, V_rms=230.0, f=60.0, phase=0.0, noise_std=0.5):
    V_amp = V_rms * np.sqrt(2.0)
    v = V_amp * np.sin(2.0 * np.pi * f * t + phase)
    v += np.random.normal(0, noise_std, size=t.shape)
    return v

def synth_waveform_I(t, I_rms=5.0, f=60.0, phase=-np.pi/6, noise_std=0.3, harmonics: List[Tuple[int, float]] = None):
    I_amp = I_rms * np.sqrt(2.0)
    i = I_amp * np.sin(2.0 * np.pi * f * t + phase)
    if harmonics:
        for n, scale in harmonics:
            # small random phase jitter on harmonics
            i += (I_amp * scale) * np.sin(2.0 * np.pi * (n * f) * t + phase * np.random.uniform(0.8, 1.2))
    i += np.random.normal(0, noise_std, size=t.shape)
    return i

def calc_rms(x):
    return float(np.sqrt(np.mean(x**2)))

def thd(signal, f0, fs):
    # THD = sqrt(sum(harmonics^2)) / fundamental
    N = len(signal)
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    mags = np.abs(fft_vals) / N * 2.0  # rough magnitude scaling

    idx_f0 = int(np.argmin(np.abs(freqs - f0)))
    fund = mags[idx_f0] if 0 <= idx_f0 < len(mags) else 0.0

    harm_mask = (freqs > 1.5 * f0) & (freqs < (fs / 2.0 - 1e-9))
    harm = float(np.sqrt(np.sum(mags[harm_mask] ** 2)))
    return float(harm / fund) if fund > 1e-12 else 0.0

# -----------------------------
# Dataset generation
# -----------------------------
def generate_dataset(n_samples=1200, fs=6000, window_sec=0.25) -> pd.DataFrame:
    t = np.arange(0, window_sec, 1.0 / fs)
    R_eq = 0.5           # equivalent winding resistance, for copper losses
    core_loss_base = 50  # base core loss in W

    rows = []
    labels = ["normal", "overload", "overheat", "fault"]

    for _ in range(n_samples):
        label = np.random.choice(labels, p=[0.4, 0.25, 0.2, 0.15])

        # Base operating point
        V_rms = np.random.normal(230.0, 3.0)
        I_rms = abs(np.random.normal(5.0, 0.7))
        phase = np.random.uniform(-np.pi / 4, 0)  # lagging PF
        harmonics = []
        temp = np.random.normal(45.0, 3.0)
        temp_rate = np.random.normal(0.05, 0.01)
        core_loss = core_loss_base

        if label == "normal":
            harmonics = [(3, 0.01), (5, 0.005)]
            I_rms = np.clip(np.random.normal(5.0, 0.5), 3.5, 6.5)
            temp = np.clip(np.random.normal(45.0, 2.0), 35, 55)
            temp_rate = np.clip(np.random.normal(0.03, 0.01), 0.0, 0.06)
            core_loss = core_loss_base + np.random.normal(0, 5)

        elif label == "overload":
            harmonics = [(3, 0.02), (5, 0.01), (7, 0.005)]
            I_rms = np.clip(np.random.normal(8.0, 1.0), 6.0, 11.0)
            temp = np.clip(np.random.normal(65.0, 5.0), 50, 80)
            temp_rate = np.clip(np.random.normal(0.12, 0.02), 0.06, 0.18)
            core_loss = core_loss_base + np.random.normal(10, 8)

        elif label == "overheat":
            harmonics = [(3, 0.015), (5, 0.008)]
            I_rms = np.clip(np.random.normal(6.0, 0.7), 4.5, 8.0)
            temp = np.clip(np.random.normal(85.0, 6.0), 70, 100)
            temp_rate = np.clip(np.random.normal(0.20, 0.03), 0.12, 0.30)
            core_loss = core_loss_base + np.random.normal(15, 10)

        elif label == "fault":
            # emulate shorted turns or saturation
            harmonics = [(3, 0.06), (5, 0.04), (7, 0.02), (9, 0.01)]
            I_rms = np.clip(np.random.normal(12.0, 2.0), 8.0, 18.0)
            temp = np.clip(np.random.normal(95.0, 8.0), 75, 120)
            temp_rate = np.clip(np.random.normal(0.30, 0.05), 0.18, 0.45)
            core_loss = core_loss_base + np.random.normal(40, 15)

        # Synthesize waveforms
        v = synth_waveform_V(t, V_rms=V_rms, phase=np.random.uniform(-np.pi, np.pi))
        i = synth_waveform_I(t, I_rms=I_rms, phase=phase, harmonics=harmonics)

        # Measurements and features
        V_rms_m = calc_rms(v)
        I_rms_m = calc_rms(i)
        thd_i = thd(i, f0=60.0, fs=fs)
        copper_loss = (I_rms_m ** 2) * R_eq
        total_loss = copper_loss + core_loss

        rows.append({
            "V_rms": V_rms_m,
            "I_rms": I_rms_m,
            "THD_I": thd_i,
            "Temp_C": float(temp),
            "Temp_rate_C_per_s": float(temp_rate),
            "Loss_W": float(total_loss),
            "Condition": label
        })

    return pd.DataFrame(rows)

# -----------------------------
# Plots
# -----------------------------
def plot_confusion_matrix(cm: np.ndarray, labels: List[str], outpath: str):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

def plot_feature_importance(importances: np.ndarray, names: List[str], outpath: str):
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(importances)), importances)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    plt.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    # Generate dataset
    df = generate_dataset(n_samples=1200, fs=6000, window_sec=0.25)
    csv_path = "transformer_synthetic_dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path} with shape {df.shape}")

    # Train/test split
    X = df[["V_rms", "I_rms", "THD_I", "Temp_C", "Temp_rate_C_per_s", "Loss_W"]].values
    y = df["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train model
    clf = RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    labels = ["normal", "overload", "overheat", "fault"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, digits=3)
    print("Classification Report:\n", report)

    # Save artifacts
    model_path = "model_random_forest.pkl"
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")

    plot_confusion_matrix(cm, labels, "confusion_matrix.png")
    print("Saved confusion_matrix.png")

    feat_names = ["V_rms", "I_rms", "THD_I", "Temp_C", "Temp_rate_C_per_s", "Loss_W"]
    importances = clf.feature_importances_
    plot_feature_importance(importances, feat_names, "feature_importance.png")
    print("Saved feature_importance.png")

if __name__ == "__main__":
    main()
