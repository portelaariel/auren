import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from paths import (
    plots_dir,
    dataset_npz_path,
    dataset_meta_path,
    resolve_model_path,
    ensure_dir,
)


def denormalize(y_norm: np.ndarray, ymeta: np.ndarray) -> np.ndarray:
    y_norm = np.asarray(y_norm).reshape(-1)
    mean = ymeta[:, 0]
    std = ymeta[:, 1]
    return y_norm * std + mean


def safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)


def main():
    ensure_dir(plots_dir())

    model = tf.keras.models.load_model(resolve_model_path())
    data = np.load(dataset_npz_path())

    with open(dataset_meta_path(), "r", encoding="utf-8") as f:
        meta = json.load(f)

    X_test = data["X_test"]
    y_test = data["y_test"]
    emp_test = data["emp_test"]
    ymeta_test = data["ymeta_test"]

    # Predição (normalizada -> escala original)
    y_pred_norm = model.predict(
        {"series": X_test, "empresa_id": emp_test}, verbose=0
    ).reshape(-1)

    y_pred = denormalize(y_pred_norm, ymeta_test)
    y_true = denormalize(y_test, ymeta_test)

    # -------------------------
    # Plot 1) Série geral
    # -------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="Real")
    plt.plot(y_pred, label="Predito")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir(), "overall_test.png"), dpi=150)
    plt.close()

    # -------------------------
    # Plot 2) Scatter geral
    # -------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "--k")
    plt.xlabel("Real")
    plt.ylabel("Predito")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir(), "scatter_real_vs_pred.png"), dpi=150)
    plt.close()

    # -------------------------
    # Métricas por empresa
    # -------------------------
    rows = []
    company_map = meta.get("company_map", {})

    for emp_id in np.unique(emp_test):
        mask = emp_test == emp_id
        n = int(mask.sum())
        if n < 30:  # ignora empresas com poucos pontos
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        mae = float(np.mean(np.abs(y_p - y_t)))
        rmse = float(np.sqrt(np.mean((y_p - y_t) ** 2)))

        name = company_map.get(str(emp_id), f"empresa_{emp_id}")
        rows.append(
            {
                "empresa_id": int(emp_id),
                "empresa": name,
                "n_amostras": n,
                "MAE": mae,
                "RMSE": rmse,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    # Salva CSV com métricas
    metrics_csv_path = os.path.join(plots_dir(), "metricas_por_empresa.csv")
    metrics_df.to_csv(metrics_csv_path, index=False, encoding="utf-8")

    # -------------------------
    # Plot 3) MAE por empresa
    # -------------------------
    if len(metrics_df) > 0:
        plt.figure(figsize=(10, max(4, 0.35 * len(metrics_df))))
        plt.barh(metrics_df["empresa"], metrics_df["MAE"])
        plt.xlabel("MAE (Consumo diário)")
        plt.title("MAE por empresa (menor = melhor)")
        plt.grid(axis="x")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir(), "mae_por_empresa.png"), dpi=150)
        plt.close()

    # -------------------------
    # Plot 4) Séries por empresa
    # -------------------------
    for emp_id in np.unique(emp_test):
        mask = emp_test == emp_id
        if mask.sum() < 30:
            continue

        name = safe_name(company_map.get(str(emp_id), f"empresa_{emp_id}"))

        plt.figure(figsize=(12, 5))
        plt.plot(y_true[mask], label="Real")
        plt.plot(y_pred[mask], label="Predito")
        plt.title(name)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir(), f"empresa_{name}.png"), dpi=150)
        plt.close()

    print("Plots gerados em:", plots_dir())
    print("Métricas por empresa:", metrics_csv_path)


if __name__ == "__main__":
    main()