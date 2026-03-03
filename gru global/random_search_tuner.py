import os
import json
import random
import numpy as np
import tensorflow as tf

from train_global_gru import build_global_gru_model  # reutiliza o builder
from paths import (
    artifacts_dir,
    dataset_npz_path,
    dataset_meta_path,
    best_hps_path,
    tuned_model_path,
    ensure_dir,
    remove_if_exists,
)


def sample_from(options):
    return random.choice(options)


def sample_log_uniform(low, high):
    # amostra log-uniforme
    u = random.random()
    return low * (high / low) ** u


def main():
    ensure_dir(artifacts_dir())

    npz_path = dataset_npz_path()
    meta_path = dataset_meta_path()
    if not os.path.exists(npz_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Execute primeiro o prepare_dataset_mcp.py")

    data = np.load(npz_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    X_train, y_train = data["X_train"], data["y_train"]
    emp_train = data["emp_train"]

    # Random search config
    n_trials = 20
    max_epochs = 20  # tuning rápido
    val_split = 0.1
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Espaços de busca
    space = {
        # model_hps
        "emb_dim": [4, 8, 16, 32],
        "gru_units": [32, 64, 128, 256],
        "dense_units": [16, 32, 64, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        # train_hps
        "batch_size": [64, 128, 256, 512],
        "lr_low": 1e-4,
        "lr_high": 3e-3,
    }

    best = {
        "val_loss": float("inf"),
        "model_hps": None,
        "train_hps": None,
    }

    for t in range(1, n_trials + 1):
        model_hps = {
            "emb_dim": sample_from(space["emb_dim"]),
            "gru_units": sample_from(space["gru_units"]),
            "dense_units": sample_from(space["dense_units"]),
            "dropout": sample_from(space["dropout"]),
            "lr": float(sample_log_uniform(space["lr_low"], space["lr_high"])),
        }
        train_hps = {
            "batch_size": sample_from(space["batch_size"]),
            "epochs": max_epochs,
            "val_split": val_split,
            "patience_es": 4,
            "patience_rlr": 2,
        }

        model = build_global_gru_model(
            timesteps=X_train.shape[1],
            n_features=X_train.shape[2],
            n_empresas=int(meta["n_empresas"]),
            **model_hps,
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=int(train_hps["patience_es"]), restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", patience=int(train_hps["patience_rlr"])
            ),
        ]

        hist = model.fit(
            {"series": X_train, "empresa_id": emp_train},
            y_train,
            validation_split=float(train_hps["val_split"]),
            epochs=int(train_hps["epochs"]),
            batch_size=int(train_hps["batch_size"]),
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        val_loss = float(np.min(hist.history["val_loss"]))
        print(f"Trial {t:02d}/{n_trials} | val_loss={val_loss:.6f} | {model_hps} | bs={train_hps['batch_size']}")

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["model_hps"] = model_hps
            best["train_hps"] = train_hps

            # salva o modelo melhor até agora
            out_model = tuned_model_path()
            remove_if_exists(out_model)
            model.save(out_model)

    # salva hiperparâmetros
    payload = {"model_hps": best["model_hps"], "train_hps": best["train_hps"], "best_val_loss": best["val_loss"]}
    with open(best_hps_path(), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\nRandom Search finalizado!")
    print("best_val_loss:", best["val_loss"])
    print("best_hyperparameters.json:", best_hps_path())
    print("best model (.keras):", tuned_model_path())


if __name__ == "__main__":
    main()