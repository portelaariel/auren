import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

from paths import (
    artifacts_dir,
    dataset_npz_path,
    dataset_meta_path,
    best_hps_path,
    official_model_path,
    ensure_dir,
    remove_if_exists,
)


def build_global_gru_model(
    timesteps: int,
    n_features: int,
    n_empresas: int,
    emb_dim: int,
    gru_units: int,
    dense_units: int,
    dropout: float,
    lr: float,
) -> tf.keras.Model:
    inp_x = layers.Input(shape=(timesteps, n_features), name="series")
    inp_emp = layers.Input(shape=(), dtype="int32", name="empresa_id")

    emb = layers.Embedding(n_empresas, emb_dim)(inp_emp)
    emb = layers.RepeatVector(timesteps)(emb)

    x = layers.Concatenate()([inp_x, emb])
    x = layers.GRU(gru_units)(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    x = layers.Dense(dense_units, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = Model(inputs={"series": inp_x, "empresa_id": inp_emp}, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae_norm")],
    )
    return model


def denormalize(y_norm: np.ndarray, ymeta: np.ndarray) -> np.ndarray:
    y_norm = np.asarray(y_norm).reshape(-1)
    return y_norm * ymeta[:, 1] + ymeta[:, 0]


def load_hps() -> tuple[dict, dict]:
    """
    Retorna (model_hps, train_hps).
    Se existir best_hyperparameters.json (do tuner), usa ele.
    """
    model_defaults = {
        "emb_dim": 8,
        "gru_units": 64,
        "dense_units": 32,
        "dropout": 0.0,
        "lr": 1e-3,
    }
    train_defaults = {
        "batch_size": 256,
        "epochs": 50,
        "val_split": 0.1,
        "patience_es": 5,
        "patience_rlr": 3,
    }

    hp_path = best_hps_path()
    if not os.path.exists(hp_path):
        return model_defaults, train_defaults

    with open(hp_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Suporta dois formatos:
    # A) {"model_hps": {...}, "train_hps": {...}}
    # B) formato antigo flat {"emb_dim":..., "batch_size":...}
    if "model_hps" in raw or "train_hps" in raw:
        model_hps = {**model_defaults, **raw.get("model_hps", {})}
        train_hps = {**train_defaults, **raw.get("train_hps", {})}
        return model_hps, train_hps

    # formato antigo
    model_hps = model_defaults.copy()
    train_hps = train_defaults.copy()
    for k in model_hps:
        if k in raw:
            model_hps[k] = raw[k]
    for k in train_hps:
        if k in raw:
            train_hps[k] = raw[k]
    return model_hps, train_hps


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
    X_test, y_test = data["X_test"], data["y_test"]
    emp_test = data["emp_test"]
    ymeta_test = data["ymeta_test"]

    model_hps, train_hps = load_hps()

    model = build_global_gru_model(
        timesteps=X_train.shape[1],
        n_features=X_train.shape[2],
        n_empresas=int(meta["n_empresas"]),
        **model_hps,
    )

    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(
    #         monitor="val_loss", patience=int(train_hps["patience_es"]), restore_best_weights=True
    #     ),
    #     tf.keras.callbacks.ReduceLROnPlateau(
    #         monitor="val_loss", patience=int(train_hps["patience_rlr"])
    #     ),
    # ]

    model.fit(
        {"series": X_train, "empresa_id": emp_train},
        y_train,
        validation_split=float(train_hps["val_split"]),
        epochs=int(train_hps["epochs"]),
        batch_size=int(train_hps["batch_size"]),
        shuffle=True,
        #callbacks=callbacks,
        verbose=1,
    )

    results = model.evaluate(
        {"series": X_test, "empresa_id": emp_test},
        y_test,
        verbose=0,
        return_dict=True,
    )
    print("Teste (normalizado):", results)

    y_pred_norm = model.predict({"series": X_test, "empresa_id": emp_test}, verbose=0).reshape(-1)
    y_pred = denormalize(y_pred_norm, ymeta_test)
    y_true = denormalize(y_test, ymeta_test)

    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    print(f"Teste (escala real): MAE={mae:.4f} | RMSE={rmse:.4f}")

    out_model = official_model_path()
    remove_if_exists(out_model)
    model.save(out_model)
    print(f"Modelo salvo em: {out_model}")

    # salva um report simples
    report = {
        "model_hps": model_hps,
        "train_hps": train_hps,
        "test_norm": results,
        "mae_original": mae,
        "rmse_original": rmse,
        "model_path": out_model,
        "npz_path": npz_path,
        "meta_path": meta_path,
    }
    with open(os.path.join(artifacts_dir(), "train_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("Report salvo em:", os.path.join(artifacts_dir(), "train_report.json"))


if __name__ == "__main__":
    main()