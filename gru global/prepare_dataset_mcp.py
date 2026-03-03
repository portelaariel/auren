import os
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class Config:
    # Entrada
    input_csv: str = "dataset_MCP_V1.csv"   # nome do arquivo
    out_dir: str = "artifacts"              # pasta de saída

    # Colunas do MCP
    company_col: str = "NomeRazaoSocialEmpresa"
    date_col: str = "DATA"

    raw_target_col: str = "QuantidadeConsumoAtivo"  # consumo realizado (diário)
    target_col: str = "CONSUMO_DIA"

    # Dataset/treino
    lookback: int = 90
    horizon: int = 1
    train_ratio: float = 0.8

    # Tratamento de datas
    enforce_daily_frequency: bool = True  # completa datas faltantes por empresa

    # CSV
    csv_sep: str = ","
    csv_decimal: Optional[str] = None
    encoding: Optional[str] = "utf-8"


def _script_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_script_dir(), path)


def cyclical_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df[date_col].dt
    dow = d.dayofweek.astype(int)   # 0..6
    doy = d.dayofyear.astype(int)   # 1..365/366

    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = d.month.astype(int)
    return df


def read_mcp_csv(cfg: Config) -> pd.DataFrame:
    in_path = _resolve_path(cfg.input_csv)

    kwargs = dict(sep=cfg.csv_sep, encoding=cfg.encoding)
    if cfg.csv_decimal is not None:
        kwargs["decimal"] = cfg.csv_decimal

    df = pd.read_csv(in_path, **kwargs)

    required = {cfg.company_col, cfg.date_col, cfg.raw_target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {missing}")

    # parse datas
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.company_col, cfg.date_col])

    # numérico: consumo
    df[cfg.raw_target_col] = pd.to_numeric(df[cfg.raw_target_col], errors="coerce")
    df = df.dropna(subset=[cfg.raw_target_col])

    df = df.sort_values([cfg.company_col, cfg.date_col]).reset_index(drop=True)
    return df


def build_company_id_map(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[int, str]]:
    names = sorted(df[cfg.company_col].astype(str).unique().tolist())
    name_to_id = {name: i for i, name in enumerate(names)}
    df["empresa_id"] = df[cfg.company_col].astype(str).map(name_to_id).astype(int)
    id_to_name = {i: name for name, i in name_to_id.items()}
    return df, id_to_name


def aggregate_daily(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Gera 1 linha por (empresa_id, DATA).
    - consumo diário: soma de QuantidadeConsumoAtivo no dia
    """
    out = (
        df.groupby(["empresa_id", cfg.date_col], as_index=False)
          .agg({cfg.raw_target_col: "sum"})
          .rename(columns={cfg.raw_target_col: cfg.target_col})
          .sort_values(["empresa_id", cfg.date_col])
          .reset_index(drop=True)
    )
    return out


def enforce_daily_per_company(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.enforce_daily_frequency:
        return df

    parts = []
    for emp_id, g in df.groupby("empresa_id", sort=False):
        g = g.sort_values(cfg.date_col).set_index(cfg.date_col)
        full_idx = pd.date_range(g.index.min(), g.index.max(), freq="D")
        g = g.reindex(full_idx)
        g.index.name = cfg.date_col
        g["empresa_id"] = emp_id
        # alvo não pode ser inventado -> mantém NaN onde não existe
        parts.append(g.reset_index())

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["empresa_id", cfg.date_col]).reset_index(drop=True)
    return out


def normalize_per_company(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    stats_df = (
        df.groupby("empresa_id")[cfg.target_col]
          .agg(y_mean="mean", y_std="std")
          .reset_index()
    )
    stats_df["y_std"] = stats_df["y_std"].fillna(1.0).replace(0.0, 1.0)

    df = df.merge(stats_df, on="empresa_id", how="left")
    df["y_norm"] = (df[cfg.target_col] - df["y_mean"]) / df["y_std"]

    stats = {
        int(r["empresa_id"]): {"y_mean": float(r["y_mean"]), "y_std": float(r["y_std"])}
        for _, r in stats_df.iterrows()
    }
    return df, stats


def make_windows(df: pd.DataFrame, cfg: Config, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_list, y_list, emp_list, ymeta_list = [], [], [], []

    for emp_id, g in df.groupby("empresa_id", sort=False):
        g = g.sort_values(cfg.date_col).reset_index(drop=True)

        # remove dias sem consumo (NaN) — não imputamos o alvo
        g = g.dropna(subset=["y_norm"]).reset_index(drop=True)
        if len(g) < cfg.lookback + cfg.horizon:
            continue

        Xnum = g[feature_cols].to_numpy(dtype=np.float32)
        y = g["y_norm"].to_numpy(dtype=np.float32)

        y_mean = float(g["y_mean"].iloc[0])
        y_std = float(g["y_std"].iloc[0])

        for i in range(cfg.lookback, len(g) - cfg.horizon + 1):
            X_list.append(Xnum[i - cfg.lookback: i])
            y_list.append(float(y[i + cfg.horizon - 1]))
            emp_list.append(int(emp_id))
            ymeta_list.append((y_mean, y_std))

    if not X_list:
        raise RuntimeError("Não foi possível gerar janelas. Ajuste lookback/horizon ou verifique dados.")

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    emp_ids = np.array(emp_list, dtype=np.int32)
    y_meta = np.array(ymeta_list, dtype=np.float32)  # [mean, std]
    return X, y, emp_ids, y_meta


def temporal_split_indices(df: pd.DataFrame, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = [], []
    cursor = 0

    for emp_id, g in df.groupby("empresa_id", sort=False):
        g = g.sort_values(cfg.date_col).reset_index(drop=True)
        g = g.dropna(subset=["y_norm"]).reset_index(drop=True)

        n_possible = len(g) - cfg.lookback - cfg.horizon + 1
        if n_possible <= 0:
            continue

        split = int(np.floor(cfg.train_ratio * n_possible))
        if n_possible >= 2:
            split = min(max(split, 1), n_possible - 1)
        else:
            split = n_possible

        train_idx.extend(range(cursor, cursor + split))
        test_idx.extend(range(cursor + split, cursor + n_possible))
        cursor += n_possible

    return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)


def main():
    cfg = Config()

    out_dir = _resolve_path(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    in_path = _resolve_path(cfg.input_csv)
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {in_path}")

    df = read_mcp_csv(cfg)
    df, id_to_name = build_company_id_map(df, cfg)

    daily = aggregate_daily(df, cfg)
    daily = enforce_daily_per_company(daily, cfg)

    daily = cyclical_time_features(daily, cfg.date_col)
    daily, stats = normalize_per_company(daily, cfg)

    # ✅ features mínimas (somente calendário)
    feature_cols = ["dow_sin", "dow_cos", "doy_sin", "doy_cos", "month"]

    # garante sem NaN em features (calendário não deveria ter NaN, mas por segurança)
    daily[feature_cols] = daily[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X, y, emp_ids, y_meta = make_windows(daily, cfg, feature_cols)
    train_idx, test_idx = temporal_split_indices(daily, cfg)

    out_npz = os.path.join(out_dir, "global_dataset_daily.npz")
    np.savez_compressed(
        out_npz,
        X_train=X[train_idx], y_train=y[train_idx], emp_train=emp_ids[train_idx], ymeta_train=y_meta[train_idx],
        X_test=X[test_idx], y_test=y[test_idx], emp_test=emp_ids[test_idx], ymeta_test=y_meta[test_idx],
    )

    meta = {
        "dataset_type": "MCP_V1_MINIMAL",
        "input_csv": cfg.input_csv,
        "lookback": cfg.lookback,
        "horizon": cfg.horizon,
        "train_ratio": cfg.train_ratio,
        "feature_cols": feature_cols,
        "date_col": cfg.date_col,
        "company_col": cfg.company_col,
        "raw_target_col": cfg.raw_target_col,
        "target_col": cfg.target_col,
        "n_empresas": int(emp_ids.max() + 1) if emp_ids.size else 0,
        "company_map": {str(k): v for k, v in id_to_name.items()},
        "y_stats": stats,
    }

    out_meta = os.path.join(out_dir, "global_dataset_daily_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("Dataset MCP concluído.")
    print(f"- NPZ:  {out_npz}")
    print(f"- META: {out_meta}")
    print(f"- feature_cols: {feature_cols}")
    print(f"- X_train: {X[train_idx].shape} | X_test: {X[test_idx].shape}")
    print(f"- n_empresas: {meta['n_empresas']}")


if __name__ == "__main__":
    main()