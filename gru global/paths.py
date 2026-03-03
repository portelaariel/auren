import os
import glob
import shutil


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_if_exists(path: str) -> None:
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def _base_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def artifacts_dir() -> str:
    env = os.environ.get("ARTIFACTS_DIR")
    if env:
        return env if os.path.isabs(env) else os.path.join(_base_dir(), env)
    return os.path.join(_base_dir(), "artifacts")


def dataset_npz_path() -> str:
    return os.path.join(artifacts_dir(), "global_dataset_daily.npz")


def dataset_meta_path() -> str:
    return os.path.join(artifacts_dir(), "global_dataset_daily_meta.json")


def best_hps_path() -> str:
    return os.path.join(artifacts_dir(), "best_hyperparameters.json")


def official_model_path() -> str:
    return os.path.join(artifacts_dir(), "model_global_gru.keras")


def tuned_model_path() -> str:
    return os.path.join(artifacts_dir(), "best_global_gru.keras")


def resolve_model_path(prefer_official: bool = True) -> str:
    art = artifacts_dir()
    off = official_model_path()
    best = tuned_model_path()

    if prefer_official and os.path.isfile(off):
        return off
    if os.path.isfile(best):
        return best
    if os.path.isfile(off):
        return off

    candidates = glob.glob(os.path.join(art, "*.keras"))
    if not candidates:
        raise FileNotFoundError(
            f"Nenhum modelo .keras encontrado em {art}. "
            "Rode train_global_gru.py ou random_search_tuner.py."
        )
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def plots_dir() -> str:
    return os.path.join(artifacts_dir(), "plots")