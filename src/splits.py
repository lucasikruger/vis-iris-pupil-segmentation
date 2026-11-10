"""Particiones LODO: reproduce RandomState y orden de sujetos del código original."""
import numpy as np

def partition(df, fold="mobius", seed=42):
    df = df.copy()
    if df.dataset.str.contains("webcam").any():
        raise ValueError("Webcam es exclusivamente prueba; no puede entrar al entrenamiento")
    bg = df.dataset.isin(["closed_eyes_bg", "white_eyes_bg"])
    test = df.dataset.eq(fold)
    df["split"] = "train"
    df.loc[test, "split"] = "test"
    rng = np.random.RandomState(seed)
    for _, rows in df[~bg & ~test].groupby("dataset"):
        subjects = np.asarray(rows.subject_id.unique(), dtype=object).copy()
        rng.shuffle(subjects)
        chosen = subjects[:max(1, int(round(len(subjects)*0.15)))]
        df.loc[rows.index[rows.subject_id.isin(chosen)], "split"] = "val"
    return df
