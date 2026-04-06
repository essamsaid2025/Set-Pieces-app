import os
from typing import List

import pandas as pd


TEXT_COLS = [
    "match_id",
    "team",
    "opponent",
    "date",
    "competition",
    "set_piece_type",
    "event",
    "side",
    "delivery_type",
    "taker",
    "phase",
    "target_zone",
    "outcome",
    "result",
    "target_player",
    "first_contact_player",
    "shot_player",
    "routine_type",
]

NUMERIC_COLS = [
    "sequence_id",
    "x", "y", "x2", "y2", "x3", "y3",
    "first_contact_win", "second_ball_win",
    "players_near_post", "players_far_post", "players_6yard", "players_penalty",
    "defenders_near_post", "defenders_far_post",
    "xg",
]


def load_data(uploaded_file) -> pd.DataFrame:
    ext = os.path.splitext(uploaded_file.name)[1].lower()

    if ext == ".csv":
        for enc in ["utf-8", "utf-8-sig", "cp1256", "cp1252", "latin1"]:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc)
            except Exception:
                continue
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)

    if ext in [".xlsx", ".xls"]:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Use CSV or Excel.")


def _normalize_text_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"nan": pd.NA, "": pd.NA})


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _resolve_set_piece_type(out: pd.DataFrame) -> pd.DataFrame:
    """
    Priority:
    1) set_piece_type
    2) event
    3) outcome
    """
    if "set_piece_type" not in out.columns:
        out["set_piece_type"] = pd.NA

    if "event" in out.columns:
        out["set_piece_type"] = out["set_piece_type"].fillna(out["event"])

    if "outcome" in out.columns:
        out["set_piece_type"] = out["set_piece_type"].fillna(out["outcome"])

    if "set_piece_type" in out.columns:
        out["set_piece_type"] = out["set_piece_type"].replace(
            {
                "free kick": "free_kick",
                "freekick": "free_kick",
                "corner kick": "corner",
            }
        )

    return out


def normalize_set_piece_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # lowercase all column names first
    out.columns = [str(c).strip().lower() for c in out.columns]

    # auto column mapping
    rename_map = {
        "type": "set_piece_type",
        "event_type": "set_piece_type",
        "start_x": "x",
        "start_y": "y",
        "end_x": "x2",
        "end_y": "y2",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})

    # normalize text columns
    for c in TEXT_COLS:
        if c in out.columns:
            out[c] = _normalize_text_series(out[c])

    # numeric columns
    out = _to_num(out, NUMERIC_COLS)

    # phase cleanup
    if "phase" in out.columns:
        out["phase"] = out["phase"].replace(
            {
                "first contact": "first_contact",
                "second ball": "second_ball",
            }
        )

    # delivery cleanup
    if "delivery_type" in out.columns:
        out["delivery_type"] = out["delivery_type"].replace(
            {
                "in swing": "inswing",
                "out swing": "outswing",
            }
        )

    # outcome cleanup
    if "outcome" in out.columns:
        out["outcome"] = out["outcome"].replace(
            {
                "success": "successful",
                "fail": "unsuccessful",
                "failed": "unsuccessful",
            }
        )

    # if set_piece_type is missing, use event or outcome
    out = _resolve_set_piece_type(out)

    return out


def ensure_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def bool01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1", "true", "yes", "y"]).astype(int)


def apply_flip_y(df: pd.DataFrame, flip_y: bool = False) -> pd.DataFrame:
    out = df.copy()
    if not flip_y:
        return out

    for c in ["y", "y2", "y3"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = 100 - out[c]

    return out
