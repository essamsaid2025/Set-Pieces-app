import os
import re
from typing import Dict, List, Optional

import pandas as pd


TEXT_COLS = [
    "match_id", "team", "opponent", "date", "competition", "set_piece_type",
    "analysis_phase", "event", "side", "delivery_type", "taker", "phase", "target_zone",
    "outcome", "result", "target_player", "first_contact_player",
    "shot_player", "routine_type", "lost_first_contact_player",
]

NUMERIC_COLS = [
    "sequence_id",
    # legacy coords
    "x", "y", "x2", "y2", "x3", "y3",
    # explicit event coords
    "delivery_start_x", "delivery_start_y",
    "delivery_end_x", "delivery_end_y",
    "first_contact_x", "first_contact_y",
    "second_ball_x", "second_ball_y",
    "clearance_x", "clearance_y",
    "shot_x", "shot_y",
    # outcomes / structure
    "first_contact_win", "second_ball_win",
    "players_near_post", "players_far_post", "players_6yard", "players_penalty",
    "players_small_area", "players_penalty_area",
    "attack_players_near_post", "attack_players_far_post",
    "attack_players_small_area", "attack_players_penalty_area",
    "defenders_near_post", "defenders_far_post",
    "defenders_small_area", "defenders_penalty_area",
    "man_marking_in_box", "zonal_in_box", "xg",
]

_BOOL_COLS = ["first_contact_win", "second_ball_win"]


def _slug(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\s_\-\/\\.\(\)\[\]\{\}:]+", "", text)
    return text


COLUMN_ALIASES: Dict[str, List[str]] = {
    "match_id": ["match_id","match id","matchid","game_id","game id","fixture_id"],
    "team": ["team","team_name","team name","club","squad"],
    "opponent": ["opponent","opponent_team","opponent team","against","vs","opposition"],
    "date": ["date","match_date","match date","game_date","fixture_date"],
    "competition": ["competition","comp","league","tournament"],
    "sequence_id": ["sequence_id","sequence id","seq_id","routine_id","pattern_id","play_id"],
    "set_piece_type": ["set_piece_type","set piece type","setpiece","set piece","event_type","restart_type"],
    "analysis_phase": ["analysis_phase", "analysis phase", "phase_type", "attack_defence", "attack/defence", "type"],
    "event": ["event","action","event_name","action_type"],
    "side": ["side","flank","wing","left_right","delivery_side"],
    "delivery_type": ["delivery_type","delivery type","delivery","cross_type","service_type"],
    "taker": ["taker","kicker","taken_by","delivery_player","server"],
    "phase": ["phase","set_piece_phase","play_phase","event_phase"],
    "target_zone": ["target_zone","target zone","zone","delivery_zone","end_zone","landing_zone"],
    "outcome": ["outcome","success","event_outcome","status"],
    "result": ["result","final_result","shot_result","play_result"],
    "target_player": ["target_player","target player","receiver","intended_target"],
    "first_contact_player": ["first_contact_player","first contact player","first_touch_player","contact_player"],
    "lost_first_contact_player": ["lost_first_contact_player", "lost first contact player"],
    "shot_player": ["shot_player","shot player","shooter","finisher"],
    "routine_type": ["routine_type","routine type","routine","pattern","play_pattern"],
    # legacy / explicit coords
    "x": ["x","start_x","start x","from_x","origin_x","x1","x 1","startx"],
    "y": ["y","start_y","start y","from_y","origin_y","y1","y 1","starty"],
    "x2": ["x2","x 2","x_2","end_x","end x","to_x","target_x","finish_x","destination_x"],
    "y2": ["y2","y 2","y_2","end_y","end y","to_y","target_y","finish_y","destination_y"],
    "x3": ["x3","x 3","x_3","third_x","final_x"],
    "y3": ["y3","y 3","y_3","third_y","final_y"],
    "delivery_start_x": ["delivery_start_x", "delivery start x"],
    "delivery_start_y": ["delivery_start_y", "delivery start y"],
    "delivery_end_x": ["delivery_end_x", "delivery end x"],
    "delivery_end_y": ["delivery_end_y", "delivery end y"],
    "first_contact_x": ["first_contact_x", "first contact x"],
    "first_contact_y": ["first_contact_y", "first contact y"],
    "second_ball_x": ["second_ball_x", "second ball x"],
    "second_ball_y": ["second_ball_y", "second ball y"],
    "clearance_x": ["clearance_x", "clearance x"],
    "clearance_y": ["clearance_y", "clearance y"],
    "shot_x": ["shot_x", "shot x"],
    "shot_y": ["shot_y", "shot y"],
    "first_contact_win": ["first_contact_win","first contact win","fc_win","first_header_win"],
    "second_ball_win": ["second_ball_win","second ball win","sb_win","won_second_ball"],
    "players_near_post": ["players_near_post","players near post","near_post_players","attackers_near_post"],
    "players_far_post": ["players_far_post","players far post","far_post_players","attackers_far_post"],
    "players_6yard": ["players_6yard","players 6yard","players 6 yard","six_yard_players","attackers_6yard"],
    "players_penalty": ["players_penalty","penalty_players","players_penalty_area","attackers_penalty","players_penalty area"],
    "players_small_area": ["players_small_area","players small area","small_area_players"],
    "players_penalty_area": ["players_penalty_area","players penalty area"],
    "attack_players_near_post": ["attack_players_near_post"],
    "attack_players_far_post": ["attack_players_far_post"],
    "attack_players_small_area": ["attack_players_small_area"],
    "attack_players_penalty_area": ["attack_players_penalty_area"],
    "defenders_near_post": ["defenders_near_post","near_post_defenders","def_near_post"],
    "defenders_far_post": ["defenders_far_post","far_post_defenders","def_far_post"],
    "defenders_small_area": ["defenders_small_area"],
    "defenders_penalty_area": ["defenders_penalty_area"],
    "man_marking_in_box": ["man_marking_in_box", "man marking in box"],
    "zonal_in_box": ["zonal_in_box", "zonal in box"],
    "xg": ["xg","expected_goals","exp_goals","shot_xg"],
}


def _build_alias_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for name in [canonical] + aliases:
            key = _slug(name)
            if key and key not in lookup:
                lookup[key] = canonical
    return lookup


ALIAS_LOOKUP = _build_alias_lookup()


def _dedupe_columns(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapped_cols: List[str] = []
    for col in out.columns:
        raw = str(col).strip()
        slug = _slug(raw)
        canonical = ALIAS_LOOKUP.get(slug, raw.strip().lower())
        canonical = re.sub(r"\s+", "_", canonical)
        mapped_cols.append(canonical)
    out.columns = _dedupe_columns(mapped_cols)
    return out


def load_data(uploaded_file) -> pd.DataFrame:
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".csv":
        for enc in ["utf-8","utf-8-sig","cp1256","cp1252","latin1"]:
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
    return (s.astype(str).str.strip().str.lower()
            .replace({"nan": pd.NA,"": pd.NA,"none": pd.NA,"null": pd.NA,"na": pd.NA,"n/a": pd.NA}))


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _convert_bool_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    yes_vals = {"yes","y","true","1","won","win","successful","success"}
    no_vals  = {"no","n","false","0","lost","lose","unsuccessful","failed"}
    for c in cols:
        if c in out.columns:
            s = out[c].astype(str).str.strip().str.lower()
            if s.isin(yes_vals | no_vals | {pd.NA, "nan", "none", ""}).any():
                out[c] = s.map(lambda v: 1 if v in yes_vals else (0 if v in no_vals else pd.NA))
                out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _first_existing_series(out: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    for c in candidates:
        if c in out.columns:
            s = out[c]
            if s.notna().any():
                return s
    return None


def _resolve_set_piece_type(out: pd.DataFrame) -> pd.DataFrame:
    if "set_piece_type" not in out.columns:
        out["set_piece_type"] = pd.NA
    fallback_event = _first_existing_series(out, ["event"])
    fallback_outcome = _first_existing_series(out, ["outcome"])
    if fallback_event is not None:
        out["set_piece_type"] = out["set_piece_type"].fillna(fallback_event)
    if fallback_outcome is not None:
        out["set_piece_type"] = out["set_piece_type"].fillna(fallback_outcome)
    if "set_piece_type" in out.columns:
        out["set_piece_type"] = out["set_piece_type"].replace({
            "free kick":"free_kick","freekick":"free_kick","free-kick":"free_kick","fk":"free_kick",
            "corner kick":"corner","cornerkick":"corner","ck":"corner",
            "throw in":"throw_in","throw-in":"throw_in","throwin":"throw_in",
        })
    return out


def _resolve_analysis_phase(out: pd.DataFrame) -> pd.DataFrame:
    if "analysis_phase" not in out.columns:
        out["analysis_phase"] = pd.NA
    out["analysis_phase"] = out["analysis_phase"].replace({
        "attacking": "attack",
        "offence": "attack",
        "offense": "attack",
        "defensive": "defence",
        "defense": "defence",
    })
    return out


def _sync_explicit_coords_to_legacy(out: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "x": ["delivery_start_x"],
        "y": ["delivery_start_y"],
        "x2": ["delivery_end_x", "first_contact_x"],
        "y2": ["delivery_end_y", "first_contact_y"],
        "x3": ["shot_x", "second_ball_x", "clearance_x"],
        "y3": ["shot_y", "second_ball_y", "clearance_y"],
    }
    for legacy, candidates in mapping.items():
        if legacy not in out.columns:
            out[legacy] = pd.NA
        for c in candidates:
            if c in out.columns:
                out[legacy] = out[legacy].fillna(out[c])
    return out


def normalize_set_piece_df(df: pd.DataFrame) -> pd.DataFrame:
    out = _canonicalize_columns(df.copy())
    for c in TEXT_COLS:
        if c in out.columns:
            out[c] = _normalize_text_series(out[c])
    out = _resolve_analysis_phase(out)
    out = _convert_bool_cols(out, _BOOL_COLS)
    out = _to_num(out, NUMERIC_COLS)
    if "phase" in out.columns:
        out["phase"] = out["phase"].replace({"first contact":"first_contact","second ball":"second_ball"})
    if "delivery_type" in out.columns:
        out["delivery_type"] = out["delivery_type"].replace({
            "in swing":"inswing","in-swing":"inswing","out swing":"outswing","out-swing":"outswing",
            "short corner":"short","short free kick":"short","driven ball":"driven",
        })
    if "outcome" in out.columns:
        out["outcome"] = out["outcome"].replace({"success":"successful","fail":"unsuccessful","failed":"unsuccessful"})
    out = _resolve_set_piece_type(out)
    out = _sync_explicit_coords_to_legacy(out)
    return out


def ensure_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def bool01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["1","true","yes","y","won","win","successful","success"]).astype(int)


def apply_flip_y(df: pd.DataFrame, flip_y: bool = False, pitch_width: float = 64) -> pd.DataFrame:
    out = df.copy()
    if not flip_y:
        return out
    for c in [
        "y","y2","y3",
        "delivery_start_y","delivery_end_y","first_contact_y",
        "second_ball_y","clearance_y","shot_y",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = pitch_width - out[c]
    return out
