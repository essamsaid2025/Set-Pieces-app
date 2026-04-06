import os
import re
from typing import Dict, List, Optional

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


# =========================================================
# COLUMN ALIAS ENGINE
# =========================================================
def _slug(text: str) -> str:
    """
    Normalize a column name aggressively:
    - lowercase
    - trim
    - remove spaces, underscores, hyphens, slashes, dots, brackets
    Example:
        "Set Piece Type" -> "setpiecetype"
        "end_x" -> "endx"
        "X 2" -> "x2"
    """
    text = str(text).strip().lower()
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)  # zero-width chars / BOM leftovers
    text = re.sub(r"[\s_\-\/\\\.\(\)\[\]\{\}:]+", "", text)
    return text


COLUMN_ALIASES: Dict[str, List[str]] = {
    "match_id": [
        "match_id", "match id", "matchid", "game_id", "game id", "gameid",
        "fixture_id", "fixture id", "fixtureid", "id_match", "match"
    ],
    "team": [
        "team", "team_name", "team name", "club", "club_name", "club name",
        "squad", "attacking_team", "attacking team", "team in possession"
    ],
    "opponent": [
        "opponent", "opponent_team", "opponent team", "against", "vs",
        "versus", "opposition", "defending_team", "defending team", "rival"
    ],
    "date": [
        "date", "match_date", "match date", "game_date", "game date",
        "fixture_date", "fixture date", "event_date", "event date"
    ],
    "competition": [
        "competition", "comp", "league", "tournament", "competition_name",
        "competition name"
    ],
    "sequence_id": [
        "sequence_id", "sequence id", "sequenceid",
        "seq_id", "seq id", "seqid",
        "routine_id", "routine id", "routineid",
        "pattern_id", "pattern id", "patternid",
        "possession_id", "possession id", "possessionid",
        "play_id", "play id", "playid"
    ],
    "set_piece_type": [
        "set_piece_type", "set piece type", "setpiecetype",
        "set_piece", "set piece", "setpiece",
        "sp_type", "sptype",
        "restart_type", "restart type", "restarttype",
        "dead_ball_type", "dead ball type", "deadballtype",
        "type", "event_type", "event type", "play_type", "play type"
    ],
    "event": [
        "event", "action", "event_name", "event name", "action_type", "action type"
    ],
    "side": [
        "side", "flank", "wing", "left_right", "left right",
        "delivery_side", "delivery side", "set_piece_side", "set piece side"
    ],
    "delivery_type": [
        "delivery_type", "delivery type", "deliverytype",
        "delivery", "cross_type", "cross type", "crosstype",
        "ball_type", "ball type", "service_type", "service type",
        "delivery_style", "delivery style"
    ],
    "taker": [
        "taker", "set_piece_taker", "set piece taker",
        "delivery_player", "delivery player",
        "kicker", "taken_by", "taken by", "server", "assist_player", "assist player"
    ],
    "phase": [
        "phase", "set_piece_phase", "set piece phase",
        "play_phase", "play phase", "sequence_phase", "sequence phase",
        "event_phase", "event phase"
    ],
    "target_zone": [
        "target_zone", "target zone", "targetzone",
        "zone", "delivery_zone", "delivery zone",
        "end_zone", "end zone", "landing_zone", "landing zone",
        "target_area", "target area", "aim_zone", "aim zone"
    ],
    "outcome": [
        "outcome", "result_outcome", "result outcome", "success",
        "event_outcome", "event outcome", "outcome_type", "outcome type",
        "status"
    ],
    "result": [
        "result", "final_result", "final result", "shot_result", "shot result",
        "play_result", "play result", "sequence_result", "sequence result"
    ],
    "target_player": [
        "target_player", "target player", "targetplayer",
        "receiver", "receiver_player", "receiver player",
        "intended_target", "intended target", "aimed_player", "aimed player"
    ],
    "first_contact_player": [
        "first_contact_player", "first contact player", "firstcontactplayer",
        "first_touch_player", "first touch player",
        "contact_player", "contact player",
        "first_header", "first header", "first_ball_player", "first ball player"
    ],
    "shot_player": [
        "shot_player", "shot player", "shotplayer",
        "shooter", "finisher", "final_action_player", "final action player"
    ],
    "routine_type": [
        "routine_type", "routine type", "routinetype",
        "routine", "pattern", "pattern_type", "pattern type",
        "play_pattern", "play pattern", "set_piece_routine", "set piece routine"
    ],

    # Coordinates
    "x": [
        "x", "start_x", "start x", "from_x", "from x", "origin_x", "origin x",
        "x1", "x_1", "x 1", "startx", "fromx", "originx",
        "loc_x", "loc x", "location_x", "location x",
        "start_location_x", "start location x"
    ],
    "y": [
        "y", "start_y", "start y", "from_y", "from y", "origin_y", "origin y",
        "y1", "y_1", "y 1", "starty", "fromy", "originy",
        "loc_y", "loc y", "location_y", "location y",
        "start_location_y", "start location y"
    ],
    "x2": [
        "x2", "x_2", "x 2", "end_x", "end x", "to_x", "to x",
        "target_x", "target x", "finish_x", "finish x",
        "receive_x", "receive x", "dest_x", "dest x", "destination_x", "destination x",
        "endx", "tox", "targetx", "finishx", "receivex", "destx", "destinationx"
    ],
    "y2": [
        "y2", "y_2", "y 2", "end_y", "end y", "to_y", "to y",
        "target_y", "target y", "finish_y", "finish y",
        "receive_y", "receive y", "dest_y", "dest y", "destination_y", "destination y",
        "endy", "toy", "targety", "finishy", "receivey", "desty", "destinationy"
    ],
    "x3": [
        "x3", "x_3", "x 3", "third_x", "third x", "shot_x", "shot x",
        "second_end_x", "second end x", "final_x", "final x"
    ],
    "y3": [
        "y3", "y_3", "y 3", "third_y", "third y", "shot_y", "shot y",
        "second_end_y", "second end y", "final_y", "final y"
    ],

    # Binary / numeric tactical
    "first_contact_win": [
        "first_contact_win", "first contact win", "firstcontactwin",
        "first_contact_won", "first contact won", "firstcontactwon",
        "first_header_win", "first header win", "first duel won",
        "fc_win", "fcwin", "first_touch_win", "first touch win"
    ],
    "second_ball_win": [
        "second_ball_win", "second ball win", "secondballwin",
        "second_ball_won", "second ball won", "secondballwon",
        "sb_win", "sbwin", "recovery_win", "recovery win", "won_second_ball"
    ],
    "players_near_post": [
        "players_near_post", "players near post", "playersnearpost",
        "near_post_players", "near post players",
        "attackers_near_post", "attackers near post",
        "near_post_attackers", "near post attackers"
    ],
    "players_far_post": [
        "players_far_post", "players far post", "playersfarpost",
        "far_post_players", "far post players",
        "attackers_far_post", "attackers far post",
        "far_post_attackers", "far post attackers"
    ],
    "players_6yard": [
        "players_6yard", "players 6yard", "players_6_yard", "players 6 yard",
        "six_yard_players", "six yard players",
        "players_in_6yard", "players in 6yard", "players in 6 yard",
        "attackers_6yard", "attackers 6yard", "attackers 6 yard"
    ],
    "players_penalty": [
        "players_penalty", "players penalty",
        "penalty_players", "penalty players",
        "players_penalty_area", "players penalty area",
        "players_in_penalty", "players in penalty",
        "attackers_penalty", "attackers penalty",
        "penalty_area_players", "penalty area players"
    ],
    "defenders_near_post": [
        "defenders_near_post", "defenders near post", "defendersnearpost",
        "near_post_defenders", "near post defenders",
        "def_near_post", "def near post"
    ],
    "defenders_far_post": [
        "defenders_far_post", "defenders far post", "defendersfarpost",
        "far_post_defenders", "far post defenders",
        "def_far_post", "def far post"
    ],
    "xg": [
        "xg", "expected_goals", "expected goals", "exp_goals", "exp goals",
        "shot_xg", "shot xg", "x_g"
    ],
}


def _build_alias_lookup() -> Dict[str, str]:
    """
    Build slug -> canonical_name lookup.
    First alias wins if duplicates happen.
    """
    lookup: Dict[str, str] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        all_names = [canonical] + aliases
        for name in all_names:
            key = _slug(name)
            if key and key not in lookup:
                lookup[key] = canonical

    return lookup


ALIAS_LOOKUP = _build_alias_lookup()


def _dedupe_columns(cols: List[str]) -> List[str]:
    """
    Ensure unique output column names if two source columns map to same name.
    Keeps first as-is, later ones become:
        x -> x__dup1, x__dup2, ...
    """
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


# =========================================================
# LOADERS
# =========================================================
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


# =========================================================
# NORMALIZERS
# =========================================================
def _normalize_text_series(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .replace(
            {
                "nan": pd.NA,
                "": pd.NA,
                "none": pd.NA,
                "null": pd.NA,
                "na": pd.NA,
                "n/a": pd.NA,
            }
        )
    )


def _to_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
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
    """
    Priority:
    1) set_piece_type
    2) event
    3) outcome
    """
    if "set_piece_type" not in out.columns:
        out["set_piece_type"] = pd.NA

    fallback_event = _first_existing_series(out, ["event"])
    fallback_outcome = _first_existing_series(out, ["outcome"])

    if fallback_event is not None:
        out["set_piece_type"] = out["set_piece_type"].fillna(fallback_event)

    if fallback_outcome is not None:
        out["set_piece_type"] = out["set_piece_type"].fillna(fallback_outcome)

    if "set_piece_type" in out.columns:
        out["set_piece_type"] = out["set_piece_type"].replace(
            {
                "free kick": "free_kick",
                "freekick": "free_kick",
                "free-kick": "free_kick",
                "fk": "free_kick",
                "corner kick": "corner",
                "cornerkick": "corner",
                "ck": "corner",
                "throw in": "throw_in",
                "throw-in": "throw_in",
                "throwin": "throw_in",
                "penalty kick": "penalty",
                "penaltykick": "penalty",
                "goal kick": "goal_kick",
                "goalkick": "goal_kick",
            }
        )

    return out


def _cleanup_phase(out: pd.DataFrame) -> pd.DataFrame:
    if "phase" in out.columns:
        out["phase"] = out["phase"].replace(
            {
                "delivery": "delivery",
                "ball in": "delivery",
                "first contact": "first_contact",
                "first_contact": "first_contact",
                "first touch": "first_contact",
                "second ball": "second_ball",
                "second_ball": "second_ball",
                "shot": "shot",
                "finish": "shot",
                "recycle": "recycle",
            }
        )
    return out


def _cleanup_side(out: pd.DataFrame) -> pd.DataFrame:
    if "side" in out.columns:
        out["side"] = out["side"].replace(
            {
                "l": "left",
                "left side": "left",
                "left_side": "left",
                "r": "right",
                "right side": "right",
                "right_side": "right",
                "centre": "center",
                "central": "center",
                "middle": "center",
            }
        )
    return out


def _cleanup_delivery_type(out: pd.DataFrame) -> pd.DataFrame:
    if "delivery_type" in out.columns:
        out["delivery_type"] = out["delivery_type"].replace(
            {
                "in swing": "inswing",
                "in-swing": "inswing",
                "in_swing": "inswing",
                "outswinger": "outswing",
                "out swing": "outswing",
                "out-swing": "outswing",
                "out_swing": "outswing",
                "straight ball": "straight",
                "driven ball": "driven",
                "driven cross": "driven",
                "short corner": "short",
                "short free kick": "short",
                "cut back": "cutback",
                "cut-back": "cutback",
            }
        )
    return out


def _cleanup_outcome(out: pd.DataFrame) -> pd.DataFrame:
    if "outcome" in out.columns:
        out["outcome"] = out["outcome"].replace(
            {
                "success": "successful",
                "successful": "successful",
                "complete": "successful",
                "completed": "successful",
                "win": "successful",
                "won": "successful",
                "fail": "unsuccessful",
                "failed": "unsuccessful",
                "failure": "unsuccessful",
                "lost": "unsuccessful",
                "loss": "unsuccessful",
                "unsuccessful": "unsuccessful",
            }
        )
    return out


def _cleanup_target_zone(out: pd.DataFrame) -> pd.DataFrame:
    if "target_zone" in out.columns:
        out["target_zone"] = out["target_zone"].replace(
            {
                "near post": "near_post",
                "nearpost": "near_post",
                "far post": "far_post",
                "farpost": "far_post",
                "6 yard": "six_yard",
                "6-yard": "six_yard",
                "six yard": "six_yard",
                "penalty spot": "penalty_spot",
                "penaltyspot": "penalty_spot",
                "central": "central",
                "center": "central",
                "centre": "central",
                "edge of box": "edge_box",
                "edge box": "edge_box",
            }
        )
    return out


def normalize_set_piece_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1) Strong auto mapping for many possible headers
    out = _canonicalize_columns(out)

    # 2) Normalize text columns
    for c in TEXT_COLS:
        if c in out.columns:
            out[c] = _normalize_text_series(out[c])

    # 3) Numeric columns
    out = _to_num(out, NUMERIC_COLS)

    # 4) Cleanup categorical values
    out = _cleanup_phase(out)
    out = _cleanup_side(out)
    out = _cleanup_delivery_type(out)
    out = _cleanup_outcome(out)
    out = _cleanup_target_zone(out)

    # 5) Resolve set piece type if missing
    out = _resolve_set_piece_type(out)

    return out


# =========================================================
# HELPERS
# =========================================================
def ensure_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]


def bool01(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

    s = series.astype(str).str.strip().str.lower()
    return s.isin(
        [
            "1", "true", "yes", "y",
            "won", "win", "successful", "success"
        ]
    ).astype(int)


def apply_flip_y(df: pd.DataFrame, flip_y: bool = False, pitch_width: float = 64) -> pd.DataFrame:
    """
    Flip Y to match a 100x64 custom pitch.
    Current charts use pitch_width=64.
    """
    out = df.copy()
    if not flip_y:
        return out

    for c in ["y", "y2", "y3"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = pitch_width - out[c]

    return out
