#!/usr/bin/env python
"""
Build a clean player_snapshot table from raw hackathon CSVs.

Usage:
    Put this file in the ROOT of your project (the same folder that has the `data/` folder)
    and run:

        python build_player_snapshot.py

Input (in data/):
    players.csv
    player_valuations.csv
    appearances.csv
    games.csv
    clubs.csv
    competitions.csv
    transfers.csv
    club_games.csv

Output:
    data/processed/player_snapshot.parquet
"""
import os
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SRC_DIR = os.path.dirname(SCRIPT_DIR)

PROJECT_ROOT = os.path.dirname(SRC_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)



# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def get_season_year_from_date(d):
    """Map a calendar date to a 'season year' (European style).

    If date is in Jul–Dec: season year = this year
    If date is in Jan–Jun: season year = previous year
    """
    if pd.isna(d):
        return np.nan
    year = d.year
    return year if d.month >= 7 else year - 1


def check_nulls(df, name="DataFrame"):
    """Check for null values in the dataframe."""
    print(f"\n[{name}] Null Value Check:")
    print("-" * 80)

    null_counts = df.isnull().sum()
    null_pct = (null_counts / len(df) * 100)

    # Columns where nulls are expected from original CSVs
    expected_nulls = ["foot", "country_of_citizenship", "sub_position"]

    # Columns that are critical for modeling
    critical_cols = [
        "market_value_in_eur",
        "age",
        "position",
        "minutes_total",
        "goals_per_90",
        "club_total_market_value",
        "league_strength",
    ]

    issues = []
    all_passed = True

    for col in df.columns:
        pct = null_pct[col]
        if col in expected_nulls:
            # Informational only
            print(f"    (Expected) {col:35s} {pct:6.2f}% nulls")
            continue

        if pct == 0:
            print(f"    ✅ {col:35s} {pct:6.2f}% nulls")
        elif pct <= 5:
            print(f"    ⚠️  {col:35s} {pct:6.2f}% nulls (≤ 5% is usually OK)")
        else:
            print(f"    ❌ {col:35s} {pct:6.2f}% nulls (> 5%)")
            if col in critical_cols:
                issues.append(f"Critical column '{col}' has {pct:.2f}% nulls")
                all_passed = False

    if all_passed:
        print("    ✅ All non-expected nulls within acceptable range")

    return {
        "null_counts": null_counts.to_dict(),
        "null_pct": null_pct.round(2).to_dict(),
        "issues": issues,
        "all_passed": all_passed,
    }


def test_modeling_readiness(df, name="DataFrame"):
    """Comprehensive test to check if data is ready for modeling."""
    print("\n" + "=" * 80)
    print(f"MODELING READINESS TEST: {name}")
    print("=" * 80)

    issues = []
    all_passed = True

    # 1) Basic info
    print("\n[1] Basic Information:")
    print(f"    Total rows   : {len(df):,}")
    print(f"    Total columns: {len(df.columns)}")

    if len(df) == 0:
        issues.append("DataFrame is empty!")
        print("    ❌ DataFrame is empty!")
        return False, issues

    # 2) Target variable check
    print("\n[2] Target Variable Check (y_growth):")
    if "y_growth" in df.columns:
        target_nulls = df["y_growth"].isnull().sum()
        target_pct = (target_nulls / len(df) * 100)
        print(f"    Nulls: {target_nulls} ({target_pct:.2f}%)")
        if target_nulls > 0:
            issues.append(f"Target variable has {target_pct:.2f}% nulls")
            all_passed = False
    else:
        print("    ❌ Target 'y_growth' NOT found!")
        issues.append("Target variable 'y_growth' is missing")
        all_passed = False

    # 3) Quick sanity on key numeric columns
    print("\n[3] Numeric Column Sanity Checks:")
    numeric_cols = [
        "market_value_in_eur",
        "mv_ratio_to_peak",
        "minutes_total",
        "goals_per_90",
        "assists_per_90",
        "mv_1y_change",
        "perf_1y_change",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            print(f"    ❌ {col} missing")
            issues.append(f"Missing numeric column: {col}")
            all_passed = False
            continue
        col_min = df[col].min()
        col_max = df[col].max()
        print(f"    {col:20s} min={col_min:10.4f}  max={col_max:10.4f}")

    # 4) Check key ID/date fields
    print("\n[4] Key ID / Date Fields:")
    for col in ["player_id", "snapshot_date", "season_year"]:
        if col not in df.columns:
            print(f"    ❌ {col} missing")
            issues.append(f"Missing key field: {col}")
            all_passed = False
        else:
            nulls = df[col].isnull().sum()
            print(f"    {col:15s} nulls={nulls}")

    if all_passed:
        print("\n✅ Data looks ready for modeling!")
    else:
        print("\n⚠️  Data has some issues that may need fixing before modeling.")

    return all_passed, issues


# ---------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------
def build_player_snapshot(data_dir: str = DATA_DIR,
                          output_path: str = None) -> pd.DataFrame:
    """End-to-end pipeline that builds the player_snapshot table."""

    if output_path is None:
        output_path = os.path.join(PROCESSED_DIR, "player_snapshot.parquet")

    print("=" * 80)
    print("BUILDING PLAYER SNAPSHOT")
    print("=" * 80)
    print(f"DATA_DIR        : {data_dir}")
    print(f"OUTPUT (parquet): {output_path}")

    # ------------------------------------------------------------------
    # 1) Load data
    # ------------------------------------------------------------------
    print("\nLoading data files...")
    players = pd.read_csv(os.path.join(data_dir, "players.csv"))
    valuations = pd.read_csv(os.path.join(data_dir, "player_valuations.csv"))
    apps = pd.read_csv(os.path.join(data_dir, "appearances.csv"))
    games = pd.read_csv(os.path.join(data_dir, "games.csv"))
    clubs = pd.read_csv(os.path.join(data_dir, "clubs.csv"))
    comps = pd.read_csv(os.path.join(data_dir, "competitions.csv"))
    transfers = pd.read_csv(os.path.join(data_dir, "transfers.csv"))
    club_games = pd.read_csv(os.path.join(data_dir, "club_games.csv"))

    print("✅ Data loaded successfully!")
    print(f"  Players      : {len(players):,}")
    print(f"  Valuations   : {len(valuations):,}")
    print(f"  Appearances  : {len(apps):,}")
    print(f"  Games        : {len(games):,}")
    print(f"  Clubs        : {len(clubs):,}")
    print(f"  Competitions : {len(comps):,}")
    print(f"  Transfers    : {len(transfers):,}")
    print(f"  Club Games   : {len(club_games):,}")

    # ------------------------------------------------------------------
    # 2) Base snapshot from player valuations
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CREATING BASE SNAPSHOT TABLE")
    print("=" * 80)

    snap = valuations.copy()
    snap["snapshot_date"] = pd.to_datetime(snap["date"], errors="coerce")
    snap = snap.sort_values(["player_id", "snapshot_date"])

    # Future MV & days to future
    snap["future_snapshot_date"] = snap.groupby("player_id")["snapshot_date"].shift(-1)
    snap["future_market_value"] = snap.groupby("player_id")["market_value_in_eur"].shift(-1)

    snap["delta_days_to_future"] = (
        snap["future_snapshot_date"] - snap["snapshot_date"]
    ).dt.days

    # Keep only 6–18 months horizon
    valid_mask = snap["delta_days_to_future"].between(180, 540)
    snap = snap[valid_mask].copy()

    # y_growth = log(MV_t+1y) − log(MV_t)
    snap["y_growth"] = np.log(snap["future_market_value"] + 1) - np.log(
        snap["market_value_in_eur"] + 1
    )

    print(f"Created {len(snap):,} valid snapshots")
    print(f"Unique players: {snap['player_id'].nunique():,}")

    # ------------------------------------------------------------------
    # 3) Add player-level information
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADDING PLAYER INFORMATION")
    print("=" * 80)

    p = players.copy()
    p["date_of_birth"] = pd.to_datetime(p["date_of_birth"], errors="coerce")
    p["contract_expiration_date"] = pd.to_datetime(
        p["contract_expiration_date"], errors="coerce"
    )

    snap = snap.merge(
        p[
            [
                "player_id",
                "name",
                "date_of_birth",
                "position",
                "sub_position",
                "foot",
                "height_in_cm",
                "country_of_citizenship",
                "current_club_id",
                "current_club_domestic_competition_id",
                "current_club_name",
                "market_value_in_eur",
                "highest_market_value_in_eur",
                "contract_expiration_date",
            ]
        ],
        on="player_id",
        how="left",
        suffixes=("", "_player"),
    )

    # Age and years to contract end
    snap["age"] = (snap["snapshot_date"] - snap["date_of_birth"]).dt.days / 365.25
    snap["years_to_contract_end"] = (
        (snap["contract_expiration_date"] - snap["snapshot_date"]).dt.days / 365.25
    )

    # MV ratio to peak
    snap["mv_ratio_to_peak"] = (
        snap["market_value_in_eur"] / (snap["highest_market_value_in_eur"] + 1)
    )

    print("✅ Player information added")

    # ------------------------------------------------------------------
    # 4) Season performance features (from appearances + games)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADDING SEASON PERFORMANCE FEATURES")
    print("=" * 80)

    # Unified season-year (European style)
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games["season_year"] = games["date"].apply(get_season_year_from_date)

    # Attach season_year to appearances and aggregate per player-season
    apps2 = apps.merge(
        games[["game_id", "season_year"]],
        on="game_id",
        how="left",
    )

    perf_season = (
        apps2.groupby(["player_id", "season_year"])
        .agg(
            games_played=("appearance_id", "count"),
            minutes_total=("minutes_played", "sum"),
            goals_total=("goals", "sum"),
            assists_total=("assists", "sum"),
        )
        .reset_index()
    )

    # Per-90 metrics
    perf_season["goals_per_90"] = (
        perf_season["goals_total"]
        / perf_season["minutes_total"].replace(0, np.nan)
        * 90
    )
    perf_season["assists_per_90"] = (
        perf_season["assists_total"]
        / perf_season["minutes_total"].replace(0, np.nan)
        * 90
    )
    perf_season["minutes_per_game"] = (
        perf_season["minutes_total"]
        / perf_season["games_played"].replace(0, np.nan)
    )
    # For convenience
    perf_season["minutes_per_90"] = perf_season["minutes_per_game"]

    # Clean infs
    for col in ["goals_per_90", "assists_per_90", "minutes_per_game"]:
        perf_season[col] = (
            perf_season[col]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    # Map snapshot_date to same season_year logic
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"], errors="coerce")
    snap["season_year"] = snap["snapshot_date"].apply(get_season_year_from_date)

    # Create complete player-season grid and merge perf
    all_players = snap["player_id"].unique()
    all_seasons = sorted(
        set(snap["season_year"].dropna().unique())
        | set(perf_season["season_year"].dropna().unique())
    )
    player_season_grid = pd.MultiIndex.from_product(
        [all_players, all_seasons], names=["player_id", "season_year"]
    ).to_frame(index=False)

    perf_complete = player_season_grid.merge(
        perf_season,
        on=["player_id", "season_year"],
        how="left",
    ).sort_values(["player_id", "season_year"])

    # Forward/back fill all performance columns
    perf_cols = [
        "minutes_total",
        "goals_per_90",
        "assists_per_90",
        "minutes_per_game",
        "minutes_per_90",
        "games_played",
        "goals_total",
        "assists_total",
    ]
    for col in perf_cols:
        if col in perf_complete.columns:
            perf_complete[col] = perf_complete.groupby("player_id")[col].ffill()
            perf_complete[col] = perf_complete.groupby("player_id")[col].bfill()
            perf_complete[col] = perf_complete[col].fillna(0.0)

    # Join back to snapshots
    snap = snap.merge(
        perf_complete,
        on=["player_id", "season_year"],
        how="left",
    )

    # Season deltas (growth vs previous season)
    snap = snap.sort_values(["player_id", "season_year"])
    for col in ["minutes_total", "goals_per_90", "assists_per_90",
                "minutes_per_game", "minutes_per_90"]:
        if col in snap.columns:
            snap[f"prev_{col}"] = snap.groupby("player_id")[col].shift(1).fillna(0)
            snap[f"delta_{col}"] = snap[col] - snap[f"prev_{col}"]

    print("✅ Season performance features added")

    # ------------------------------------------------------------------
    # 5) Club & league features
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADDING CLUB AND LEAGUE FEATURES")
    print("=" * 80)

    # Club info
    c = clubs[
        [
            "club_id",
            "name",
            "domestic_competition_id",
            "total_market_value",
            "squad_size",
            "average_age",
        ]
    ].rename(
        columns={
            "name": "club_name",
            "total_market_value": "club_total_market_value",
        }
    )

    # League strength = avg club MV in each competition
    league_strength_df = (
        clubs[clubs["total_market_value"].notna()]
        .groupby("domestic_competition_id")["total_market_value"]
        .mean()
        .reset_index()
        .rename(columns={"total_market_value": "league_strength"})
    )

    # Competition info (league-level)
    comp = comps[
        [
            "competition_id",
            "competition_code",
            "name",
            "country_name",
            "is_major_national_league",
        ]
    ].rename(
        columns={
            "competition_id": "domestic_competition_id",
            "name": "league_name",
            "country_name": "league_country",
        }
    )
    comp = comp.merge(league_strength_df, on="domestic_competition_id", how="left")

    # Join base club info
    snap = snap.merge(
        c,
        left_on="current_club_id",
        right_on="club_id",
        how="left",
        suffixes=("", "_club"),
    )

    # Competition ID to merge
    if "domestic_competition_id" in snap.columns:
        snap["comp_id_for_merge"] = snap["domestic_competition_id"]
    else:
        snap["comp_id_for_merge"] = np.nan

    if "current_club_domestic_competition_id" in snap.columns:
        snap["comp_id_for_merge"] = snap["comp_id_for_merge"].fillna(
            snap["current_club_domestic_competition_id"]
        )

    # If still missing, infer from club table
    if snap["comp_id_for_merge"].isna().any():
        club_comp_map = clubs[["club_id", "domestic_competition_id"]].drop_duplicates()
        snap = snap.merge(
            club_comp_map,
            left_on="current_club_id",
            right_on="club_id",
            how="left",
            suffixes=("", "_from_club"),
        )
        if "domestic_competition_id_from_club" in snap.columns:
            snap["comp_id_for_merge"] = snap["comp_id_for_merge"].fillna(
                snap["domestic_competition_id_from_club"]
            )
            snap = snap.drop(
                columns=["domestic_competition_id_from_club"],
                errors="ignore",
            )

    # Join competition info
    snap = snap.merge(
        comp,
        left_on="comp_id_for_merge",
        right_on="domestic_competition_id",
        how="left",
        suffixes=("", "_league"),
    )

    # Fill missing club_total_market_value
    if "club_total_market_value" in snap.columns:
        snap["club_total_market_value"] = snap.groupby("comp_id_for_merge")[
            "club_total_market_value"
        ].transform(lambda x: x.fillna(x.median()))
        snap["club_total_market_value"] = snap["club_total_market_value"].fillna(
            snap["league_strength"]
        )
        overall_median = c["club_total_market_value"].median()
        if pd.notna(overall_median):
            snap["club_total_market_value"] = snap["club_total_market_value"].fillna(
                overall_median
            )
        else:
            snap["club_total_market_value"] = snap["club_total_market_value"].fillna(
                50_000_000
            )

    # Fill missing league_strength
    if "league_strength" in snap.columns:
        overall_avg = league_strength_df["league_strength"].mean()
        if pd.notna(overall_avg):
            snap["league_strength"] = snap["league_strength"].fillna(overall_avg)
        else:
            snap["league_strength"] = snap["league_strength"].fillna(50_000_000)

    # League indicators
    snap["league_is_major"] = snap["is_major_national_league"].fillna(False).astype("int8")
    top5_countries = {"England", "Spain", "Germany", "Italy", "France"}
    snap["is_top5_league"] = snap["league_country"].isin(top5_countries).astype("int8")

    print("✅ Club and league features added")

    # ------------------------------------------------------------------
    # 6) Club season stats (win rate, goal diff)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADDING CLUB SEASON STATS")
    print("=" * 80)

    cg = club_games.merge(
        games[["game_id", "date", "season"]],
        on="game_id",
        how="left",
    )

    club_season_stats = (
        cg.groupby(["club_id", "season"])
        .agg(
            club_games_played=("game_id", "count"),
            club_wins=("is_win", "sum"),
            club_goals_for=("own_goals", "sum"),
            club_goals_against=("opponent_goals", "sum"),
        )
        .reset_index()
    )

    club_season_stats["club_win_rate"] = (
        club_season_stats["club_wins"]
        / club_season_stats["club_games_played"].replace(0, np.nan)
    )
    club_season_stats["club_goal_diff_per_game"] = (
        (club_season_stats["club_goals_for"] - club_season_stats["club_goals_against"])
        / club_season_stats["club_games_played"].replace(0, np.nan)
    )
    club_season_stats = club_season_stats.rename(columns={"season": "season_year"})

    # Create complete club-season grid
    all_clubs = snap["current_club_id"].dropna().unique()
    all_seasons_club = sorted(snap["season_year"].dropna().unique())
    club_season_grid = pd.MultiIndex.from_product(
        [all_clubs, all_seasons_club], names=["club_id", "season_year"]
    ).to_frame(index=False)

    club_season_complete = club_season_grid.merge(
        club_season_stats,
        on=["club_id", "season_year"],
        how="left",
    ).sort_values(["club_id", "season_year"])

    # Fill missing club-season stats
    club_stat_cols = ["club_win_rate", "club_goal_diff_per_game", "club_games_played"]
    for col in club_stat_cols:
        if col in club_season_complete.columns:
            club_season_complete[col] = club_season_complete.groupby("club_id")[col].ffill()
            club_season_complete[col] = club_season_complete.groupby("club_id")[col].bfill()

    # Fallback filling
    if "club_win_rate" in club_season_complete.columns:
        league_avg = club_season_complete.groupby("season_year")["club_win_rate"].transform(
            "mean"
        )
        club_season_complete["club_win_rate"] = club_season_complete["club_win_rate"].fillna(
            league_avg
        )
        club_season_complete["club_win_rate"] = club_season_complete["club_win_rate"].fillna(0.5)

    if "club_goal_diff_per_game" in club_season_complete.columns:
        club_season_complete["club_goal_diff_per_game"] = (
            club_season_complete["club_goal_diff_per_game"].fillna(0)
        )

    # Join back to snapshots
    snap = snap.merge(
        club_season_complete[
            ["club_id", "season_year", "club_win_rate", "club_goal_diff_per_game"]
        ],
        left_on=["current_club_id", "season_year"],
        right_on=["club_id", "season_year"],
        how="left",
        suffixes=("", "_club_season"),
    )

    print("✅ Club season stats added")

    # ------------------------------------------------------------------
    # 7) Position group, league level, mv_1y_change, perf_1y_change
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADDING REQUIRED INPUT FEATURES (SPEC)")
    print("=" * 80)

    # 1) position_group
    def map_position_group(pos):
        if pd.isna(pos):
            return np.nan
        pos = str(pos).lower()

        # Goalkeeper
        if "keeper" in pos or pos == "gk":
            return "GK"

        # Defenders
        if any(
            k in pos
            for k in [
                "back",
                "defend",
                "centre-back",
                "center-back",
                "cb",
                "fullback",
                "wing back",
                "wing-back",
            ]
        ):
            return "DF"

        # Midfielders
        if any(k in pos for k in ["midfield", "dm", "am", "cm", "lm", "rm", "mid"]):
            return "MF"

        # Everything else → forward/attacker
        return "FW"

    snap["position_group"] = snap["position"].apply(map_position_group)

    # 2) league_level (1=Top 5, 2=other major, 3=other)
    if "league_level" not in snap.columns:
        if ("is_top5_league" in snap.columns) or ("league_is_major" in snap.columns):
            snap["league_level"] = 3  # default

            if "league_is_major" in snap.columns:
                snap.loc[snap["league_is_major"] == 1, "league_level"] = 2

            if "is_top5_league" in snap.columns:
                snap.loc[snap["is_top5_league"] == 1, "league_level"] = 1

        elif "league_strength" in snap.columns:
            # Fallback using quantiles
            q_low, q_high = snap["league_strength"].quantile([0.33, 0.66]).values

            def league_level_from_strength(x):
                if pd.isna(x):
                    return 3
                if x >= q_high:
                    return 1
                if x >= q_low:
                    return 2
                return 3

            snap["league_level"] = snap["league_strength"].apply(league_level_from_strength)

    snap["league_level"] = snap["league_level"].astype("int8", errors="ignore")

    # 3) mv_1y_change (season-over-season log MV change)
    mv_year = (
        snap.groupby(["player_id", "season_year"])["market_value_in_eur"]
        .mean()
        .reset_index()
        .sort_values(["player_id", "season_year"])
    )
    mv_year["prev_mv_year"] = mv_year.groupby("player_id")["market_value_in_eur"].shift(1)
    mv_year["mv_1y_change"] = np.log(mv_year["market_value_in_eur"] + 1) - np.log(
        mv_year["prev_mv_year"] + 1
    )

    snap = snap.merge(
        mv_year[["player_id", "season_year", "mv_1y_change"]],
        on=["player_id", "season_year"],
        how="left",
    )
    snap["mv_1y_change"] = snap["mv_1y_change"].fillna(0.0)

    # 4) perf_1y_change — year-over-year performance composite
    perf_year = (
        snap.groupby(["player_id", "season_year"])
        .agg(
            goals_per_90=("goals_per_90", "mean"),
            assists_per_90=("assists_per_90", "mean"),
        )
        .reset_index()
        .sort_values(["player_id", "season_year"])
    )
    perf_year["perf_base"] = (
        0.6 * perf_year["goals_per_90"].fillna(0)
        + 0.4 * perf_year["assists_per_90"].fillna(0)
    )
    perf_year["prev_perf_base"] = perf_year.groupby("player_id")["perf_base"].shift(1)
    perf_year["perf_1y_change"] = perf_year["perf_base"] - perf_year["prev_perf_base"]

    snap = snap.merge(
        perf_year[["player_id", "season_year", "perf_1y_change"]],
        on=["player_id", "season_year"],
        how="left",
    )
    snap["perf_1y_change"] = snap["perf_1y_change"].fillna(0.0)

    print("✅ Required input features created: position_group, league_level, mv_1y_change, perf_1y_change")

    # ------------------------------------------------------------------
    # 8) Transfer features
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("ADDING TRANSFER FEATURES")
    print("=" * 80)

    transfers2 = transfers.copy()
    transfers2["transfer_date"] = pd.to_datetime(
        transfers2["transfer_date"], errors="coerce"
    )

    def season_str_to_year(s):
        if pd.isna(s):
            return np.nan
        first = int(str(s).split("/")[0])
        return 2000 + first

    transfers2["season_year"] = transfers2["transfer_season"].apply(season_str_to_year)

    # moved_to_bigger_club flag
    from_club_mv = clubs[["club_id", "total_market_value"]].rename(
        columns={"club_id": "from_club_id", "total_market_value": "from_club_mv"}
    )
    to_club_mv = clubs[["club_id", "total_market_value"]].rename(
        columns={"club_id": "to_club_id", "total_market_value": "to_club_mv"}
    )
    transfers2 = transfers2.merge(from_club_mv, on="from_club_id", how="left")
    transfers2 = transfers2.merge(to_club_mv, on="to_club_id", how="left")
    transfers2["moved_to_bigger_club"] = (
        transfers2["to_club_mv"] > transfers2["from_club_mv"]
    ).astype("int8")

    transfer_season = (
        transfers2.groupby(["player_id", "season_year"])
        .agg(
            has_recent_transfer_count=("transfer_date", "count"),
            moved_to_bigger_club_flag=("moved_to_bigger_club", "max"),
        )
        .reset_index()
    )
    transfer_season["has_recent_transfer"] = (
        transfer_season["has_recent_transfer_count"] > 0
    ).astype("int8")

    snap = snap.merge(
        transfer_season[
            ["player_id", "season_year", "has_recent_transfer", "moved_to_bigger_club_flag"]
        ],
        on=["player_id", "season_year"],
        how="left",
    )
    snap["has_recent_transfer"] = snap["has_recent_transfer"].fillna(0).astype("int8")
    snap["moved_to_bigger_club_flag"] = (
        snap["moved_to_bigger_club_flag"].fillna(0).astype("int8")
    )

    print("✅ Transfer features added")

    # ------------------------------------------------------------------
    # 9) Final cleanup & feature selection
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL CLEANUP AND SAVE")
    print("=" * 80)

    # Keep 2010+ only
    snap = snap[snap["snapshot_date"].dt.year >= 2010].copy()

    # Fill critical nulls
    if "position" in snap.columns:
        snap["age"] = snap.groupby("position")["age"].transform(
            lambda x: x.fillna(x.median() if x.median() > 0 else 25)
        )
    snap["age"] = snap["age"].fillna(25)

    if "height_in_cm" in snap.columns:
        if "position" in snap.columns:
            snap["height_in_cm"] = snap.groupby("position")["height_in_cm"].transform(
                lambda x: x.fillna(x.median())
            )
        snap["height_in_cm"] = snap["height_in_cm"].fillna(180)

    snap["years_to_contract_end"] = snap["years_to_contract_end"].fillna(5)

    perf_cols_final = [
        "minutes_total",
        "goals_per_90",
        "assists_per_90",
        "minutes_per_game",
        "games_played",
    ]
    for col in perf_cols_final:
        if col in snap.columns:
            snap[col] = snap[col].fillna(0)

    # Delta columns (include delta_minutes_per_90 here)
    delta_cols = [
        "delta_minutes_total",
        "delta_minutes_per_90",
        "delta_goals_per_90",
        "delta_assists_per_90",
    ]
    for col in delta_cols:
        if col in snap.columns:
            snap[col] = snap[col].fillna(0)

    snap["club_win_rate"] = snap["club_win_rate"].fillna(0.5)
    snap["club_goal_diff_per_game"] = snap["club_goal_diff_per_game"].fillna(0)

    if "foot" in snap.columns:
        snap["foot"] = snap["foot"].fillna("right")

    # Core columns for modeling
    core_cols = [
        # Keys
        "player_id",
        "snapshot_date",
        "season_year",
        "name",
        "age",
        "position",
        "position_group",
        "sub_position",
        "foot",
        "height_in_cm",
        "country_of_citizenship",
        # Market
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "mv_ratio_to_peak",
        "y_growth",
        "future_market_value",
        "years_to_contract_end",
        "mv_1y_change",
        # Performance level
        "minutes_total",
        "games_played",
        "minutes_per_game",
        "minutes_per_90",
        "goals_total",
        "assists_total",
        "goals_per_90",
        "assists_per_90",
        # Performance growth / momentum
        "delta_minutes_total",
        "delta_minutes_per_90",
        "delta_goals_per_90",
        "delta_assists_per_90",
        "perf_1y_change",
        # Club & league
        "current_club_id",
        "club_name",
        "club_total_market_value",
        "club_win_rate",
        "club_goal_diff_per_game",
        "league_name",
        "league_country",
        "league_strength",
        "league_is_major",
        "is_top5_league",
        "league_level",
        # Transfer
        "has_recent_transfer",
        "moved_to_bigger_club_flag",
    ]

    # Keep only columns that exist
    core_cols = [c for c in core_cols if c in snap.columns]
    player_snapshot = snap[core_cols].copy()

    print("✅ Final cleanup complete")
    print(
        f"\nFinal snapshot: {len(player_snapshot):,} rows × {len(player_snapshot.columns)} columns"
    )

    # ------------------------------------------------------------------
    # 10) Data quality checks & save
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("RUNNING DATA QUALITY CHECKS")
    print("=" * 80)

    _ = check_nulls(player_snapshot, "Final Player Snapshot")
    _ = test_modeling_readiness(player_snapshot, "Final Player Snapshot")

    # Save
    player_snapshot.to_parquet(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")

    print("\n" + "=" * 80)
    print("DATA PREVIEW")
    print("=" * 80)
    print(player_snapshot.head())

    return player_snapshot


if __name__ == "__main__":
    # Simple CLI: allow overriding data dir via env var if needed
    data_dir = os.environ.get("PLAYER_DATA_DIR", DATA_DIR)
    out_path = os.path.join(PROCESSED_DIR, "player_snapshot.parquet")
    build_player_snapshot(data_dir=data_dir, output_path=out_path)
