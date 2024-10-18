import time
import pandas as pd
import math
import numpy as np
from firebase_client import FirebaseClient
from typing import List
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression


# based on https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=C5H_wlbqGwCJ


@dataclass
class BattleResult:
    model_a: str
    model_b: str
    winner: str
    userId: str
    interval: int
    timestamp: int
    username: str


def get_battle_df(outcomes_df, incl_models, interval_size):
    # Sort outcomes_df by timestamp from oldest to newest
    outcomes_df = outcomes_df.sort_values(by=["timestamp"])

    outcomes_df = (
        outcomes_df.assign(
            winner=lambda df: df["acceptedIndex"].map({0: "model_a", 1: "model_b"}),
            model_a=lambda df: df["completionItems"].str[0].str["model"],
            model_b=lambda df: df["completionItems"].str[1].str["model"],
            completion_id_a=lambda df: df["completionItems"].str[0].str["completionId"],
            completion_id_b=lambda df: df["completionItems"].str[1].str["completionId"],
            username=lambda df: df['completionItems'].str[0].str["username"],
        )
        .loc[
            lambda df: df["model_a"].isin(incl_models) & df["model_b"].isin(incl_models)
        ]
        .sort_values(by=["timestamp"])
        .assign(interval=lambda df: (np.arange(len(df)) // interval_size).astype(int))
    )

    if outcomes_df.empty:
        return pd.DataFrame()

    battle_results = [
        BattleResult(
            model_a=row["model_a"],
            model_b=row["model_b"],
            winner=row["winner"],
            userId=row["userId"],
            interval=row["interval"],
            timestamp=row["timestamp"],
            username=row["username"]
        )
        for _, row in outcomes_df.iterrows()
    ]

    # Create DataFrame
    battles = pd.DataFrame([vars(result) for result in battle_results])

    return battles

def get_vote_data(battles):
    votes = pd.concat([battles['model_a'], battles['model_b']]).value_counts().reset_index()
    votes.columns = ['model', 'votes']
    votes['votes'] = votes['votes'].astype(int)
    return votes

def get_user_data(battles):
    user_counts = battles['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'count']
    user_counts['count'] = user_counts['count'].astype(int)
    return user_counts

def process_completions(completions_df, timestamp_to_interval, max_outcome_timestamp):
    completions_df = completions_df.sort_values(by=["timestamp"])
    pairs = {}
    completion_data = []

    for _, row in completions_df.iterrows():
        key = row["completionId"] if row["pairIndex"] == 0 else row["pairCompletionId"]
        if key not in pairs:
            pairs[key] = [(row["model"], row["userId"], row["timestamp"])]
        else:
            insert_index = 0 if row["pairIndex"] == 0 else len(pairs[key])
            pairs[key].insert(
                insert_index,
                (row["model"], row["userId"], row["timestamp"]),
            )

    for key, value in pairs.items():
        if len(value) == 2:
            timestamp = max(
                value[0][2], value[1][2]
            )  # Use the later timestamp of the pair
            interval = assign_interval(
                timestamp, timestamp_to_interval, max_outcome_timestamp
            )
            completion_data.append(
                {
                    "model_a": value[0][0],
                    "model_b": value[1][0],
                    "winner": "tie (bothbad)",
                    "userId": value[0][1],
                    "interval": interval,
                    "timestamp": timestamp,
                }
            )

    return completion_data


def assign_interval(timestamp, timestamp_to_interval, max_outcome_timestamp):
    if timestamp > max_outcome_timestamp:
        return max(
            timestamp_to_interval.values()
        )  # Assign to the last interval if after all outcomes
    for outcome_timestamp, interval in sorted(timestamp_to_interval.items()):
        if timestamp <= outcome_timestamp:
            return interval
    return max(timestamp_to_interval.values())  # Fallback to last interval


def compute_mle_elo(df, scale=400, base=10, init_rating=1000, sample_weight=None):
    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    # if no tie, create a zero matrix
    if sum(df["winner"].isin(["tie", "tie (bothbad)"])) == 0:
        ptbl_tie = pd.DataFrame(0, index=ptbl_a_win.index, columns=ptbl_a_win.columns)
    else:
        ptbl_tie = pd.pivot_table(
            df[df["winner"].isin(["tie", "tie (bothbad)"])],
            index="model_a",
            columns="model_b",
            aggfunc="size",
            fill_value=0,
        )
        ptbl_tie = ptbl_tie + ptbl_tie.T

    ptbl_b_win = pd.pivot_table(
        df[df["winner"] == "model_b"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )
    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    # Convert nans to zeros
    ptbl_win = ptbl_win.fillna(0)

    # Create a mapping from model name to its index
    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            if m_a not in models or m_b not in models:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(base)
            X[cur_row, models[m_b]] = -math.log(base)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(base)
            X[cur_row + 1, models[m_b]] = -math.log(base)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty="l2", C=1.0, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_interval_group(battles_df, start_interval, end_interval):
    """
    Retrieve battles for a specific group of intervals.

    :param battles_df: DataFrame returned by get_battle_dfs
    :param start_interval: Starting interval (inclusive)
    :param end_interval: Ending interval (inclusive)
    :return: DataFrame with battles from the specified interval range
    """
    return battles_df[
        (battles_df["interval"] >= start_interval)
        & (battles_df["interval"] <= end_interval)
    ]


def get_interval_group_results(battles_df, start_interval, end_interval):
    """
    Get aggregated results for a group of intervals.

    :param battles_df: DataFrame returned by get_battle_dfs
    :param start_interval: Starting interval (inclusive)
    :param end_interval: Ending interval (inclusive)
    :return: Dictionary with aggregated results
    """
    group_df = get_interval_group(battles_df, start_interval, end_interval)

    results = {
        "total_battles": len(group_df),
        "winner_counts": group_df["winner"].value_counts().to_dict(),
        "model_a_counts": group_df["model_a"].value_counts().to_dict(),
        "model_b_counts": group_df["model_b"].value_counts().to_dict(),
        "unique_users": group_df["userId"].nunique(),
    }

    return results


def get_scores(
    outcomes_df: pd.DataFrame,
    models: List,
    interval_size=20,
):
    # Check if outcomes_df is empty
    battles = get_battle_df(
        outcomes_df, incl_models=models, interval_size=interval_size
    )
    if battles.empty:
        return []

    battles = battles.sort_values(by=["interval"])
    max_interval = battles["interval"].max()

    def get_bootstrap_result(func_compute_elo, num_round):
        rows = []
        for i in range(num_round):
            interval_group = get_interval_group(battles, 0, max_interval)
            rows.append(func_compute_elo(interval_group.sample(frac=1.0, replace=True)))
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]

    BOOTSTRAP_ROUNDS = 50

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(compute_mle_elo, BOOTSTRAP_ROUNDS)

    bars = pd.DataFrame(dict(
            lower = round(bootstrap_elo_lu.quantile(.025), 2),
            rating = round(bootstrap_elo_lu.quantile(.5), 2),
            upper = round(bootstrap_elo_lu.quantile(.975), 2))).reset_index(names="model").sort_values("rating", ascending=False)

    vote_data = get_vote_data(battles)

    user_data = get_user_data(battles)
    user_df = user_data[user_data['username'].str.len() > 0]

    models_df = pd.read_csv('backend/leaderboard_data.csv')
    quartiles_df = models_df.merge(bars, on='model', how='left')
    elo_df = quartiles_df.merge(vote_data, on="model", how="left")
    elo_df = elo_df.dropna(axis=0, how='any')
    elo_df.rename(columns={'rating': 'score'}, inplace=True)

    return user_df, elo_df


if __name__ == "__main__":
    # Test out firebase retrieval
    firebase_client = FirebaseClient()
    models = ['gpt-4o-mini-2024-07-18', 'llama-3.1-70b-instruct',
               'llama-3.1-405b-instruct', 'codestral-2405', 'deepseek-coder-fim', 'gemini-1.5-flash-001',
               'gemini-1.5-pro-001', 'gemini-1.5-flash-002', 'gemini-1.5-pro-002', 'claude-3-5-sonnet-20240620',
               'gpt-4o-2024-08-06', 'gemini-1.5-pro-exp-0827', 'gemini-1.5-flash-exp-0827']
    models = ['gpt-4o-mini-2024-07-18', 'llama-3.1-70b-instruct', 'llama-3.1-405b-instruct',
              'codestral-2405', 'deepseek-coder-fim', 'gemini-1.5-flash-002', 'gemini-1.5-pro-002',
              'claude-3-5-sonnet-20240620', 'gpt-4o-2024-08-06']

    ###Replace this with however you get autocomplete_outcomes and autocomplete_compeltions###
    version_nums = ["v1", "5"]  # Multiple version numbers
    ################################################################################################
    outcomes_dfs = []
    start_time = time.time()
    for version_num in version_nums:
        autocomplete_outcomes_collection_name = f"autocomplete_outcomes_{version_num}"
        outcomes_df = firebase_client.get_autocomplete_outcomes(
            autocomplete_outcomes_collection_name  # , user_id
        )
        outcomes_dfs.append(outcomes_df)

    # Combine all outcomes dataframes
    outcomes_df = pd.concat(outcomes_dfs, ignore_index=True)

    end_time = time.time()
    print(f"Time taken to retrieve data: {end_time - start_time:.2f} seconds")

    user_data, elo_data = get_scores(outcomes_df, models=models)
    user_data.to_csv('backend/user_leaderboard.csv', index=False)
    elo_data.to_csv('backend/leaderboard.csv', index=False)