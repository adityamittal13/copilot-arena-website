import time
import pandas as pd
import math
import numpy as np
from firebase_client import FirebaseClient
from typing import List
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
import json


# based on https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=C5H_wlbqGwCJ


@dataclass
class BattleResult:
    model_a: str
    model_b: str
    winner: str
    userId: str
    timestamp: int
    username: str


def get_battle_df(outcomes_df, incl_models, is_edit):
    # Sort outcomes_df by timestamp from oldest to newest
    outcomes_df = outcomes_df.sort_values(by=["timestamp"])

    completion_items_name = "responseItems" if is_edit else "completionItems"
    completion_id_name = "responseId" if is_edit else "completionId"

    if is_edit:
        outcomes_df = (
            outcomes_df.assign(
                winner=lambda df: df["acceptedIndex"].map({0: "model_a", 1: "model_b", -2: "tie (bothbad)"}),
                model_a=lambda df: df[completion_items_name].str[0].str["model"],
                model_b=lambda df: df[completion_items_name].str[1].str["model"],
                completion_id_a=lambda df: df[completion_items_name].str[0].str[completion_id_name],
                completion_id_b=lambda df: df[completion_items_name].str[1].str[completion_id_name],
                username=lambda df: df[completion_items_name].str[0].str["username"],
                completion = lambda df: df[completion_items_name].str[0].map(lambda x: x.get("completion", ""))
            )
            .loc[
                lambda df: df["model_a"].isin(incl_models) & df["model_b"].isin(incl_models) & ~df["completion"].str.startswith("```")
            ]
            .sort_values(by=["timestamp"])
        )
    else:
        outcomes_df = (
            outcomes_df.assign(
                winner=lambda df: df["acceptedIndex"].map({0: "model_a", 1: "model_b"}),
                model_a=lambda df: df[completion_items_name].str[0].str["model"],
                model_b=lambda df: df[completion_items_name].str[1].str["model"],
                completion_id_a=lambda df: df[completion_items_name].str[0].str[completion_id_name],
                completion_id_b=lambda df: df[completion_items_name].str[1].str[completion_id_name],
                username=lambda df: df[completion_items_name].str[0].str["username"],
                completion = lambda df: df[completion_items_name].str[0].map(lambda x: x.get("completion", ""))
            )
            .loc[
                lambda df: df["model_a"].isin(incl_models) & df["model_b"].isin(incl_models) & ~df["completion"].str.startswith("```")
            ]
            .sort_values(by=["timestamp"])
        )

    if outcomes_df.empty:
        return pd.DataFrame()

    battle_results = [
        BattleResult(
            model_a=row["model_a"],
            model_b=row["model_b"],
            winner=row["winner"],
            userId=row["userId"],
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

def compute_mle_elo(df, scale=400, base=10, init_rating=1000, ANCHOR=None):
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
            if m_a not in ptbl_win.columns or m_b not in ptbl_win.index:
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

    lr = LogisticRegression(fit_intercept=False, penalty="l2", tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    if ANCHOR == None:
        scaler = 0
    else:
        scaler = ANCHOR

    elo_scores = scale * lr.coef_[0] + init_rating + scaler
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

def get_scores(
    outcomes_df: pd.DataFrame,
    models: List,
    is_edit: bool = False,
):
    # Check if outcomes_df is empty
    battles = get_battle_df(
        outcomes_df, incl_models=models, is_edit=is_edit
    )
    if battles.empty:
        return []

    def get_bootstrap_result(func_compute_elo, num_round, anchor_delta):
        rows = []
        for i in range(num_round):
            rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), ANCHOR=anchor_delta))
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]

    BOOTSTRAP_ROUNDS = 100

    elo_ratings = compute_mle_elo(battles)
    if is_edit:
        anchor_delta = elo_ratings[elo_ratings.index == "gemini-1.5-pro-002"].values[0]-1000 #change this model later
    else:
        anchor_delta = elo_ratings[elo_ratings.index == "codestral-2405"].values[0]-1000

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(compute_mle_elo, BOOTSTRAP_ROUNDS, anchor_delta)

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

    return user_df, elo_df, battles['userId'].nunique()


if __name__ == "__main__":
    # Test out firebase retrieval
    firebase_client = FirebaseClient()
    models = ['gpt-4o-mini-2024-07-18', 'llama-3.1-70b-instruct', 'llama-3.1-405b-instruct',
              'codestral-2405', 'deepseek-coder', 'deepseek-coder-fim', 'gemini-1.5-flash-002', 'gemini-1.5-pro-002',
              'claude-3-5-sonnet-20240620', 'gpt-4o-2024-08-06', 'gpt-4o-2024-11-20', 'qwen-2.5-coder-32b-instruct', "claude-3-5-sonnet-20241022"]
    edit_models = ['gpt-4o-mini-2024-07-18', 'llama-3.1-405b-instruct', 'gemini-1.5-pro-002', 
                   'gpt-4o-2024-08-06', 'qwen-2.5-coder-32b-instruct', "claude-3-5-sonnet-20241022"]

    ###Replace this with however you get autocomplete_outcomes and autocomplete_compeltions###
    version_nums = ["v1", "5"]  # Multiple version numbers
    edit_version_num = "v1"
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
    edit_df = firebase_client.get_autocomplete_outcomes(f"edit_outcomes_{edit_version_num}")

    end_time = time.time()
    print(f"Time taken to retrieve data: {end_time - start_time:.2f} seconds")

    user_data, elo_data, num_users = get_scores(outcomes_df, models=models)
    # elo_data = elo_data.sort_values(by="score", ascending=False)
    # print(elo_data)
    _, edit_elo_data, _ = get_scores(edit_df, models=edit_models, is_edit=True)
    leaderboard_data_json = {
        "user_data": user_data.to_dict('records'),
        "elo_data": elo_data.to_dict('records'),
        "edit_elo_data": edit_elo_data.to_dict('records'),
        "num_users": num_users
    }
    with open("backend/leaderboard.json", "w") as json_file:
        json.dump(leaderboard_data_json, json_file, indent=4)
