import time
import pandas as pd
import math
import numpy as np
from typing import List
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from firebase_client import FirebaseClient

@dataclass
class BattleResult:
    model_a: str
    model_b: str
    winner: str
    userId: str
    interval: int
    timestamp: int
    modelWinner: str
    username: str


def get_battle_df(outcomes_df, incl_models, interval_size):
    # Sort outcomes_df by timestamp from oldest to newest
    outcomes_df = outcomes_df.sort_values(by=["timestamp"])
    outcomes_df = outcomes_df[outcomes_df['acceptedIndex'] != -1]
    outcomes_df = outcomes_df[outcomes_df['acceptedIndex'] != '']

    outcomes_df = (
        outcomes_df.assign(
            winner=lambda df: df["acceptedIndex"].map({0: "model_a", 1: "model_b"}),
            model_a=lambda df: df["completionItems"].str[0].str["model"],
            model_b=lambda df: df["completionItems"].str[1].str["model"],
            completion_id_a=lambda df: df["completionItems"].str[0].str["completionId"],
            completion_id_b=lambda df: df["completionItems"].str[1].str["completionId"],
            model_winner=lambda df: df.apply(lambda row: row["completionItems"][int(row["acceptedIndex"])]["model"], axis=1),
            username=lambda df: df['completionItems'].str[0].str["username"]
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
            modelWinner=row["model_winner"],
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


def get_coefficients(df, lam, model_list, base=10):

    ptbl_a_win = pd.pivot_table(
        df[df["winner"] == "model_a"],
        index="model_a",
        columns="model_b",
        aggfunc="size",
        fill_value=0,
    )

    ptbl_a_win = ptbl_a_win.reindex(index=model_list, columns=model_list, fill_value=0)

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

    ptbl_b_win = ptbl_b_win.reindex(index=model_list, columns=model_list, fill_value=0)

    ptbl_win = ptbl_a_win * 2 + ptbl_b_win.T * 2 + ptbl_tie

    ptbl_win = ptbl_win.fillna(
        0
    )

    # Create a mapping from model name to its index
    models = pd.Series(np.arange(len(model_list)), index=model_list)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in model_list:
        for m_b in model_list:
            if m_a == m_b:
                continue

            if m_a not in ptbl_win.columns or m_b not in ptbl_win.index:
                continue

            if not math.isnan(ptbl_win.loc[m_a, m_b]):
                X[cur_row, models[m_a]] = +math.log(base)
                X[cur_row, models[m_b]] = -math.log(base)
                Y[cur_row] = 1.0
                sample_weights.append(ptbl_win.loc[m_a, m_b] * lam / len(df))
                cur_row += 1

            if m_a not in ptbl_win.columns or m_b not in ptbl_win.index:
                continue

            if not math.isnan(ptbl_win.loc[m_b, m_a]):
                X[cur_row, models[m_a]] = math.log(base)
                X[cur_row, models[m_b]] = -math.log(base)
                Y[cur_row] = 0.0
                sample_weights.append(ptbl_win.loc[m_b, m_a] * lam / len(df))
                cur_row += 1
    X = X[:cur_row]
    Y = Y[:cur_row]

    return X, Y, sample_weights, models


def compute_mle_elo(
    global_df, user_df, models, scale=400, base=10, init_rating=1000, sample_weight=None
):
    def get_loss_for_lambda(lam, train_user_df, test_user_rows):
        global_X, global_Y, global_sample_weights, global_models = get_coefficients(
            global_df, lam, models
        )
        user_X, user_Y, user_sample_weights, user_models = get_coefficients(
            train_user_df, 1 - lam, models
        )

        # Combine global and user data
        X = np.concatenate((global_X, user_X), axis=0)
        Y = np.concatenate((global_Y, user_Y), axis=0)
        sample_weights = np.concatenate(
            (global_sample_weights, user_sample_weights), axis=0
        )

        # Fit the model
        lr = LogisticRegression(fit_intercept=False, penalty="l2", C=1.0, tol=1e-6)
        lr.fit(X, Y, sample_weight=sample_weights)

        # Prepare test data
        test_X = np.zeros((len(test_user_rows), len(global_models)))
        test_Y = np.zeros(len(test_user_rows))

        for idx, (_, row) in enumerate(test_user_rows.iterrows()):
            winner = row["winner"]
            model_a = row["model_a"]
            model_b = row["model_b"]

            if winner == "model_a":
                test_X[idx, global_models[model_a]] = math.log(base)
                test_X[idx, global_models[model_b]] = -math.log(base)
                test_Y[idx] = 1.0
            elif winner == "model_b":
                test_X[idx, global_models[model_a]] = -math.log(base)
                test_X[idx, global_models[model_b]] = math.log(base)
                test_Y[idx] = 0.0
            else:
                # Handle tie cases
                test_X[idx, global_models[model_a]] = 0.0
                test_X[idx, global_models[model_b]] = 0.0
                test_Y[idx] = 0.5  # Represents a tie

        # Predict probabilities
        y_pred = lr.predict_proba(test_X)
        return log_loss(test_Y, y_pred, labels=[0, 1])

    def get_avg_loss_for_user_data(user_df, lam):
        losses = []
        test_set_size = 10
        for i in range(0, len(user_df), test_set_size):
            test_rows = user_df.iloc[i : i + 5]
            train_user_df = user_df.drop(user_df.index[i : i + 5])
            loss = get_loss_for_lambda(lam, train_user_df, test_rows)
            losses.append(loss)

        avg_loss = sum(losses) / len(losses)
        # Set decimls of avg loss to 4
        avg_loss = round(avg_loss, 4)
        return avg_loss

    best_lambda = 0
    best_loss = get_avg_loss_for_user_data(user_df, best_lambda)
    best_lambda = 0.5
    lower_bound = 0
    upper_bound = 1
    steps = 5
    for _ in range(steps):
        upper_lambda = (upper_bound + best_lambda) / 2.0
        lower_lambda = (lower_bound + best_lambda) / 2.0
        upper_loss = get_avg_loss_for_user_data(user_df, upper_lambda)
        lower_loss = get_avg_loss_for_user_data(user_df, lower_lambda)
        # print(f"Upper Lambda {upper_lambda:.5f}: {upper_loss}")
        # print(f"Lower Lambda {lower_lambda:.5f}: {lower_loss}")
        if best_loss <= lower_loss and best_loss <= upper_loss:
            break
        elif lower_loss < upper_loss:
            upper_bound = best_lambda
            best_lambda = lower_lambda
        elif lower_loss > upper_loss:
            lower_bound = best_lambda
            best_lambda = upper_lambda
        else:
            break

    # print(f"Lambda {best_lambda:.5f}: {loss}")

    # global user_lambdas
    # if best_lambda not in user_lambdas:
    #     user_lambdas[best_lambda] = 0
    # user_lambdas[best_lambda] += 1

    global_X, global_Y, global_sample_weights, global_models = get_coefficients(
        global_df, best_lambda, models
    )
    user_X, user_Y, user_sample_weights, user_models = get_coefficients(
        user_df, 1 - best_lambda, models
    )

    # combine global and user
    X = np.concatenate((global_X, user_X), axis=0)
    Y = np.concatenate((global_Y, user_Y), axis=0)
    sample_weights = np.concatenate(
        (global_sample_weights, user_sample_weights), axis=0
    )

    # Create a mapping from model name to its index
    lr = LogisticRegression(fit_intercept=False, penalty="l2", C=1.0, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = scale * lr.coef_[0] + init_rating
    return pd.Series(elo_scores, index=global_models.index).sort_values(ascending=False)


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
    full_outcomes_df: pd.DataFrame,
    global_outcomes_df: pd.DataFrame,
    user_outcomes_df: pd.DataFrame,
    models: List,
    interval_size=20,
):
    # Check if user_outcomes_df is empty
    if len(user_outcomes_df) < 20:
        return []

    global_battles = get_battle_df(
        global_outcomes_df, incl_models=models, interval_size=interval_size
    )
    user_battles = get_battle_df(
        user_outcomes_df, incl_models=models, interval_size=interval_size
    )
    full_battles = get_battle_df(
        full_outcomes_df, incl_models=models, interval_size=interval_size
    )

    if len(user_battles["winner"].unique()) == 1:
        return []

    global_battles = global_battles.sort_values(by=["interval"])
    user_battles = user_battles.sort_values(by=["interval"])

    def get_bootstrap_result(func_compute_elo, num_round):
        rows = []
        for i in range(num_round):
            rows.append(func_compute_elo(global_battles.sample(frac=1.0, replace=True),
                                        user_battles.sample(frac=1.0, replace=True), models))
        df = pd.DataFrame(rows)
        return df[df.median().sort_values(ascending=False).index]


    BOOTSTRAP_ROUNDS = 50

    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(compute_mle_elo, BOOTSTRAP_ROUNDS)

    bars = pd.DataFrame(dict(
            lower = round(bootstrap_elo_lu.quantile(.025), 2),
            rating = round(bootstrap_elo_lu.quantile(.5), 2),
            upper = round(bootstrap_elo_lu.quantile(.975), 2))).reset_index(names="model").sort_values("rating", ascending=False)

    vote_data = get_vote_data(full_battles)

    user_data = get_user_data(full_battles)
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
    models = ['gpt-4o-mini-2024-07-18', 'claude-3-haiku-20240307', 'llama-3.1-70b-instruct',
               'llama-3.1-405b-instruct', 'codestral-2405', 'deepseek-coder-fim', 'gemini-1.5-flash-001',
               'gemini-1.5-pro-001', 'gemini-1.5-flash-002', 'gemini-1.5-pro-002', 'claude-3-5-sonnet-20240620',
               'gpt-4o-2024-08-06', 'gemini-1.5-pro-exp-0827', 'gemini-1.5-flash-exp-0827']

    version_num = "v1" 
    ################################################################################################
    autocomplete_outcomes_collection_name = "autocomplete_outcomes_" + str(version_num)
    start_time = time.time()
    global_outcomes_df = firebase_client.get_autocomplete_outcomes(
        autocomplete_outcomes_collection_name
    )

    user_ids = global_outcomes_df["userId"].unique()
    user_ids = [user_id for user_id in user_ids]

    success = 0
    for user_id in user_ids:
        user_outcomes_df = global_outcomes_df[global_outcomes_df["userId"] == user_id]
        temp_global_outcomes_df = global_outcomes_df[
            global_outcomes_df["userId"] != user_id
        ]
        if len(user_outcomes_df) < 20:
            continue

        start_time = time.time()
        user_data, elo_data = get_scores(
            global_outcomes_df, temp_global_outcomes_df, user_outcomes_df, models=models
        )
        end_time = time.time()
        print(f"Time taken for get_scores: {end_time - start_time:.2f} seconds")

        if len(user_data) > 0 and len(elo_data) > 0:
            user_data.to_csv('backend/user_leaderboard.csv', index=False)
            elo_data.to_csv('backend/leaderboard.csv', index=False)
            success += 1
            break