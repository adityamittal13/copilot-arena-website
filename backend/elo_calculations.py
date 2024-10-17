import os
import pandas as pd
import math
import numpy as np
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "servicekey_admin.json"
import firebase_admin
from firebase_admin import firestore
from google.api_core.retry import Retry


def get_battle_df(outcomes_df, incl_models):
    outcomes_df = outcomes_df[outcomes_df['acceptedIndex'] != -1]

    model_a = []
    model_b = []
    winner = []
    model_winner = []
    user_ids = []

    for _, row in outcomes_df.iterrows():
        try:
            accepted_idx = int(row['acceptedIndex'])
        except ValueError: # no index was accepted
            continue

        model_a_name = row['completionItems'][0]['model']
        model_b_name = row['completionItems'][1]['model']
        model_winner_name = row['completionItems'][accepted_idx]['model']

        winner.append('model_a' if accepted_idx == 0 else "model_b")
        model_a.append(model_a_name)
        model_b.append(model_b_name)
        user_ids.append(row['completionItems'][0].get('username', ""))    
        model_winner.append(model_winner_name)

    battles = pd.DataFrame({'model_a': model_a, 'model_b': model_b, 'winner': winner, 'model_winner': model_winner, 'username':user_ids})
    
    #if model_a or model_b in this list then we want to exclude
    battles = battles[battles['model_a'].isin(incl_models)]
    battles = battles[battles['model_b'].isin(incl_models)]
    battles = battles[battles["username"] != ""]

    return battles

def get_win_data(battles):
    wins = battles['model_winner'].value_counts().reset_index()
    wins.columns = ['model', 'wins']
    wins['wins'] = wins['wins'].astype(int)
    return wins

def get_user_data(battles):
    user_counts = battles['username'].value_counts().reset_index()
    user_counts.columns = ['username', 'count']
    user_counts['count'] = user_counts['count'].astype(int)
    return user_counts

def compute_mle_elo( 
    df, SCALE=400, BASE=10, INIT_RATING=1000, sample_weight=None
):
    from sklearn.linear_model import LogisticRegression
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
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING
    if "mixtral-8x7b-instruct-v0.1" in models.index:
        elo_scores += 1114 - elo_scores[models["mixtral-8x7b-instruct-v0.1"]]
    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)

###Replace this with however you get autocomplete_outcomes and autocomplete_compeltions###
version_num = "v1"
app = firebase_admin.initialize_app()
db = firestore.client()
docs = db.collection(f'autocomplete_outcomes_{version_num}').get(retry=Retry(timeout=120))
outcomes_df = pd.DataFrame([x.to_dict() for x in docs])
################################################################################################
incl_models = ['gpt-4o-mini-2024-07-18', 'claude-3-haiku-20240307', 'llama-3.1-70b-instruct',
               'llama-3.1-405b-instruct', 'codestral-2405', 'deepseek-coder-fim', 'gemini-1.5-flash-001',
               'gemini-1.5-pro-001', 'gemini-1.5-flash-002', 'gemini-1.5-pro-002', 'claude-3-5-sonnet-20240620',
               'gpt-4o-2024-08-06', 'gemini-1.5-pro-exp-0827', 'gemini-1.5-flash-exp-0827']

remove_users = ['d0fdbdd90881f84353451cf61410db0fc10cd31010d6764896ab2423f56035bd',
 '04a825412bd523e6d1d1fba9b5aa7651afbf3c105727cf218d404652bee779bd',
 'd5a3f022323c294dd71cb955d07b30ebe7cd02c79e37b50c086a1791783777e9',
 'f7c51699ea0e957e7d4ba2d1cc5b72b0e694ffaa8a00b7d8fe0dfb69611678b9', 'test',
 'd779845fba123029730570befe88e0281ee9532640c43cab0659a5f77686c88e',
 '329adf05f3fe87f0e73ea367fdceae4e18bbab1f8ad7a2816123cc623a168a5b']

#remove remove_users from outcomes_df
outcomes_df = outcomes_df[~outcomes_df['userId'].isin(remove_users)]

battles = get_battle_df(outcomes_df, incl_models)
win_data = get_win_data(battles)
elo_ratings = compute_mle_elo(battles)
user_data = get_user_data(battles)
user_data.to_csv('backend/user_leaderboard.csv', index=False)

def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in range(num_round):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


BOOTSTRAP_ROUNDS = 100

np.random.seed(42)
bootstrap_elo_lu = get_bootstrap_result(battles, compute_mle_elo, BOOTSTRAP_ROUNDS)

bars = pd.DataFrame(dict(
        lower = bootstrap_elo_lu.quantile(.025),
        rating = bootstrap_elo_lu.quantile(.5),
        upper = bootstrap_elo_lu.quantile(.975))).reset_index(names="model").sort_values("rating", ascending=False)

models_df = pd.read_csv('backend/leaderboard_data.csv')
merged_df = models_df.merge(bars, on='model', how='left')
merged_df = merged_df.merge(win_data, on="model", how="left")
merged_df = merged_df.dropna(axis=0, how='any')
merged_df.rename(columns={'rating': 'score'}, inplace=True)
merged_df.to_csv('backend/leaderboard.csv', index=False)