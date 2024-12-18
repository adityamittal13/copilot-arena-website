import glob
import gradio as gr
import numpy as np
import pandas as pd
import argparse
import json


stylesheet = """
<style>
    .column {
        padding: 20px;
    }

    #arena_user_leaderboard table {
        max-height: 600px;
        overflow-y: auto;
    }

    #arena_hard_leaderboard table {
        max-height: 600px;
        overflow-y: auto;
    }
</style>
"""

def recompute_final_ranking(arena_df):
    q025 = arena_df["rating_q025"].values
    q975 = arena_df["rating_q975"].values

    sorted_q025 = np.sort(q025)
    insertion_indices = np.searchsorted(sorted_q025, q975, side="right")
    counts = len(sorted_q025) - insertion_indices

    rankings = 1 + counts
    ranking_series = pd.Series(rankings, index=arena_df.index)
    return ranking_series.tolist()

def process_leaderboard(leaderboard):
    # Round scores to integers
    leaderboard['score'] = leaderboard['score'].round().astype(int)
    leaderboard["rating_q975"] = leaderboard["upper"].round().astype(int)
    leaderboard["rating_q025"] = leaderboard["lower"].round().astype(int)

    leaderboard["upper_diff"] = leaderboard["rating_q975"] - leaderboard["score"]
    leaderboard["lower_diff"] = leaderboard["score"] - leaderboard["rating_q025"]

    # Combine the differences into a single column with +/- format
    leaderboard['confidence_interval'] = '+' + leaderboard['upper_diff'].astype(str) + ' / -' + leaderboard['lower_diff'].astype(str)

    rankings_ub = recompute_final_ranking(leaderboard)
    leaderboard.insert(loc=0, column="Rank* (UB)", value=rankings_ub)
    leaderboard['Rank'] = leaderboard['score'].rank(ascending=False).astype(int)

    # Sort the leaderboard by rank and then by score
    leaderboard = leaderboard.sort_values(by=['Rank'], ascending=[True])
    
    return leaderboard

def process_user_leaderboard(leaderboard):
    # Sort the leaderboard by votes
    leaderboard = leaderboard.sort_values(by=['count'], ascending=[False])

    return leaderboard, len(leaderboard)

def build_leaderboard(leaderboard_json):
    link_color = "#1976D2"  # noqa: F841
    default_md = "Welcome to Copilot Arena leaderboard 🏆"

    with open(leaderboard_json, "r") as f:
        leaderboard_json = json.load(f)

    leaderboard = pd.DataFrame(leaderboard_json["elo_data"])
    user_leaderboard = pd.DataFrame(leaderboard_json['user_data'])
    edit_leaderboard = pd.DataFrame(leaderboard_json["edit_elo_data"])
    num_users = leaderboard_json["num_users"]

    with gr.Row():
        with gr.Column(scale=4):
            md_1 = gr.Markdown(default_md, elem_id="leaderboard_markdown")  # noqa: F841
    with gr.Tabs() as tabs:
        with gr.Tab("ELO Leaderboard", id=0):
            dataFrame = process_leaderboard(leaderboard)
            dataFrame = dataFrame.rename(
                columns= {
                    "name": "Model",
                    "confidence_interval": "Confidence Interval",
                    "score": "Arena Score",
                    "organization": "Organization",
                    "votes": "Votes",
                }
            )

            column_order = ["Rank", "Rank* (UB)", "Model", "Arena Score", "Confidence Interval", "Votes", "Organization"]
            dataFrame = dataFrame[column_order]
            num_models = len(dataFrame) 
            total_battles = int(dataFrame['Votes'].sum())//2
            md = f"This is the leaderboard of all {num_models} models, and their relative performance in Copilot Arena. There are currently a total of {total_battles} battles."

            gr.Markdown(md, elem_id="leaderboard_markdown")
            gr.DataFrame(
                dataFrame,
                datatype=[
                    "str"
                    for _ in dataFrame.columns 
                ],
                elem_id="arena_hard_leaderboard",
                max_height=600,
                wrap=True,
                interactive=False,
                column_widths=[50, 50, 130, 60, 80, 50, 80],
            )

            gr.Markdown(
                """
        ***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
        Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval). \n
        **Confidence Interval**: represents the range of uncertainty around the Arena Score. It's displayed as +X / -Y, where X is the difference between the upper bound and the score, and Y is the difference between the score and the lower bound.
        """,
                elem_id="leaderboard_markdown",
            )
        with gr.Tab("User Leaderboard", id=1):
            dataFrame, num_registered_users = process_user_leaderboard(user_leaderboard)
            dataFrame = dataFrame.rename(
                columns= {
                    "username": "Username",
                    "count": "Votes"
                }
            )
            column_order = ["Username", "Votes"]
            dataFrame = dataFrame[column_order]

            md = f"This is the leaderboard of all registered users. There are {num_registered_users} registered users and {num_users} total users."
            gr.Markdown(md, elem_id="leaderboard_markdown")
            gr.DataFrame(
                dataFrame,
                datatype=[
                    "str"
                    for _ in dataFrame.columns 
                ],
                elem_id="arena_user_leaderboard",
                max_height=600,
                wrap=True,
                interactive=False,
                column_widths=[180, 20],
            )
        with gr.Tab("Edit ELO Leaderboard", id=2):
            dataFrame = process_leaderboard(edit_leaderboard)
            dataFrame = dataFrame.rename(
                columns= {
                    "name": "Model",
                    "confidence_interval": "Confidence Interval",
                    "score": "Arena Score",
                    "organization": "Organization",
                    "votes": "Votes",
                }
            )

            column_order = ["Rank", "Rank* (UB)", "Model", "Arena Score", "Confidence Interval", "Votes", "Organization"]
            dataFrame = dataFrame[column_order]
            num_models = len(dataFrame) 
            total_battles = int(dataFrame['Votes'].sum())//2
            md = f"This is the leaderboard of all {num_models} models, and their relative performance in Copilot Arena. There are currently a total of {total_battles} edit battles."

            gr.Markdown(md, elem_id="leaderboard_markdown")
            gr.DataFrame(
                dataFrame,
                datatype=[
                    "str"
                    for _ in dataFrame.columns 
                ],
                elem_id="arena_hard_leaderboard",
                max_height=600,
                wrap=True,
                interactive=False,
                column_widths=[50, 50, 130, 60, 80, 50, 80],
            )

            gr.Markdown(
                """
        ***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
        Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval). \n
        **Confidence Interval**: represents the range of uncertainty around the Arena Score. It's displayed as +X / -Y, where X is the difference between the upper bound and the score, and Y is the difference between the score and the lower bound.
        """,
                elem_id="leaderboard_markdown",
            )

def build_demo(leaderboard_json):
    text_size = gr.themes.sizes.text_lg
    # load theme from theme.json
    theme = gr.themes.Default.load("theme.json")
    # set text size to large
    theme.text_size = text_size

    try: 
        with gr.Blocks(
            title="Chatbot Arena Leaderboard",
            theme=theme,
            css=stylesheet,
        ) as demo:
                build_leaderboard(leaderboard_json)
    except Exception as e:
        print(e)
        with gr.Blocks(
            title="Chatbot Arena Leaderboard",
            theme=theme,
        ) as demo:
            with gr.Row():
                gr.Markdown("Something went wrong while setting up the leaderboard. Please check in later.")
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    leaderboard_json = glob.glob("backend/leaderboard.json")
    leaderboard_json = leaderboard_json[-1]

    demo = build_demo(leaderboard_json)
    demo.launch(share=args.share, server_name=args.host, server_port=args.port)