from fastchat.serve.monitor.monitor import basic_component_values, leader_component_values
from fastchat.utils import build_logger

import argparse
import glob
import gradio as gr
import pandas as pd

stylesheet = """
<style>
    .column {
        padding: 20px;
    }
</style>
"""

def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_component_values + leader_component_values

def recompute_ub_ranking(arena_df):
    # From https://github.com/lm-sys/FastChat/blob/e208d5677c6837d590b81cb03847c0b9de100765/fastchat/serve/monitor/monitor.py#L51
    ranking = {}
    for i, model_a in enumerate(arena_df.index):
        ranking[model_a] = 1
        for j, model_b in enumerate(arena_df.index):
            if i == j:
                continue
            if (
                arena_df.loc[model_b]["lower"]
                > arena_df.loc[model_a]["upper"]
            ):
                ranking[model_a] += 1
    return ranking

def process_leaderboard(filepath):
    leaderboard = pd.read_csv(filepath)

    # Round scores to integers
    leaderboard['score'] = leaderboard['score'].round().astype(int)
    leaderboard['upper'] = leaderboard['upper'].round().astype(int)
    leaderboard['lower'] = leaderboard['lower'].round().astype(int)

    # Calculate the difference for upper and lower bounds
    leaderboard['upper_diff'] = leaderboard['upper'] - leaderboard['score']
    leaderboard['lower_diff'] = leaderboard['score'] - leaderboard['lower']

    # Combine the differences into a single column with +/- format
    leaderboard['confidence_interval'] = '+' + leaderboard['upper_diff'].astype(str) + ' / -' + leaderboard['lower_diff'].astype(str)

    rankings_ub = recompute_ub_ranking(leaderboard)
    leaderboard.insert(loc=0, column="Rank* (UB)", value=rankings_ub)
    leaderboard['Rank'] = leaderboard['score'].rank(ascending=False).astype(int)

    # Sort the leaderboard by rank and then by score
    leaderboard = leaderboard.sort_values(by=['Rank'], ascending=[True])
    
    return leaderboard

def process_user_leaderboard(filepath):
    NUM_SHOW_USERS = 14

    leaderboard = pd.read_csv(filepath)

    # Sort the leaderboard by votes
    leaderboard = leaderboard.sort_values(by=['count'], ascending=[False])

    return leaderboard.head(NUM_SHOW_USERS), len(leaderboard)

def build_leaderboard(leaderboard_table_file, user_leaderboard_table_file):
    link_color = "#1976D2"  # noqa: F841
    default_md = "Welcome to Copilot Arena leaderboard 🏆"

    with gr.Row():
        with gr.Column(scale=4):
            md_1 = gr.Markdown(default_md, elem_id="leaderboard_markdown")  # noqa: F841
    with gr.Row():
        with gr.Column(scale=1, elem_classes="column"):
            dataFrame = process_leaderboard(leaderboard_table_file)
            dataFrame = dataFrame.rename(
                columns= {
                    "name": "Model",
                    "confidence_interval": "Confidence Interval",
                    "score": "Arena Score",
                    "organization": "Organization",
                    "wins": "Votes",
                }
            )

            column_order = ["Rank", "Rank* (UB)", "Model", "Arena Score", "Confidence Interval", "Votes", "Organization"]
            dataFrame = dataFrame[column_order]
            num_models = len(dataFrame) 
            total_votes = int(dataFrame['Votes'].sum())
            md = f"This is the leaderboard of all the models. There are {num_models} models and a total of {total_votes} votes."

            gr.Markdown(md, elem_id="leaderboard_markdown")
            gr.DataFrame(
                dataFrame,
                datatype=[
                    "str"
                    for _ in dataFrame.columns 
                ],
                elem_id="arena_hard_leaderboard",
                height=800,
                wrap=True,
                interactive=False,
                column_widths=[50, 50, 130, 60, 80, 50, 80],
            )
            
        with gr.Column(scale=1, elem_classes="column"):
            dataFrame, num_users = process_user_leaderboard(user_leaderboard_table_file)
            dataFrame = dataFrame.rename(
                columns= {
                    "userId": "User ID",
                    "count": "Votes"
                }
            )
            column_order = ["User ID", "Votes"]
            dataFrame = dataFrame[column_order]

            md = f"This is the leaderboard of all the users. There are {num_users} users."
            gr.Markdown(md, elem_id="leaderboard_markdown")
            gr.DataFrame(
                dataFrame,
                datatype=[
                    "str"
                    for _ in dataFrame.columns 
                ],
                elem_id="arena_user_leaderboard",
                height=800,
                wrap=True,
                interactive=False,
                column_widths=[170, 30],
            )
    with gr.Row():
        gr.Markdown(
                """
        ***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
        Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval). \n
        **Confidence Interval**: represents the range of uncertainty around the Arena Score. It's displayed as +X / -Y, where X is the difference between the upper bound and the score, and Y is the difference between the score and the lower bound.
        """,
                elem_id="leaderboard_markdown",
            )

def build_demo(leaderboard_table_file, user_leaderboard_table_file):
    from fastchat.serve.gradio_web_server import block_css

    text_size = gr.themes.sizes.text_lg
    # load theme from theme.json
    theme = gr.themes.Default.load("theme.json")
    # set text size to large
    theme.text_size = text_size
    theme.set(
        button_large_text_size="40px",
        button_small_text_size="40px",
        button_large_text_weight="1000",
        button_small_text_weight="1000",
        button_shadow="*shadow_drop_lg",
        button_shadow_hover="*shadow_drop_lg",
        checkbox_label_shadow="*shadow_drop_lg",
        button_shadow_active="*shadow_inset",
        button_secondary_background_fill="*primary_300",
        button_secondary_background_fill_dark="*primary_700",
        button_secondary_background_fill_hover="*primary_200",
        button_secondary_background_fill_hover_dark="*primary_500",
        button_secondary_text_color="*primary_800",
        button_secondary_text_color_dark="white",
    )

    with gr.Blocks(
        title="Chatbot Arena Leaderboard",
        theme=theme,
        css=block_css,
    ) as demo:
        build_leaderboard(leaderboard_table_file, user_leaderboard_table_file)
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logger = build_logger("monitor", "monitor.log")
    logger.info(f"args: {args}")

    leaderboard_table_files = glob.glob("backend/leaderboard.csv")
    leaderboard_table_file = leaderboard_table_files[-1]

    user_leaderboard_table_files = glob.glob("backend/user_leaderboard.csv")
    user_leaderboard_table_file = user_leaderboard_table_files[-1]

    demo = build_demo(leaderboard_table_file, user_leaderboard_table_file)
    demo.launch(share=args.share, server_name=args.host, server_port=args.port)