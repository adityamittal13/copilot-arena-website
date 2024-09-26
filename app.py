from fastchat.serve.monitor.monitor import basic_component_values, leader_component_values
from fastchat.utils import build_logger

import argparse
import glob
import gradio as gr
import pandas as pd


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")
    return basic_component_values + leader_component_values

def recompute_final_ranking(arena_df):
    # compute ranking based on CI
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
    return list(ranking.values())

def process_leaderboard(filepath):
    leaderboard = pd.read_csv(filepath)

    leaderboard['score'] = leaderboard['score'].round(2)
    leaderboard['upper'] = leaderboard['upper'].round(2)
    leaderboard['lower'] = leaderboard['lower'].round(2)

    rankings = recompute_final_ranking(leaderboard)
    leaderboard.insert(loc=0, column="Rank* (UB)", value=rankings)
    return leaderboard

def build_leaderboard_tab(leaderboard_table_file, mirror=False):
    link_color = "#1976D2"
    default_md = "Welcome to Chatbot Arena leaderboard..."

    with gr.Row():
        with gr.Column(scale=4):
            md_1 = gr.Markdown(default_md, elem_id="leaderboard_markdown")
    with gr.Tab("Leaderboard", id=0):
        dataFrame = process_leaderboard(leaderboard_table_file)
        dataFrame = dataFrame.rename(
            columns= {
                "model": "Model",
                "name": "Model Name",
                "lower": "25% Quartile",
                "upper": "75% Quartile",
                "score": "Arena Score",
                "organization": "Organization"
            }
        )
        model_to_score = {}
        for i in range(len(dataFrame)):
            model_to_score[dataFrame.loc[i, "Model"]] = dataFrame.loc[
                i, "Arena Score"
            ]
        column_order = ["Rank* (UB)", "Model", "Model Name", "Arena Score", "25% Quartile", "75% Quartile", "Organization"]
        dataFrame = dataFrame[column_order]
        md = "This is the leaderboard of all the values..."
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
            column_widths=[70, 130, 120, 70, 70, 70, 80],
        )

        gr.Markdown(
        """
***Rank (UB)**: model's ranking (upper-bound), defined by one + the number of models that are statistically better than the target model.
Model A is statistically better than model B when A's lower-bound score is greater than B's upper-bound score (in 95% confidence interval).
""",
        elem_id="leaderboard_markdown",
    )

def build_demo(leaderboard_table_file):
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
        build_leaderboard_tab(leaderboard_table_file)
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

    demo = build_demo(leaderboard_table_file)
    demo.launch(share=args.share, server_name=args.host, server_port=args.port)