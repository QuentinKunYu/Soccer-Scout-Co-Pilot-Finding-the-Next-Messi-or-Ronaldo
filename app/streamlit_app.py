"""Streamlit prototype covering Eason's roadmap responsibilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so 'src' module can be found
ROOT_DIR = Path(__file__).parent.parent.resolve()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.llm.llm_client import LLMClient
from src.llm.prompts import render_report_to_markdown
from src.llm.schema import PlayerLLMInput
from src.utils.io_utils import load_player_recommendations

from app.components.filters import render_filters
from app.components.player_detail import render_player_detail
from app.components.player_table import render_player_table

DATA_PATH = Path("app/mock_data/mock_player_recommendations.csv")


@st.cache_data(show_spinner=False)
def load_data(path: Path):
    """Cache the CSV read so UI interactions stay snappy."""

    return load_player_recommendations(path)


def main() -> None:
    """Entry point for Streamlit."""

    st.set_page_config(page_title="Scouting Co-Pilot", layout="wide")
    st.title("⚽️ Smart Scouting Co-Pilot")
    st.caption("Filter undervalued prospects and ask the LLM for context.")

    df = load_data(DATA_PATH)
    filtered_df, filter_state = render_filters(df)
    selected_row = render_player_table(filtered_df)

    if selected_row is None:
        return

    render_player_detail(selected_row)

    if "llm_client" not in st.session_state:
        # Persist the client across reruns so we do not re-initialize SDK state.
        st.session_state.llm_client = LLMClient()

    if st.button("Generate LLM Report", type="primary"):
        player_input = PlayerLLMInput.from_row(selected_row)
        report = st.session_state.llm_client.generate_report(player_input)
        st.markdown(render_report_to_markdown(report))


if __name__ == "__main__":
    main()
