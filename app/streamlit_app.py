"""Streamlit prototype covering Eason's roadmap responsibilities."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path so 'src' module can be found
# ROOT_DIR: Project root directory path to ensure Python can find the 'src' module
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

# DATA_PATH: Path to the CSV file containing player recommendations
DATA_PATH = Path("app/mock_data/mock_player_recommendations.csv")


@st.cache_data(show_spinner=False)
def load_data(path: Path):
    """
    Cache CSV data loading to avoid re-reading the file on every interaction.
    
    Uses @st.cache_data decorator to cache results, improving UI responsiveness.
    show_spinner=False: Disables loading animation since data is cached and loads quickly.
    """
    return load_player_recommendations(path)


def _render_custom_css():
    """
    Inject custom CSS styles with theme-aware design.
    
    Design principles:
    - Adapts to light/dark theme automatically
    - Consistent card heights for visual harmony
    - Aligned tooltips and legends
    - Professional color palette
    - Responsive and accessible
    """
    st.markdown(
        """
        <style>
        /* Import professional font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        }
        
        /* Typography hierarchy - theme aware */
        h1 {
            font-size: 2.25rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            letter-spacing: -0.02em;
        }
        
        h2 {
            font-size: 1.75rem !important;
            font-weight: 600 !important;
            margin-top: 2rem !important;
            margin-bottom: 1rem !important;
        }
        
        h3 {
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            margin-top: 1.5rem !important;
            margin-bottom: 0.75rem !important;
        }
        
        h4 {
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }
        
        /* Consistent metric cards with fixed height */
        [data-testid="stMetric"] {
            background: var(--background-color);
            border: 1px solid var(--border-color);
            border-radius: 0.5rem;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            min-height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        [data-testid="stMetricValue"] {
            font-size: 1.75rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.875rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            line-height: 1.4 !important;
        }
        
        [data-testid="stMetricDelta"] {
            font-size: 0.875rem !important;
            margin-top: 0.25rem !important;
        }
        
        /* Info box styling - theme aware with red accent */
        .info-box {
            border: 1px solid var(--border-color);
            border-left: 4px solid #ef4444;
            border-radius: 0.5rem;
            padding: 1.25rem;
            margin: 1.5rem 0;
        }
        
        .info-box p {
            margin: 0.5rem 0;
            line-height: 1.6;
        }
        
        .info-box strong {
            font-weight: 600 !important;
        }
        
        /* Primary button - red theme */
        .stButton > button {
            background: #ef4444 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 0.5rem !important;
            padding: 0.75rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        
        .stButton > button:hover {
            background: #dc2626 !important;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3) !important;
        }
        
        /* DataFrames */
        .dataframe {
            font-size: 0.9rem !important;
            border: 1px solid var(--border-color) !important;
            border-radius: 0.5rem !important;
        }
        
        .dataframe thead tr th {
            font-weight: 600 !important;
            padding: 0.75rem !important;
            border-bottom: 2px solid var(--border-color) !important;
        }
        
        .dataframe tbody tr {
            border-bottom: 1px solid var(--border-color) !important;
        }
        
        .dataframe tbody tr td {
            padding: 0.75rem !important;
        }
        
        /* Sidebar - theme aware */
        [data-testid="stSidebar"] {
            border-right: 1px solid var(--border-color);
        }
        
        
        /* Selectbox styling */
        .stSelectbox label,
        .stSlider label {
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }
        
        /* Tooltip alignment */
        .stTooltipIcon {
            vertical-align: middle !important;
            margin-left: 0.25rem !important;
        }
        
        /* Captions */
        [data-testid="stCaptionContainer"] {
            font-size: 0.875rem !important;
            line-height: 1.5 !important;
        }
        
        /* Horizontal rule */
        hr {
            border-color: var(--border-color) !important;
            margin: 2rem 0 !important;
        }
        
        /* Alerts */
        .stAlert {
            border-radius: 0.5rem !important;
        }
        
        /* Markdown lists alignment */
        .stMarkdown ul {
            padding-left: 1.5rem;
        }
        
        .stMarkdown li {
            margin-bottom: 0.25rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """
    Main entry point for the application.
    
    Execution flow:
    1. Configure page settings (title, wide layout)
    2. Inject custom CSS styles
    3. Display application title and description
    4. Load player data
    5. Render filters and get filtered data
    6. Display player table and let user select
    7. Display detailed information for selected player
    8. Provide AI analysis report functionality
    """
    # set_page_config: Set page title to "Scout Co-Pilot" and use wide layout to display more information
    st.set_page_config(page_title="Scout Co-Pilot", layout="wide")
    
    # Inject custom CSS styles
    _render_custom_css()
    
    # Display application title
    st.title("Smart Scouting Co-Pilot")
    st.markdown(
        """
        <p style='font-size: 1.15rem; color: #d1d5db; margin-bottom: 2rem;'>
        Filter undervalued prospects and get deep insights with AI
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Load player data (uses cache, only loads on first execution)
    df = load_data(DATA_PATH)
    
    # render_filters: Display sidebar filters, return filtered DataFrame and filter state
    filtered_df, filter_state = render_filters(df)
    
    # render_player_table: Display player table, return selected player data (one row)
    selected_row = render_player_table(filtered_df)

    # If no player selected, exit function early
    if selected_row is None:
        return

    # render_player_detail: Display complete detailed information for selected player
    render_player_detail(selected_row)

    # Initialize LLM client (only create on first use, retrieve from session_state afterwards)
    if "llm_client" not in st.session_state:
        # LLMClient: Large language model client for generating AI analysis reports
        st.session_state.llm_client = LLMClient()

    # Provide "Generate AI Analysis Report" button
    st.markdown("---")  # Separator line
    st.markdown("### AI Deep Analysis")
    
    if st.button("Generate Complete Analysis Report", type="primary", use_container_width=True):
        # Display loading animation
        with st.spinner("AI is analyzing player data..."):
            # PlayerLLMInput.from_row: Convert player data to LLM input format
            player_input = PlayerLLMInput.from_row(selected_row)
            
            # generate_report: Call LLM to generate analysis report
            report = st.session_state.llm_client.generate_report(player_input)
            
            # render_report_to_markdown: Convert report to Markdown format and display
            st.markdown(render_report_to_markdown(report))


if __name__ == "__main__":
    main()
