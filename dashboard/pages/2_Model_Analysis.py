"""
dashboard/pages/2_Model_Analysis.py

Model Analysis page — feature importances, walk-forward fold results, confusion matrix.
This page will be fully implemented in the next phase.
"""
import streamlit as st

st.set_page_config(
    page_title="Model Analysis — ML Crypto",
    page_icon="🤖",
    layout="wide",
)

st.title("Model Analysis")
st.caption("Feature importances · walk-forward validation folds · confusion matrix")
st.info("This page is coming in the next phase of development.", icon="🚧")
