"""
dashboard/pages/3_Risk_Management.py

Risk Management page — ATR stop-loss visualisation, drawdown chart, risk parameters.
This page will be fully implemented in the next phase.
"""
import streamlit as st

st.set_page_config(
    page_title="Risk Management — ML Crypto",
    page_icon="🛡️",
    layout="wide",
)

st.title("Risk Management")
st.caption("ATR stops · daily loss limit · max drawdown circuit breaker")
st.info("This page is coming in the next phase of development.", icon="🚧")
