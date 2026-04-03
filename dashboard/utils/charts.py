"""
dashboard/utils/charts.py

Reusable Plotly chart builders shared across all dashboard pages.

Each function accepts pandas DataFrames or plain Python types and returns
a go.Figure so that pages can call:
    st.plotly_chart(fig, width="stretch")
"""
from typing import List

import pandas as pd
import plotly.graph_objects as go


# ── Overview charts ───────────────────────────────────────────────────────────

def equity_curve_chart(df: pd.DataFrame) -> go.Figure:
    """
    Normalised equity curve: strategy vs buy-and-hold benchmark.

    Parameters
    ----------
    df : backtest DataFrame — must contain equity_norm and bh_equity columns.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df["equity_norm"],
        name="Strategy",
        line=dict(color="#4C9BE8", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["bh_equity"],
        name="Buy & Hold",
        line=dict(color="#FF7F0E", width=1.2, dash="dot"),
        opacity=0.8,
    ))
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="grey", line_width=0.8,
        annotation_text="Break-even", annotation_position="bottom right",
    )
    fig.update_layout(
        title="OOS Equity Curve (normalised to 1.0)",
        xaxis_title="Date",
        yaxis_title="Equity (normalised)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(t=50, b=40, l=60, r=20),
    )
    return fig


def drawdown_chart(df: pd.DataFrame) -> go.Figure:
    """
    Underwater equity / drawdown chart.
    Shows percentage below the rolling equity peak at each bar.

    Parameters
    ----------
    df : backtest DataFrame — must contain equity_norm column.
    """
    roll_max = df["equity_norm"].cummax()
    drawdown = (df["equity_norm"] - roll_max) / roll_max * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=drawdown,
        name="Drawdown (%)",
        fill="tozeroy",
        line=dict(color="#E84C4C", width=1.2),
        fillcolor="rgba(232, 76, 76, 0.18)",
    ))
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.update_layout(
        title="Drawdown (% below peak equity)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300,
        margin=dict(t=40, b=40, l=60, r=20),
    )
    return fig


# ── Model Analysis charts ─────────────────────────────────────────────────────

def feature_importance_chart(
    feature_names: List[str],
    importances: List[float],
    top_n: int = 20,
) -> go.Figure:
    """
    Horizontal bar chart of the top-N feature importances from the trained model.

    Parameters
    ----------
    feature_names : list of feature column names
    importances   : list of importance scores aligned with feature_names
    top_n         : number of features to display (sorted descending)
    """
    pairs = sorted(zip(importances, feature_names), reverse=True)[:top_n]
    imps, names = zip(*pairs)

    fig = go.Figure(go.Bar(
        x=list(imps)[::-1],
        y=list(names)[::-1],
        orientation="h",
        marker_color="#4C9BE8",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Feature Importances (top {top_n})",
        xaxis_title="Importance",
        height=max(320, top_n * 22),
        margin=dict(t=40, b=40, l=160, r=20),
    )
    return fig


def confusion_matrix_chart(
    cm: List[List[int]],
    labels: List[str] = ["Down (0)", "Up (1)"],
) -> go.Figure:
    """
    Annotated heatmap for a 2×2 confusion matrix.

    Parameters
    ----------
    cm     : [[TN, FP], [FN, TP]] as returned by sklearn.metrics.confusion_matrix
    labels : class labels — index 0 = negative, index 1 = positive
    """
    text = [[str(v) for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=18),
        colorscale="Blues",
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix (OOS test set)",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=350,
        margin=dict(t=40, b=60, l=90, r=20),
    )
    return fig


# ── Risk Management charts ────────────────────────────────────────────────────

def atr_stop_chart(
    df: pd.DataFrame,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    n_bars: int = 100,
) -> go.Figure:
    """
    Candlestick chart with ATR stop-loss and take-profit levels overlaid.

    Parameters
    ----------
    df          : raw OHLCV DataFrame (must have open/high/low/close columns)
    entry_price : hypothetical entry price (horizontal line)
    stop_price  : ATR-based stop-loss price
    tp_price    : ATR-based take-profit price
    n_bars      : number of recent candles to display
    """
    df_plot = df.tail(n_bars)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index,
        open=df_plot["open"],
        high=df_plot["high"],
        low=df_plot["low"],
        close=df_plot["close"],
        name="OHLCV",
        increasing_line_color="#2CA02C",
        decreasing_line_color="#D62728",
    ))
    fig.add_hline(
        y=entry_price, line_color="#4C9BE8", line_width=1.5, line_dash="solid",
        annotation_text=f"Entry  {entry_price:,.0f}",
        annotation_position="top right",
    )
    fig.add_hline(
        y=stop_price, line_color="#E84C4C", line_width=1.5, line_dash="dash",
        annotation_text=f"Stop  {stop_price:,.0f}",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=tp_price, line_color="#2CA02C", line_width=1.5, line_dash="dash",
        annotation_text=f"TP  {tp_price:,.0f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"ATR Stop-Loss & Take-Profit (last {n_bars} candles)",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        height=460,
        margin=dict(t=50, b=40, l=80, r=130),
    )
    return fig
