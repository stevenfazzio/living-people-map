"""Choose-N analysis: how far down the fame ranking should the map go?

Reads data/people_ranked.parquet (stage 02) and produces:
  - data/analysis/choose_n.html  -- rank curve, fame histogram, cumulative share
  - stdout: name samples at log-spaced ranks + summary stats at candidate Ns

Run: uv run python analysis/choose_n.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import config  # noqa: E402

OUT_DIR = config.DATA_DIR / "analysis"
CANDIDATE_NS = [10_000, 25_000, 50_000, 100_000]
SAMPLE_RANKS = [1, 100, 1_000, 5_000, 10_000, 25_000, 50_000, 75_000, 100_000]
SAMPLE_WIDTH = 8  # consecutive names shown at each sample rank

ACCENT = "#4269d0"  # single-series accent; text/grid stay in neutral inks
GRID = "#e5e7eb"
INK = "#374151"

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", color=INK, size=13),
    margin=dict(l=70, r=30, t=60, b=60),
    hovermode="x unified",
)


def rank_curve(df: pd.DataFrame) -> go.Figure:
    # thin to ~2k points for a light HTML (log-spaced ranks keep the shape)
    idx = np.unique(np.geomspace(1, len(df), 2000).astype(int)) - 1
    sub = df.iloc[idx]
    fig = go.Figure(
        go.Scatter(
            x=sub["rank"],
            y=sub["median_views"].clip(lower=1),
            mode="lines",
            line=dict(color=ACCENT, width=2),
            name="median monthly views",
            hovertemplate="rank %{x:,}<br>median %{y:,.0f} views/mo<extra></extra>",
        )
    )
    for n in CANDIDATE_NS:
        v = df.loc[df["rank"] == n, "median_views"].iloc[0]
        fig.add_vline(x=n, line_width=1, line_dash="dot", line_color="#9ca3af")
        fig.add_annotation(
            x=np.log10(n),
            y=np.log10(max(v, 1)),
            text=f"N={n // 1000}k<br>{v:,.0f}/mo",
            showarrow=True,
            arrowhead=0,
            ax=25,
            ay=-35,
            font=dict(size=11),
        )
    fig.update_xaxes(type="log", title="fame rank (by median monthly enwiki pageviews)", gridcolor=GRID)
    fig.update_yaxes(type="log", title="median monthly pageviews", gridcolor=GRID)
    fig.update_layout(title="Fame falls off smoothly: median monthly views vs. rank", showlegend=False, **LAYOUT)
    return fig


def fame_histogram(df: pd.DataFrame) -> go.Figure:
    logv = np.log10(df["median_views"].clip(lower=0.5))  # zero-view mass sits at the left edge
    fig = go.Figure(
        go.Histogram(
            x=logv,
            nbinsx=80,
            marker=dict(color=ACCENT, line=dict(color="white", width=1)),
            hovertemplate="log10(median views) %{x}<br>%{y:,} people<extra></extra>",
        )
    )
    for n in CANDIDATE_NS:
        v = df.loc[df["rank"] == n, "median_views"].iloc[0]
        fig.add_vline(
            x=np.log10(max(v, 0.5)),
            line_width=1,
            line_dash="dot",
            line_color="#9ca3af",
            annotation_text=f"top {n // 1000}k",
            annotation_font_size=11,
        )
    fig.update_xaxes(title="log10(median monthly pageviews) -- all 1.15M living people", gridcolor=GRID)
    fig.update_yaxes(title="people", gridcolor=GRID)
    fig.update_layout(title="Where candidate cutoffs land in the full fame distribution", showlegend=False, **LAYOUT)
    return fig


def cumulative_share(df: pd.DataFrame) -> go.Figure:
    cum = df["total_views"].cumsum() / df["total_views"].sum()
    idx = np.unique(np.geomspace(1, len(df), 2000).astype(int)) - 1
    fig = go.Figure(
        go.Scatter(
            x=df["rank"].iloc[idx],
            y=cum.iloc[idx],
            mode="lines",
            line=dict(color=ACCENT, width=2),
            hovertemplate="top %{x:,} people<br>%{y:.1%} of all views<extra></extra>",
        )
    )
    for n in CANDIDATE_NS:
        fig.add_annotation(
            x=np.log10(n),
            y=cum.iloc[n - 1],
            text=f"{n // 1000}k: {cum.iloc[n - 1]:.0%}",
            showarrow=True,
            arrowhead=0,
            ax=0,
            ay=30,
            font=dict(size=11),
        )
    fig.update_xaxes(type="log", title="top N people", gridcolor=GRID)
    fig.update_yaxes(title="share of all living-people pageviews", tickformat=".0%", gridcolor=GRID)
    fig.update_layout(title="Share of total fame captured by top N", showlegend=False, **LAYOUT)
    return fig


def name_samples(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for r in SAMPLE_RANKS:
        rows = df[(df["rank"] >= r) & (df["rank"] < r + SAMPLE_WIDTH)][["rank", "title", "median_views", "n_months"]]
        frames.append(rows)
    return pd.concat(frames)


def main() -> None:
    df = pd.read_parquet(config.RANKED_PARQUET, columns=["rank", "title", "median_views", "total_views", "n_months"])
    df = df.sort_values("rank").reset_index(drop=True)

    print("=== Name samples at log-spaced ranks ===")
    for r in SAMPLE_RANKS:
        rows = df[(df["rank"] >= r) & (df["rank"] < r + SAMPLE_WIDTH)]
        names = ", ".join(rows["title"])
        v = rows["median_views"].iloc[0]
        print(f"\nrank {r:>7,} (~{v:,.0f} views/mo): {names}")

    print("\n=== Summary at candidate Ns ===")
    cum = df["total_views"].cumsum() / df["total_views"].sum()
    for n in CANDIDATE_NS:
        v = df.loc[n - 1, "median_views"]
        print(f"top {n:>7,}: floor {v:>8,.0f} median views/mo, captures {cum.iloc[n - 1]:.1%} of all views")
    zero = (df["median_views"] == 0).sum()
    print(f"\npeople with median 0 (viewed in <=6 of 12 months or never): {zero:,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figs = [rank_curve(df), fame_histogram(df), cumulative_share(df)]
    html_parts = [
        f.to_html(full_html=False, include_plotlyjs=("inline" if i == 0 else False)) for i, f in enumerate(figs)
    ]
    out = OUT_DIR / "choose_n.html"
    out.write_text(
        "<html><head><meta charset='utf-8'><title>living-people-map: choosing N</title></head>"
        "<body style='max-width:1000px;margin:2rem auto;font-family:sans-serif'>"
        "<h1>Choosing N</h1>" + "".join(html_parts) + "</body></html>"
    )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
