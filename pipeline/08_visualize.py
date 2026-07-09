"""Stage 08: render the interactive map via DataMapPlot.

Composes UMAP coords + Toponymy labels + texts + Wikidata facets into a single
interactive HTML page: portrait thumbnails on hover, click-through to the
Wikipedia article, colormap dropdowns (occupation, citizenship, gender, birth
year, fame, volatility), and title search.

v1 keeps post-render patches minimal -- just the two label-rendering fixes the
sibling projects documented (explicit characterSet; font <link> in <head> so
the SDF atlas isn't built before the font loads). Filter panel / richer search
/ mobile support are later polish (see steam-atlas 09_visualize.py to port).

Inputs: people_texts.parquet, people_wikidata.parquet, labels_lead.parquet,
        umap_coords_lead.npz
Output: data/living_people_map_lead.html + docs/index.html
"""

import json
from html import escape
from pathlib import Path
from urllib.parse import quote

import datamapplot
import glasbey
import numpy as np
import pandas as pd

import config

N_TOP_CATEGORIES = 15  # categorical colormaps show top N + "Other"

CUSTOM_CSS = """
:root {
    --ink: #101318;
    --panel: rgba(23, 27, 34, 0.88);
    --rule: rgba(255, 255, 255, 0.10);
    --text: #e8eaee;
    --text-dim: #9aa3b0;
    --accent: #64b5f6;
}
body { background: var(--ink) !important; color: var(--text) !important; }
.container-box {
    background: var(--panel) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 6px !important;
    backdrop-filter: blur(10px);
}
#title-container { padding: 16px 20px !important; max-width: 430px; }
#main-title { color: var(--text) !important; letter-spacing: 0.01em; }
#title-container > span:last-of-type { color: var(--text-dim) !important; }
#text-search {
    background: rgba(10, 13, 18, 0.8) !important;
    border: 1px solid var(--rule) !important;
    border-radius: 4px !important;
    color: var(--text) !important;
    padding: 7px 11px !important;
}
#text-search:focus { border-color: var(--accent) !important; outline: none !important; }
.deck-tooltip {
    background: transparent !important; border: none !important;
    padding: 0 !important; box-shadow: none !important; max-width: 340px !important;
}
.pc {
    background: #171b22; border: 1px solid var(--rule); border-radius: 6px;
    padding: 14px 16px; box-shadow: 0 14px 40px rgba(0, 0, 0, 0.65);
    font-family: 'Archivo', system-ui, sans-serif;
}
.pc-img {
    float: right; width: 84px; max-height: 116px; object-fit: cover;
    border-radius: 4px; margin: 0 0 8px 12px; background: #0a0d12;
}
.pc-img[src=""] { display: none; }
.pc-title { font-size: 15.5px; font-weight: 650; line-height: 1.25; color: var(--text); }
.pc-desc { font-size: 12.5px; color: var(--accent); margin-top: 4px; line-height: 1.35; }
.pc-desc:empty { display: none; }
.pc-chips { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; clear: none; }
.pc-chip {
    padding: 2px 7px; border: 1px solid var(--rule); border-radius: 3px;
    font-size: 10px; font-variant-numeric: tabular-nums; letter-spacing: 0.05em;
    text-transform: uppercase; color: var(--text-dim); white-space: nowrap;
}
.pc-lead {
    font-size: 12.5px; line-height: 1.5; color: var(--text-dim);
    margin-top: 10px; padding-top: 9px; border-top: 1px solid var(--rule);
    display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical; overflow: hidden;
}
.pc-lead:empty { display: none; }
"""

HOVER_TEMPLATE = (
    '<div class="pc">'
    '<img src="{thumb_url}" class="pc-img" alt="" onerror="this.style.display=\'none\'"/>'
    '<div class="pc-title">{name}</div>'
    '<div class="pc-desc">{description}</div>'
    '<div class="pc-chips">'
    '<span class="pc-chip">#{rank}</span>'
    '<span class="pc-chip">{views_str} views/mo</span>'
    '<span class="pc-chip">{born_str}</span>'
    "</div>"
    '<div class="pc-lead">{lead_excerpt}</div>'
    "</div>"
)


def top_n_or_other(values: pd.Series, n: int = N_TOP_CATEGORIES) -> np.ndarray:
    top = values.value_counts().head(n).index
    return np.where(values.isin(top), values, "Other")


def categorical_colormap(field: str, description: str, values: np.ndarray) -> tuple[np.ndarray, dict]:
    unique_vals = sorted(set(values))
    if "Other" in unique_vals:  # keep Other last in the legend
        unique_vals.remove("Other")
        unique_vals.append("Other")
    palette = glasbey.create_palette(palette_size=len(unique_vals))
    mapping = dict(zip(unique_vals, palette))
    if "Other" in mapping:
        mapping["Other"] = "#5a6172"
    if "Unknown" in mapping:
        mapping["Unknown"] = "#3d434e"
    return values, {"field": field, "description": description, "kind": "categorical", "color_mapping": mapping}


def main() -> None:
    people = pd.read_parquet(config.TEXTS_PARQUET)
    wikidata = pd.read_parquet(config.WIKIDATA_PARQUET).drop(columns=["qid"])
    labels = pd.read_parquet(config.labels_parquet())
    coords_data = np.load(config.umap_coords_npz(), allow_pickle=False)
    coords = coords_data["coords"]
    assert (coords_data["pageid"] == people["pageid"].to_numpy()).all(), "coords/texts row order mismatch"

    df = people.merge(wikidata, on="pageid", how="left").merge(labels, on="pageid", how="left")
    assert len(df) == len(people), "merge changed row count"
    print(f"Rendering {len(df):,} people")

    label_columns = sorted(c for c in df.columns if c.startswith("label_layer_"))
    # The parquet stores label_layer_0 = COARSEST. DataMapPlot wants label
    # layers FINEST-FIRST (create_interactive_plot docstring), and its
    # hierarchical_collision_priority gives later (coarser) layers the win.
    # Passing coarsest-first inverts zoom gating AND collision priority --
    # fine labels crowd out coarse ones at overview zoom.
    topic_name_vectors = [df[c].fillna("Unlabelled").to_numpy() for c in reversed(label_columns)]

    # -- per-point display fields ------------------------------------------
    occupations = df["occupations"].apply(lambda o: o[0].title() if isinstance(o, np.ndarray) and len(o) else "Unknown")
    citizenships = df["citizenships"].apply(lambda c: c[0] if isinstance(c, np.ndarray) and len(c) else "Unknown")
    gender = df["gender"].fillna("Unknown").str.title()
    born = df["birth_year"]

    views_str = df["median_views"].apply(lambda v: f"{v / 1000:.0f}k" if v >= 10_000 else f"{v:,.0f}")
    born_str = born.apply(lambda y: f"b. {int(y)}" if pd.notna(y) else "b. ?")
    lead_excerpt = df["lead"].fillna("").str.slice(0, 480).apply(escape)
    wiki_urls = df["title"].apply(lambda t: "https://en.wikipedia.org/wiki/" + quote(t.replace(" ", "_")))

    extra_data = pd.DataFrame(
        {
            "name": df["title"].apply(escape),
            "description": df["description"].fillna("").apply(escape),
            "thumb_url": df["thumb_url"].fillna(""),
            "rank": df["rank"].astype(int).astype(str),
            "views_str": views_str,
            "born_str": born_str,
            "lead_excerpt": lead_excerpt,
            "wiki_url": wiki_urls,
        }
    )

    log_views = np.log10(df["median_views"].clip(lower=1))
    marker_sizes = (2.5 + 10 * (log_views - log_views.min()) / (log_views.max() - log_views.min())).to_numpy()

    # -- colormaps ----------------------------------------------------------
    all_rawdata, all_metadata = [], []
    for field, description, series in [
        ("occupation", "Occupation (Wikidata)", occupations),
        ("citizenship", "Citizenship (Wikidata)", citizenships),
    ]:
        values, meta = categorical_colormap(field, description, top_n_or_other(series))
        all_rawdata.append(values)
        all_metadata.append(meta)

    values, meta = categorical_colormap("gender", "Gender (Wikidata)", top_n_or_other(gender, 4))
    all_rawdata.append(values)
    all_metadata.append(meta)

    all_rawdata.append(born.fillna(born.median()).to_numpy(dtype=float))
    all_metadata.append({"field": "birth_year", "description": "Birth Year", "kind": "continuous", "cmap": "plasma"})

    all_rawdata.append(log_views.to_numpy())
    all_metadata.append(
        {"field": "fame", "description": "Median Monthly Views (log10)", "kind": "continuous", "cmap": "viridis"}
    )

    volatility = np.log10((df["max_views"] / df["median_views"].clip(lower=1)).clip(lower=1))
    all_rawdata.append(volatility.to_numpy())
    all_metadata.append(
        {
            "field": "volatility",
            "description": "Fame Volatility (log10 max/median)",
            "kind": "continuous",
            "cmap": "inferno",
        }
    )

    # Global-roster extras (stage 02b columns, carried through stage 03).
    if "top_lang" in df.columns:
        values, meta = categorical_colormap(
            "top_lang", "Dominant Attention Language", top_n_or_other(df["top_lang"].fillna("Unknown"), 12)
        )
        all_rawdata.append(values)
        all_metadata.append(meta)
    if "english_share" in df.columns:
        all_rawdata.append(df["english_share"].fillna(1.0).to_numpy(dtype=float))
        all_metadata.append(
            {
                "field": "english_share",
                "description": "Share of Views from English Wikipedia",
                "kind": "continuous",
                "cmap": "cividis",
            }
        )

    fig = datamapplot.create_interactive_plot(
        coords,
        *topic_name_vectors,
        hover_text=df["title"].tolist(),
        hover_text_html_template=HOVER_TEMPLATE,
        marker_size_array=marker_sizes,
        extra_point_data=extra_data,
        on_click="window.open(`{wiki_url}`, '_blank')",
        colormap_rawdata=all_rawdata,
        colormap_metadata=all_metadata,
        title="Living People Map",
        sub_title=config.MAP_SUBTITLE,
        enable_search=True,
        custom_css=CUSTOM_CSS,
        font_family="Archivo",
        tooltip_font_family="Archivo",
        darkmode=True,
        background_color="#101318",
    )
    fig.save(str(config.MAP_HTML))
    print(f"Saved {config.MAP_HTML}")

    # -- post-render patches (both fix region-label rendering; see docstring) --
    html = Path(config.MAP_HTML).read_text()
    chars = sorted({ch for vec in topic_name_vectors for s in vec for ch in str(s)})
    if 'characterSet:"auto"' not in html:
        raise RuntimeError("characterSet:'auto' marker not found; DataMapPlot template changed -- update patch")
    html = html.replace('characterSet:"auto"', "characterSet:" + json.dumps(chars, ensure_ascii=False), 1)

    font_link = (
        '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Archivo:wght@400;500;600;700&display=block">\n'
    )
    html = html.replace("</head>", font_link + "</head>", 1)
    Path(config.MAP_HTML).write_text(html)
    print(f"Patched characterSet ({len(chars)} glyphs) + font preload")

    config.DOCS_DIR.mkdir(exist_ok=True)
    config.DOCS_HTML.write_text(html)
    print(f"Copied to {config.DOCS_HTML}")


if __name__ == "__main__":
    main()
