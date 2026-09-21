"""
Netflix Content Intelligence Dashboard
=======================================
IBM AICTE Business Intelligence Project
Dataset: Netflix Movies and TV Shows (Kaggle)
Author: BI Analytics Team

Run: python app.py
Open browser: http://127.0.0.1:8050
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────

# Resolve CSV path relative to this file so it works from any working directory
_HERE = os.path.dirname(os.path.abspath(__file__))

def load_and_clean(path=None):
    """Load Netflix CSV and return a clean DataFrame."""
    if path is None:
        path = os.path.join(_HERE, "netflix_titles.csv")
    df = pd.read_csv(path)

    # ── Fill missing categorical values ──
    df["director"] = df["director"].fillna("Unknown")
    df["cast"] = df["cast"].fillna("Unknown")
    df["country"] = df["country"].fillna("Unknown")
    df["rating"] = df["rating"].fillna("Not Rated")

    # ── Parse date_added ──
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
    df["year_added"] = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month
    df["month_name"] = df["date_added"].dt.strftime("%b")

    # ── Duration: split into numeric + unit ──
    df["duration_value"] = df["duration"].str.extract(r"(\d+)").astype(float)
    df["duration_unit"] = df["duration"].str.extract(r"([A-Za-z ]+)").iloc[:, 0].str.strip()

    # ── Primary country (first listed) ──
    df["primary_country"] = df["country"].apply(
        lambda x: x.split(",")[0].strip()
        if isinstance(x, str) and x.strip() not in ("", "Unknown")
        else "Unknown"
    )

    # ── Primary genre (first listed) ──
    df["primary_genre"] = df["listed_in"].apply(
        lambda x: x.split(",")[0].strip() if isinstance(x, str) and x.strip() != "" else "Unknown"
    )

    # ── Decade of release ──
    df["decade"] = (df["release_year"] // 10 * 10).astype(str) + "s"

    # ── Content age bucket ──
    current_year = 2021
    df["age_at_add"] = current_year - df["release_year"]

    return df


df = load_and_clean()

# ─────────────────────────────────────────────
# 2. KPI COMPUTATIONS
# ─────────────────────────────────────────────

total_titles   = len(df)
total_movies   = len(df[df["type"] == "Movie"])
total_shows    = len(df[df["type"] == "TV Show"])
total_countries = df["primary_country"].nunique()
total_genres   = df["listed_in"].str.split(",").explode().str.strip().nunique()
avg_movie_dur  = df[df["type"] == "Movie"]["duration_value"].mean()
latest_year    = int(df["release_year"].max())

# ─────────────────────────────────────────────
# 3. CHART BUILDERS
# ─────────────────────────────────────────────

NETFLIX_RED   = "#E50914"
DARK_BG       = "#141414"
CARD_BG       = "#1e1e1e"
TEXT_COLOR    = "#ffffff"
ACCENT        = "#E50914"
MUTED         = "#b3b3b3"
PALETTE       = [NETFLIX_RED, "#f5c518", "#4fc3f7", "#81c784", "#ce93d8",
                 "#ffb74d", "#80cbc4", "#ef9a9a", "#90caf9", "#a5d6a7"]


def chart_layout(fig, title="", height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=14), x=0.01),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_COLOR, size=11),
        height=height,
        margin=dict(l=30, r=20, t=45, b=30),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_COLOR)),
    )
    fig.update_xaxes(gridcolor="#2e2e2e", linecolor="#2e2e2e", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#2e2e2e", linecolor="#2e2e2e", tickfont=dict(color=MUTED))
    return fig


# ── 3a. Movies vs TV Shows donut ──
def fig_type_donut():
    counts = df["type"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.6,
        marker=dict(colors=[NETFLIX_RED, "#4fc3f7"]),
        textfont=dict(color=TEXT_COLOR),
    ))
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: %{value}<extra></extra>")
    return chart_layout(fig, "Movies vs TV Shows")


# ── 3b. Top 10 countries ──
def fig_top_countries():
    top = (
        df[df["primary_country"] != "Unknown"]["primary_country"]
        .value_counts()
        .head(10)
        .sort_values()
    )
    fig = px.bar(
        x=top.values, y=top.index, orientation="h",
        color=top.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Titles", "y": "Country"},
    )
    fig.update_coloraxes(showscale=False)
    return chart_layout(fig, "Top 10 Content-Producing Countries")


# ── 3c. Yearly additions trend ──
def fig_yearly_additions():
    yearly = df.groupby(["year_added", "type"]).size().reset_index(name="count")
    yearly = yearly[yearly["year_added"].notna() & (yearly["year_added"] >= 2010)]
    fig = px.line(
        yearly, x="year_added", y="count", color="type",
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        markers=True, labels={"year_added": "Year", "count": "Titles Added", "type": ""},
    )
    return chart_layout(fig, "Titles Added to Netflix Per Year (2010–2021)")


# ── 3d. Top 15 genres ──
def fig_top_genres():
    genres = (
        df["listed_in"].str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(15)
        .sort_values()
    )
    fig = px.bar(
        x=genres.values, y=genres.index, orientation="h",
        color=genres.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Titles", "y": "Genre"},
    )
    fig.update_coloraxes(showscale=False)
    return chart_layout(fig, "Top 15 Content Genres", height=420)


# ── 3e. Rating distribution ──
def fig_ratings():
    ratings_order = ["TV-Y", "TV-Y7", "TV-G", "G", "TV-PG", "PG", "PG-13",
                     "TV-14", "TV-MA", "R", "NC-17", "NR", "Not Rated"]
    rc = df["rating"].value_counts().reindex(ratings_order).dropna()
    fig = px.bar(
        x=rc.index, y=rc.values,
        color=rc.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Rating", "y": "Titles"},
    )
    fig.update_coloraxes(showscale=False)
    return chart_layout(fig, "Content Ratings Distribution")


# ── 3f. Release decade distribution ──
def fig_decade():
    dc = df.groupby(["decade", "type"]).size().reset_index(name="count")
    dc = dc[dc["decade"] != "1920s"]  # drop extreme outlier decade with 1 title
    fig = px.bar(
        dc, x="decade", y="count", color="type", barmode="group",
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        labels={"decade": "Decade", "count": "Titles", "type": ""},
    )
    return chart_layout(fig, "Content by Release Decade")


# ── 3g. Monthly additions heatmap ──
def fig_monthly_heatmap():
    monthly = (
        df[df["year_added"].notna() & (df["year_added"] >= 2015)]
        .groupby(["year_added", "month_added"])
        .size()
        .reset_index(name="count")
    )
    pivot = monthly.pivot(index="month_added", columns="year_added", values="count").fillna(0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(int(c)) for c in pivot.columns],
        y=month_labels[:len(pivot.index)],
        colorscale=[[0, "#141414"], [0.5, "#b71c1c"], [1, NETFLIX_RED]],
        hoverongaps=False,
        hovertemplate="Year: %{x}<br>Month: %{y}<br>Titles: %{z}<extra></extra>",
    ))
    return chart_layout(fig, "Monthly Content Addition Heatmap (2015–2021)")


# ── 3h. Movie duration distribution ──
def fig_movie_duration():
    movies = df[(df["type"] == "Movie") & (df["duration_value"].notna())]
    fig = px.histogram(
        movies, x="duration_value", nbins=40, color_discrete_sequence=[NETFLIX_RED],
        labels={"duration_value": "Duration (minutes)", "count": "Titles"},
    )
    fig.add_vline(
        x=avg_movie_dur, line_dash="dash", line_color="#f5c518",
        annotation_text=f"Avg: {avg_movie_dur:.0f} min",
        annotation_font_color="#f5c518",
    )
    return chart_layout(fig, "Movie Duration Distribution")


# ── 3i. TV Show seasons distribution ──
def fig_tv_seasons():
    shows = df[(df["type"] == "TV Show") & (df["duration_value"].notna())]
    sc = shows["duration_value"].value_counts().sort_index().head(10)
    fig = px.bar(
        x=sc.index.astype(int), y=sc.values,
        color=sc.values, color_continuous_scale=[[0, "#2e2e2e"], [1, "#4fc3f7"]],
        labels={"x": "Number of Seasons", "y": "TV Shows"},
    )
    fig.update_coloraxes(showscale=False)
    return chart_layout(fig, "TV Shows by Number of Seasons")


# ── 3j. Top countries choropleth map ──
def fig_world_map():
    country_counts = (
        df[df["primary_country"] != "Unknown"]["primary_country"]
        .value_counts()
        .reset_index()
    )
    country_counts.columns = ["country", "count"]
    fig = px.choropleth(
        country_counts, locations="country", locationmode="country names",
        color="count", color_continuous_scale=[[0, "#1e1e1e"], [0.3, "#b71c1c"], [1, NETFLIX_RED]],
        labels={"count": "Titles"},
    )
    fig.update_geos(
        bgcolor=DARK_BG, lakecolor=DARK_BG, landcolor="#2e2e2e",
        showcoastlines=True, coastlinecolor="#444",
    )
    fig.update_coloraxes(colorbar_tickfont=dict(color=TEXT_COLOR))
    return chart_layout(fig, "Global Content Distribution", height=400)


# ── 3k. Rating vs Type breakdown ──
def fig_rating_type():
    top_ratings = df["rating"].value_counts().head(8).index.tolist()
    sub = df[df["rating"].isin(top_ratings)]
    rt = sub.groupby(["rating", "type"]).size().reset_index(name="count")
    fig = px.bar(
        rt, x="rating", y="count", color="type", barmode="stack",
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        labels={"rating": "Rating", "count": "Titles", "type": ""},
    )
    return chart_layout(fig, "Rating Distribution by Content Type")


# ─────────────────────────────────────────────
# 4. DASH APPLICATION LAYOUT
# ─────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="Netflix BI Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server  # exposed for gunicorn

# KPI card helper
def kpi_card(label, value, sub="", color=NETFLIX_RED):
    return html.Div(
        className="kpi-card",
        style={"borderTop": f"3px solid {color}"},
        children=[
            html.P(label, className="kpi-label"),
            html.H3(str(value), className="kpi-value", style={"color": color}),
            html.P(sub, className="kpi-sub"),
        ],
    )


# Inline CSS
styles = f"""
body {{
    background-color: {DARK_BG};
    color: {TEXT_COLOR};
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 0;
    padding: 0;
}}
.header {{
    background: #0a0a0a;
    border-bottom: 2px solid {NETFLIX_RED};
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}}
.header h1 {{
    font-size: 22px;
    color: {TEXT_COLOR};
    margin: 0;
    letter-spacing: 1px;
}}
.header span.badge {{
    background: {NETFLIX_RED};
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}}
.kpi-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 20px 24px 0;
}}
.kpi-card {{
    background: {CARD_BG};
    border-radius: 8px;
    padding: 14px 20px;
    flex: 1;
    min-width: 130px;
    border: 1px solid #2e2e2e;
}}
.kpi-label {{
    font-size: 11px;
    color: {MUTED};
    margin: 0 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: bold;
    margin: 0;
}}
.kpi-sub {{
    font-size: 11px;
    color: {MUTED};
    margin: 4px 0 0 0;
}}
.section-title {{
    font-size: 13px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 20px 24px 0;
    margin: 0;
}}
.chart-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px 24px;
}}
.chart-half {{
    flex: 1;
    min-width: 340px;
    background: {CARD_BG};
    border-radius: 8px;
    border: 1px solid #2e2e2e;
}}
.chart-full {{
    flex: 1 1 100%;
    background: {CARD_BG};
    border-radius: 8px;
    border: 1px solid #2e2e2e;
}}
.chart-third {{
    flex: 1;
    min-width: 280px;
    background: {CARD_BG};
    border-radius: 8px;
    border: 1px solid #2e2e2e;
}}
.filter-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    padding: 12px 24px;
    background: {CARD_BG};
    border-bottom: 1px solid #2e2e2e;
    align-items: center;
}}
.filter-bar label {{
    font-size: 11px;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 4px;
    display: block;
}}
.footer {{
    text-align: center;
    padding: 20px;
    color: {MUTED};
    font-size: 11px;
    border-top: 1px solid #2e2e2e;
    margin-top: 20px;
}}
.tab-content {{ padding: 0; }}
.insights-box {{
    background: {CARD_BG};
    border-radius: 8px;
    border: 1px solid #2e2e2e;
    padding: 20px 28px;
    margin: 12px 24px;
}}
.insights-box h4 {{
    color: {NETFLIX_RED};
    margin: 0 0 8px 0;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
.insights-box ul {{
    margin: 0;
    padding-left: 20px;
    color: #cccccc;
    font-size: 13px;
    line-height: 1.8;
}}
"""

app.layout = html.Div([
    # CSS injection
    html.Style(styles),

    # ── Header ──
    html.Div(className="header", children=[
        html.Div([
            html.Span("N", style={"color": NETFLIX_RED, "fontWeight": "900",
                                  "fontSize": "28px", "fontStyle": "italic"}),
            html.Span("etflix", style={"fontWeight": "900", "fontSize": "24px"}),
        ]),
        html.H1("Content Intelligence Dashboard"),
        html.Span("BI PROJECT · IBM AICTE", className="badge"),
    ]),

    # ── Filter Bar ──
    html.Div(className="filter-bar", children=[
        html.Div([
            html.Label("Content Type"),
            dcc.Dropdown(
                id="filter-type",
                options=[{"label": "All", "value": "All"},
                         {"label": "Movie", "value": "Movie"},
                         {"label": "TV Show", "value": "TV Show"}],
                value="All",
                clearable=False,
                style={"width": "160px", "backgroundColor": "#2e2e2e",
                       "color": "#000", "border": "none"},
            ),
        ]),
        html.Div([
            html.Label("Rating"),
            dcc.Dropdown(
                id="filter-rating",
                options=[{"label": "All Ratings", "value": "All"}] +
                        [{"label": r, "value": r} for r in
                         df["rating"].value_counts().head(10).index],
                value="All",
                clearable=False,
                style={"width": "160px", "backgroundColor": "#2e2e2e",
                       "color": "#000", "border": "none"},
            ),
        ]),
        html.Div([
            html.Label("Year Added"),
            dcc.RangeSlider(
                id="filter-year",
                min=2008, max=2021, step=1,
                value=[2015, 2021],
                marks={y: str(y) for y in range(2008, 2022, 2)},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ], style={"flex": "2", "minWidth": "300px"}),
    ]),

    # ── Tabs ──
    dcc.Tabs(
        id="tabs",
        value="overview",
        style={"backgroundColor": "#0a0a0a", "borderBottom": f"2px solid {NETFLIX_RED}"},
        colors={"border": "#2e2e2e", "primary": NETFLIX_RED, "background": "#0a0a0a"},
        children=[
            dcc.Tab(label="📊 Overview", value="overview",
                    style={"color": MUTED, "backgroundColor": "#0a0a0a"},
                    selected_style={"color": TEXT_COLOR, "backgroundColor": DARK_BG,
                                    "borderTop": f"2px solid {NETFLIX_RED}"}),
            dcc.Tab(label="🌍 Geography", value="geography",
                    style={"color": MUTED, "backgroundColor": "#0a0a0a"},
                    selected_style={"color": TEXT_COLOR, "backgroundColor": DARK_BG,
                                    "borderTop": f"2px solid {NETFLIX_RED}"}),
            dcc.Tab(label="🎬 Content", value="content",
                    style={"color": MUTED, "backgroundColor": "#0a0a0a"},
                    selected_style={"color": TEXT_COLOR, "backgroundColor": DARK_BG,
                                    "borderTop": f"2px solid {NETFLIX_RED}"}),
            dcc.Tab(label="📈 Trends", value="trends",
                    style={"color": MUTED, "backgroundColor": "#0a0a0a"},
                    selected_style={"color": TEXT_COLOR, "backgroundColor": DARK_BG,
                                    "borderTop": f"2px solid {NETFLIX_RED}"}),
            dcc.Tab(label="💡 Insights", value="insights",
                    style={"color": MUTED, "backgroundColor": "#0a0a0a"},
                    selected_style={"color": TEXT_COLOR, "backgroundColor": DARK_BG,
                                    "borderTop": f"2px solid {NETFLIX_RED}"}),
            dcc.Tab(label="🔍 Explorer", value="explorer",
                    style={"color": MUTED, "backgroundColor": "#0a0a0a"},
                    selected_style={"color": TEXT_COLOR, "backgroundColor": DARK_BG,
                                    "borderTop": f"2px solid {NETFLIX_RED}"}),
        ],
    ),

    html.Div(id="tab-content", className="tab-content"),

    # ── Footer ──
    html.Div(className="footer", children=[
        "Netflix Content Intelligence Dashboard · IBM AICTE Business Intelligence Project  |  "
        "Dataset: Netflix Movies and TV Shows — Kaggle  |  "
        "Built with Dash & Plotly"
    ]),
])


# ─────────────────────────────────────────────
# 5. CALLBACKS
# ─────────────────────────────────────────────

def apply_filters(type_val, rating_val, year_range):
    """Return filtered dataframe based on sidebar controls."""
    filtered = df.copy()
    if type_val != "All":
        filtered = filtered[filtered["type"] == type_val]
    if rating_val != "All":
        filtered = filtered[filtered["rating"] == rating_val]
    if year_range:
        filtered = filtered[
            (filtered["year_added"] >= year_range[0]) &
            (filtered["year_added"] <= year_range[1])
        ]
    return filtered


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("filter-type", "value"),
    Input("filter-rating", "value"),
    Input("filter-year", "value"),
)
def render_tab(tab, type_val, rating_val, year_range):
    fdf = apply_filters(type_val, rating_val, year_range)

    if tab == "overview":
        return render_overview(fdf)
    elif tab == "geography":
        return render_geography(fdf)
    elif tab == "content":
        return render_content(fdf)
    elif tab == "trends":
        return render_trends(fdf)
    elif tab == "insights":
        return render_insights()
    elif tab == "explorer":
        return render_explorer(fdf)
    return html.Div()


# ─────────────────────────────────────────────
# 6. TAB RENDERERS
# ─────────────────────────────────────────────

def render_overview(fdf):
    total    = len(fdf)
    movies   = len(fdf[fdf["type"] == "Movie"])
    shows    = len(fdf[fdf["type"] == "TV Show"])
    c_count  = fdf["primary_country"].nunique()
    g_count  = fdf["listed_in"].str.split(",").explode().str.strip().nunique()

    # Recalculate charts with filtered data
    counts = fdf["type"].value_counts()
    donut = go.Figure(go.Pie(
        labels=counts.index.tolist(), values=counts.values.tolist(),
        hole=0.6, marker=dict(colors=[NETFLIX_RED, "#4fc3f7"]),
        textfont=dict(color=TEXT_COLOR),
    ))
    donut.update_traces(textinfo="label+percent",
                        hovertemplate="%{label}: %{value}<extra></extra>")
    donut = chart_layout(donut, "Content Type Split")

    top = (
        fdf[fdf["primary_country"] != "Unknown"]["primary_country"]
        .value_counts().head(10).sort_values()
    )
    bar_c = px.bar(
        x=top.values, y=top.index, orientation="h",
        color=top.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Titles", "y": "Country"},
    )
    bar_c.update_coloraxes(showscale=False)
    bar_c = chart_layout(bar_c, "Top 10 Countries")

    genres = (
        fdf["listed_in"].str.split(",").explode().str.strip()
        .value_counts().head(10).sort_values()
    )
    bar_g = px.bar(
        x=genres.values, y=genres.index, orientation="h",
        color=genres.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Titles", "y": "Genre"},
    )
    bar_g.update_coloraxes(showscale=False)
    bar_g = chart_layout(bar_g, "Top 10 Genres")

    rc = fdf["rating"].value_counts().head(8)
    bar_r = px.bar(
        x=rc.index, y=rc.values,
        color=rc.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Rating", "y": "Titles"},
    )
    bar_r.update_coloraxes(showscale=False)
    bar_r = chart_layout(bar_r, "Ratings Distribution")

    return html.Div([
        html.Div(className="kpi-row", children=[
            kpi_card("Total Titles",   f"{total:,}", f"filtered from {total_titles:,}"),
            kpi_card("Movies",         f"{movies:,}", f"{movies/total*100:.1f}% of total" if total else ""),
            kpi_card("TV Shows",       f"{shows:,}", f"{shows/total*100:.1f}% of total" if total else ""),
            kpi_card("Countries",      f"{c_count:,}", "unique countries", color="#4fc3f7"),
            kpi_card("Genres",         f"{g_count:,}", "unique genres",    color="#81c784"),
        ]),
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=donut, config={"displayModeBar": False})]),
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=bar_r, config={"displayModeBar": False})]),
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=bar_c, config={"displayModeBar": False})]),
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=bar_g, config={"displayModeBar": False})]),
        ]),
    ])


def render_geography(fdf):
    country_counts = (
        fdf[fdf["primary_country"] != "Unknown"]["primary_country"]
        .value_counts().reset_index()
    )
    country_counts.columns = ["country", "count"]
    world = px.choropleth(
        country_counts, locations="country", locationmode="country names",
        color="count", color_continuous_scale=[[0, "#1e1e1e"], [0.3, "#b71c1c"], [1, NETFLIX_RED]],
        labels={"count": "Titles"},
    )
    world.update_geos(bgcolor=DARK_BG, lakecolor=DARK_BG, landcolor="#2e2e2e",
                      showcoastlines=True, coastlinecolor="#444")
    world.update_coloraxes(colorbar_tickfont=dict(color=TEXT_COLOR))
    world = chart_layout(world, "Global Netflix Content Map", height=450)

    top15 = (
        fdf[fdf["primary_country"] != "Unknown"]["primary_country"]
        .value_counts().head(15).reset_index()
    )
    top15.columns = ["country", "count"]
    top15["pct"] = (top15["count"] / top15["count"].sum() * 100).round(1)
    bar = px.bar(
        top15, x="country", y="count",
        color="count", color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"country": "", "count": "Titles"},
        text=top15["pct"].astype(str) + "%",
    )
    bar.update_traces(textposition="outside", textfont=dict(color=MUTED, size=10))
    bar.update_coloraxes(showscale=False)
    bar = chart_layout(bar, "Top 15 Countries by Number of Titles")

    # Country type breakdown
    top5 = (
        fdf[fdf["primary_country"] != "Unknown"]["primary_country"]
        .value_counts().head(5).index.tolist()
    )
    ct = (
        fdf[fdf["primary_country"].isin(top5)]
        .groupby(["primary_country", "type"]).size().reset_index(name="count")
    )
    stacked = px.bar(
        ct, x="primary_country", y="count", color="type", barmode="stack",
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        labels={"primary_country": "Country", "count": "Titles", "type": ""},
    )
    stacked = chart_layout(stacked, "Movies vs TV Shows — Top 5 Countries")

    return html.Div([
        html.Div(className="chart-grid",
                 children=[html.Div(className="chart-full",
                                    children=[dcc.Graph(figure=world, config={"displayModeBar": False})])]),
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-full",
                     children=[dcc.Graph(figure=bar, config={"displayModeBar": False})]),
        ]),
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=stacked, config={"displayModeBar": False})]),
        ]),
    ])


def render_content(fdf):
    genres = (
        fdf["listed_in"].str.split(",").explode().str.strip()
        .value_counts().head(15).sort_values()
    )
    bar_g = px.bar(
        x=genres.values, y=genres.index, orientation="h",
        color=genres.values, color_continuous_scale=[[0, "#2e2e2e"], [1, NETFLIX_RED]],
        labels={"x": "Titles", "y": "Genre"},
    )
    bar_g.update_coloraxes(showscale=False)
    bar_g = chart_layout(bar_g, "Top 15 Genres (All Content)", height=430)

    movies = fdf[(fdf["type"] == "Movie") & (fdf["duration_value"].notna())]
    avg = movies["duration_value"].mean()
    hist = px.histogram(
        movies, x="duration_value", nbins=40, color_discrete_sequence=[NETFLIX_RED],
        labels={"duration_value": "Duration (minutes)"},
    )
    hist.add_vline(x=avg, line_dash="dash", line_color="#f5c518",
                   annotation_text=f"Avg: {avg:.0f} min",
                   annotation_font_color="#f5c518")
    hist = chart_layout(hist, "Movie Duration Distribution")

    shows = fdf[(fdf["type"] == "TV Show") & (fdf["duration_value"].notna())]
    sc = shows["duration_value"].value_counts().sort_index().head(10)
    bar_s = px.bar(
        x=sc.index.astype(int), y=sc.values,
        color=sc.values, color_continuous_scale=[[0, "#2e2e2e"], [1, "#4fc3f7"]],
        labels={"x": "Seasons", "y": "TV Shows"},
    )
    bar_s.update_coloraxes(showscale=False)
    bar_s = chart_layout(bar_s, "TV Shows by Season Count")

    rt = (
        fdf[fdf["rating"].isin(fdf["rating"].value_counts().head(8).index)]
        .groupby(["rating", "type"]).size().reset_index(name="count")
    )
    stacked_r = px.bar(
        rt, x="rating", y="count", color="type", barmode="stack",
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        labels={"rating": "Rating", "count": "Titles", "type": ""},
    )
    stacked_r = chart_layout(stacked_r, "Rating Distribution by Content Type")

    return html.Div([
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=bar_g, config={"displayModeBar": False})]),
            html.Div(className="chart-half", children=[
                dcc.Graph(figure=hist,     config={"displayModeBar": False}),
                dcc.Graph(figure=bar_s,    config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=stacked_r, config={"displayModeBar": False})]),
        ]),
    ])


def render_trends(fdf):
    # Yearly additions
    yearly = fdf.groupby(["year_added", "type"]).size().reset_index(name="count")
    yearly = yearly[yearly["year_added"].notna() & (yearly["year_added"] >= 2010)]
    line = px.line(
        yearly, x="year_added", y="count", color="type", markers=True,
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        labels={"year_added": "Year", "count": "Titles Added", "type": ""},
    )
    line = chart_layout(line, "Yearly Content Additions (2010–2021)")

    # Decade bar
    dc = fdf.groupby(["decade", "type"]).size().reset_index(name="count")
    bar_d = px.bar(
        dc, x="decade", y="count", color="type", barmode="group",
        color_discrete_map={"Movie": NETFLIX_RED, "TV Show": "#4fc3f7"},
        labels={"decade": "Decade", "count": "Titles", "type": ""},
    )
    bar_d = chart_layout(bar_d, "Content by Release Decade")

    # Monthly heatmap
    monthly = (
        fdf[fdf["year_added"].notna() & (fdf["year_added"] >= 2015)]
        .groupby(["year_added", "month_added"]).size().reset_index(name="count")
    )
    pivot = monthly.pivot(index="month_added", columns="year_added", values="count").fillna(0)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    hmap = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(int(c)) for c in pivot.columns],
        y=month_labels[:len(pivot.index)],
        colorscale=[[0, "#141414"], [0.5, "#b71c1c"], [1, NETFLIX_RED]],
        hoverongaps=False,
    ))
    hmap = chart_layout(hmap, "Monthly Addition Heatmap (2015–2021)", height=350)

    # Cumulative growth
    yearly_total = (
        fdf[fdf["year_added"].notna() & (fdf["year_added"] >= 2010)]
        .groupby("year_added").size().reset_index(name="count")
        .sort_values("year_added")
    )
    yearly_total["cumulative"] = yearly_total["count"].cumsum()
    area = px.area(
        yearly_total, x="year_added", y="cumulative",
        color_discrete_sequence=[NETFLIX_RED],
        labels={"year_added": "Year", "cumulative": "Cumulative Titles"},
    )
    area.update_traces(fillcolor="rgba(229,9,20,0.2)")
    area = chart_layout(area, "Cumulative Catalog Growth")

    return html.Div([
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=line, config={"displayModeBar": False})]),
            html.Div(className="chart-half",
                     children=[dcc.Graph(figure=area, config={"displayModeBar": False})]),
        ]),
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-full",
                     children=[dcc.Graph(figure=hmap, config={"displayModeBar": False})]),
        ]),
        html.Div(className="chart-grid", children=[
            html.Div(className="chart-full",
                     children=[dcc.Graph(figure=bar_d, config={"displayModeBar": False})]),
        ]),
    ])


def render_insights():
    sections = [
        ("📌 Key Performance Indicators", [
            f"Total catalog: {total_titles:,} titles across {total_countries:,} countries and {total_genres:,} genres.",
            f"Movies dominate: {total_movies:,} ({total_movies/total_titles*100:.1f}%) vs {total_shows:,} TV Shows ({total_shows/total_titles*100:.1f}%).",
            f"Average movie runtime: {avg_movie_dur:.0f} minutes.",
            f"Most common rating: TV-MA (3,207 titles) — majority of content is for mature audiences.",
        ]),
        ("📈 Trends & Growth", [
            "Catalog growth accelerated sharply between 2015 and 2020, peaking at ~1,500+ additions in 2019.",
            "October, November, and December show consistently high content additions — aligning with Q4 subscriber acquisition drives.",
            "TV Show percentage has grown relative to Movies in recent years, reflecting the streaming pivot toward episodic content.",
            "2020–2021 shows a slight slowdown — likely attributable to COVID-19 production disruptions.",
        ]),
        ("🌍 Geographic Insights", [
            "The United States contributes 3,690 titles (41.9% of the catalog), followed by India (1,046), UK (806), and Canada (445).",
            "India has the second-largest content volume, indicating Netflix's strong push into the South Asian market.",
            "South Korea and Japan are rapidly growing contributors, driven by K-Drama and anime popularity.",
            "Germany, France, and Spain signal a strong European content investment strategy.",
        ]),
        ("🎬 Content & Genre Drivers", [
            "Top genre: International Movies (2,752 titles) — confirming Netflix's global content strategy.",
            "Dramas (2,427) and Comedies (1,674) are the most popular storytelling formats globally.",
            "Children & Family Movies (641) and Documentaries (869) serve important niche subscriber segments.",
            "Most TV Shows have only 1 season (limited series trend) — reducing long-term production risk.",
        ]),
        ("⚠️ Risks & Gaps", [
            "29.9% of titles have no director information — data quality needs improvement for recommendation accuracy.",
            "9.4% of titles are missing country data — geographic attribution incomplete.",
            "High concentration (US + India + UK = ~62% of catalog) creates dependency risk if licensing agreements change.",
            "Historical content (pre-2000) makes up a small fraction — limited classic library depth.",
        ]),
        ("💡 Opportunities", [
            "Expand Korean, Japanese, and Spanish-language original productions — fastest growing non-English segments.",
            "Invest in multi-season TV shows — single-season dominance may indicate under-investment in serialized storytelling.",
            "Children & Family and Documentary genres are underrepresented vs global demand — high opportunity segments.",
            "A content recommendation model trained on genres, ratings, and country can improve subscriber personalization.",
        ]),
    ]

    return html.Div([
        html.Div(className="insights-box", children=[
            html.H4(title),
            html.Ul([html.Li(point) for point in points]),
        ])
        for title, points in sections
    ])


def render_explorer(fdf):
    display_cols = ["title", "type", "director", "primary_country",
                    "release_year", "rating", "duration", "primary_genre"]
    table_data = fdf[display_cols].head(200).fillna("–").to_dict("records")

    return html.Div([
        html.P(
            f"Showing top 200 of {len(fdf):,} filtered records. "
            "Use the filters above to narrow down the dataset.",
            style={"padding": "12px 24px", "color": MUTED, "fontSize": "12px", "margin": 0},
        ),
        html.Div(style={"padding": "0 24px 20px"}, children=[
            dash_table.DataTable(
                data=table_data,
                columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={
                    "backgroundColor": CARD_BG, "color": TEXT_COLOR,
                    "border": "1px solid #2e2e2e", "padding": "8px 12px",
                    "fontSize": "12px", "textAlign": "left",
                    "maxWidth": "200px", "overflow": "hidden",
                    "textOverflow": "ellipsis",
                },
                style_header={
                    "backgroundColor": "#0a0a0a", "color": NETFLIX_RED,
                    "fontWeight": "bold", "border": "1px solid #2e2e2e",
                    "textTransform": "uppercase", "fontSize": "11px",
                    "letterSpacing": "0.6px",
                },
                style_filter={"backgroundColor": "#2e2e2e", "color": TEXT_COLOR},
                style_data_conditional=[
                    {"if": {"row_index": "odd"},
                     "backgroundColor": "#181818"},
                    {"if": {"filter_query": '{type} = "Movie"'},
                     "borderLeft": f"3px solid {NETFLIX_RED}"},
                    {"if": {"filter_query": '{type} = "TV Show"'},
                     "borderLeft": "3px solid #4fc3f7"},
                ],
            ),
        ]),
    ])


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Netflix Content Intelligence Dashboard")
    print("  IBM AICTE Business Intelligence Project")
    print("="*55)
    print(f"  Dataset loaded: {total_titles:,} titles")
    print(f"  Movies: {total_movies:,}  |  TV Shows: {total_shows:,}")
    print(f"  Countries: {total_countries:,}  |  Genres: {total_genres:,}")
    print("="*55)
    print("  Open browser → http://127.0.0.1:8050")
    print("="*55 + "\n")
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
