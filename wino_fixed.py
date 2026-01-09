
# wino.py
# Streamlit app for exploratory analysis of two datasets:
# - winequality-red.csv
# - wine_food_pairings.csv
#
# Requirements covered:
# 1) Basic EDA for both datasets (head, shape, dtypes, missing values, duplicates)
# 2) Filtering + quick insights
# 3) Distributions + comparisons for winequality-red
# 4) 3D plots (winequality-red)

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Wino – analiza datasetów (Streamlit)"
DEFAULT_RED_PATH = Path("winequality-red.csv")
DEFAULT_PAIRINGS_PATH = Path\(\"wine_food_pairings\.csv\"\)

# Folders to search for datasets (supports keeping CSV files in subfolders)
SEARCH_DIRS = [
    Path("."),            # project root
    Path("data"),
    Path("datasets"),
    Path("dataset"),
    Path("input"),
    Path("assets"),
    Path("files"),
    Path("data/raw"),
    Path("data/processed"),
]

def find_dataset_file(filename: str, max_depth: int = 3) -> Optional[Path]:
    """
    Try to locate `filename` inside common project folders.
    Searches:
      - exact paths in SEARCH_DIRS
      - recursive search up to `max_depth` within each SEARCH_DIR
    Returns a Path if found, else None.
    """
    fn = filename

    # 1) direct candidates
    for base in SEARCH_DIRS:
        p = base / fn
        if p.exists():
            return p.resolve()

    # 2) recursive search
    for base in SEARCH_DIRS:
        if not base.exists() or not base.is_dir():
            continue
        try:
            for p in base.rglob(fn):
                # limit depth relative to base
                try:
                    rel = p.relative_to(base)
                    if len(rel.parts) <= max_depth + 1:  # filename counts as a part
                        return p.resolve()
                except Exception:
                    return p.resolve()
        except Exception:
            continue

    return None



# -----------------------------
# Utilities
# -----------------------------
def _safe_read_csv(path: Path) -> pd.DataFrame:
    # Common separators for these datasets are comma or semicolon.
    # Try comma first, then semicolon.
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def _missing_table(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if miss.empty:
        return pd.DataFrame({"column": [], "missing_count": [], "missing_pct": []})
    pct = (miss / len(df) * 100).round(2)
    return pd.DataFrame({"column": miss.index, "missing_count": miss.values, "missing_pct": pct.values})


def _basic_stats(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame({"stat": [], "value": []})

    stats = {}
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        stats[c] = {
            "mean": float(s.mean()),
            "median": float(s.median()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    if not stats:
        return pd.DataFrame({"stat": [], "value": []})

    # Flatten to display: column, mean, median, min, max
    out = []
    for c, d in stats.items():
        out.append({"column": c, **{k: round(v, 4) for k, v in d.items()}})
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def load_red(source: str, uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return _safe_read_csv(Path(source))


@st.cache_data(show_spinner=False)
def load_pairings(source: str, uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return _safe_read_csv(Path(source))


def render_basic_eda(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Wiersze", f"{df.shape[0]:,}".replace(",", " "))
    with c2:
        st.metric("Kolumny", f"{df.shape[1]}")
    with c3:
        st.metric("Duplikaty", f"{df.duplicated().sum():,}".replace(",", " "))

    st.markdown("**Podgląd danych (head)**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Typy danych**")
    dtypes_df = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
    st.dataframe(dtypes_df, use_container_width=True, height=280)

    st.markdown("**Brakujące wartości (ile i gdzie)**")
    miss_df = _missing_table(df)
    if miss_df.empty:
        st.success("Brak brakujących wartości.")
    else:
        st.dataframe(miss_df, use_container_width=True, height=280)


def _relative_or_uploaded(default_path: Path, upload_key: str) -> Tuple[str, Optional[object]]:
    """
    Returns (source_str, uploaded_file).

    Behavior:
    - If file exists at `default_path` (e.g., same folder as app), use it.
    - Else search common subfolders (data/, datasets/, etc.) via find_dataset_file().
    - Else allow upload via Streamlit uploader.
    """
    # 0) exact path as provided
    if default_path.exists():
        return str(default_path), None

    # 1) search in common project subfolders
    found = find_dataset_file(default_path.name)
    if found is not None and found.exists():
        st.sidebar.success(f"Znaleziono {default_path.name}: {found.as_posix()}")
        return str(found), None

    # 2) optional: user-provided relative folder
    st.sidebar.info(f"Nie znaleziono {default_path.name} w standardowych folderach. Możesz wskazać folder lub wgrać plik.")
    folder = st.sidebar.text_input(
        f"Opcjonalnie: folder dla {default_path.name} (np. data/)",
        value="",
        key=f"{upload_key}_folder",
        help="Podaj ścieżkę względną względem katalogu uruchomienia Streamlit.",
    ).strip()

    if folder:
        candidate = Path(folder) / default_path.name
        if candidate.exists():
            st.sidebar.success(f"Znaleziono {default_path.name}: {candidate.as_posix()}")
            return str(candidate), None
        else:
            st.sidebar.warning(f"W folderze '{folder}' nie ma {default_path.name}.")

    # 3) upload fallback
    st.warning(f"Nie znaleziono pliku {default_path.name} w katalogu aplikacji ani w folderach danych. Wgraj plik ręcznie.")
    up = st.file_uploader(f"Wgraj {default_path.name}", type=["csv"], key=upload_key)
    if up is None:
        st.stop()
    return default_path.name, up


def _sidebar_dataset_paths() -> Tuple[Tuple[str, Optional[object]], Tuple[str, Optional[object]]]:
    st.sidebar.header("Źródła danych")
    red_src, red_up = _relative_or_uploaded(DEFAULT_RED_PATH, "upload_red")
    pair_src, pair_up = _relative_or_uploaded(DEFAULT_PAIRINGS_PATH, "upload_pairings")
    return (red_src, red_up), (pair_src, pair_up)


# -----------------------------
# Filtering modules
# -----------------------------
def filter_red(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filtrowanie – winequality-red.csv")

    if "quality" not in df.columns:
        st.error("Brak kolumny 'quality' w winequality-red.csv – nie można wykonać filtrowania po jakości.")
        return df

    # Quality filter
    q_min, q_max = int(df["quality"].min()), int(df["quality"].max())
    q_range = st.slider("Zakres quality", min_value=q_min, max_value=q_max, value=(q_min, q_max))

    # Feature filter
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_candidates = [c for c in numeric_cols if c != "quality"]
    if not feature_candidates:
        st.warning("Brak numerycznych cech do filtrowania (poza quality).")
        return df[(df["quality"] >= q_range[0]) & (df["quality"] <= q_range[1])]

    feature = st.selectbox("Wybrana cecha (suwak zakresu)", feature_candidates, index=0)

    f_min = float(np.nanmin(df[feature].values))
    f_max = float(np.nanmax(df[feature].values))
    if np.isfinite(f_min) and np.isfinite(f_max) and f_min != f_max:
        f_range = st.slider(
            f"Zakres {feature}",
            min_value=float(f_min),
            max_value=float(f_max),
            value=(float(f_min), float(f_max)),
        )
    else:
        f_range = (f_min, f_max)

    filtered = df[
        (df["quality"] >= q_range[0])
        & (df["quality"] <= q_range[1])
        & (df[feature] >= f_range[0])
        & (df[feature] <= f_range[1])
    ].copy()

    st.markdown("### Wyniki filtrów")
    st.write(f"Pozostało rekordów: **{len(filtered):,}**".replace(",", " "))

    st.dataframe(filtered, use_container_width=True, height=420)

    st.markdown("### Szybkie statystyki (2–3)")
    stat_cols = [feature, "quality"]
    # Add one extra commonly useful column if available (alcohol is typical)
    if "alcohol" in df.columns and "alcohol" not in stat_cols:
        stat_cols.append("alcohol")

    stats_df = _basic_stats(filtered, stat_cols)
    if stats_df.empty:
        st.info("Brak danych do wyliczenia statystyk (po filtrach).")
    else:
        st.dataframe(stats_df, use_container_width=True)

    return filtered


def filter_pairings(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filtrowanie – wine_food_pairings.csv")

    # Flexible column naming: try a few common variants
    def pick_col(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    col_wine_type = pick_col(["wine_type", "type", "wine"])
    col_food_cat = pick_col(["food_category", "category", "food"])
    col_cuisine = pick_col(["cuisine", "region"])
    col_pair_q = pick_col(["pairing_quality", "quality", "score", "rating"])

    if col_pair_q is None:
        st.warning("Nie znaleziono kolumny oceny dopasowania (np. 'pairing_quality'). Filtr minimalnej jakości zostanie pominięty.")

    # Build filters (use multiselects)
    def ms_filter(label: str, col: Optional[str]) -> List[str]:
        if col is None:
            return []
        vals = df[col].dropna().astype(str).unique().tolist()
        vals_sorted = sorted(vals)
        return st.multiselect(label, vals_sorted, default=[])

    f_wine = ms_filter("wine_type", col_wine_type)
    f_food = ms_filter("food_category", col_food_cat)
    f_cuisine = ms_filter("cuisine", col_cuisine)

    filtered = df.copy()

    if col_wine_type and f_wine:
        filtered = filtered[filtered[col_wine_type].astype(str).isin(f_wine)]
    if col_food_cat and f_food:
        filtered = filtered[filtered[col_food_cat].astype(str).isin(f_food)]
    if col_cuisine and f_cuisine:
        filtered = filtered[filtered[col_cuisine].astype(str).isin(f_cuisine)]

    if col_pair_q and pd.api.types.is_numeric_dtype(df[col_pair_q]):
        qmin = float(np.nanmin(df[col_pair_q].values))
        qmax = float(np.nanmax(df[col_pair_q].values))
        min_q = st.slider("Minimalna pairing_quality", min_value=float(qmin), max_value=float(qmax), value=float(qmin))
        filtered = filtered[filtered[col_pair_q] >= min_q]
    elif col_pair_q:
        # Non-numeric, fallback: choose allowed values by order
        vals = sorted(filtered[col_pair_q].dropna().astype(str).unique().tolist())
        min_sel = st.selectbox("Minimalna pairing_quality (tekstowo)", vals, index=0)
        filtered = filtered[filtered[col_pair_q].astype(str) >= str(min_sel)]

    st.markdown("### Wyniki filtrów")
    st.write(f"Pozostało rekordów: **{len(filtered):,}**".replace(",", " "))

    st.dataframe(filtered, use_container_width=True, height=420)

    st.markdown("### Szybkie statystyki (2–3)")
    # Pick up to 3 numeric columns (prioritize rating, then any others)
    numeric_cols = [c for c in filtered.columns if pd.api.types.is_numeric_dtype(filtered[c])]
    stat_cols: List[str] = []
    if col_pair_q and col_pair_q in numeric_cols:
        stat_cols.append(col_pair_q)
    for c in numeric_cols:
        if c not in stat_cols:
            stat_cols.append(c)
        if len(stat_cols) >= 3:
            break

    stats_df = _basic_stats(filtered, stat_cols)
    if stats_df.empty:
        st.info("Brak numerycznych kolumn do statystyk po filtrach.")
    else:
        st.dataframe(stats_df, use_container_width=True)

    return filtered


# -----------------------------
# Distributions + comparisons
# -----------------------------
def distributions_red(df: pd.DataFrame) -> None:
    st.subheader("Rozkłady i porównania – winequality-red.csv")

    if "quality" not in df.columns:
        st.error("Brak kolumny 'quality' – nie można wykonać porównań po jakości.")
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    feature_candidates = [c for c in numeric_cols if c != "quality"]
    if not feature_candidates:
        st.warning("Brak numerycznych cech do analizy rozkładów.")
        return

    feature = st.selectbox("Wybierz cechę", feature_candidates, index=0)

    # Histogram
    st.markdown("#### Histogram")
    fig_hist = px.histogram(df, x=feature, nbins=30, marginal="rug")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Boxplot
    st.markdown("#### Boxplot")
    fig_box = px.box(df, y=feature, points="outliers")
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()
    st.markdown("### Porównanie rozkładu dla dwóch grup jakości")

    mode = st.radio(
        "Tryb porównania",
        options=["quality ≤ X vs quality > X", "quality = A vs quality = B"],
        horizontal=True,
    )

    if mode == "quality ≤ X vs quality > X":
        q_min, q_max = int(df["quality"].min()), int(df["quality"].max())
        x = st.slider("Wybierz X", min_value=q_min, max_value=q_max, value=int(np.median(df["quality"])))
        g1 = df[df["quality"] <= x].copy()
        g2 = df[df["quality"] > x].copy()
        g1["group"] = f"quality ≤ {x}"
        g2["group"] = f"quality > {x}"
    else:
        qualities = sorted(df["quality"].dropna().unique().tolist())
        if len(qualities) < 2:
            st.info("Za mało różnych wartości quality do porównania.")
            return
        a = st.selectbox("A (quality)", qualities, index=0)
        b = st.selectbox("B (quality)", qualities, index=min(1, len(qualities) - 1))
        g1 = df[df["quality"] == a].copy()
        g2 = df[df["quality"] == b].copy()
        g1["group"] = f"quality = {a}"
        g2["group"] = f"quality = {b}"

    comp = pd.concat([g1, g2], ignore_index=True)

    # Comparison histogram (overlay)
    st.markdown("#### Histogram porównawczy")
    fig_comp_hist = px.histogram(comp, x=feature, color="group", nbins=30, barmode="overlay", opacity=0.6)
    st.plotly_chart(fig_comp_hist, use_container_width=True)

    # Comparison boxplot
    st.markdown("#### Boxplot porównawczy")
    fig_comp_box = px.box(comp, x="group", y=feature, points="outliers")
    st.plotly_chart(fig_comp_box, use_container_width=True)


# -----------------------------
# 3D plots (winequality-red)
# -----------------------------
def plots_3d_red(df: pd.DataFrame) -> None:
    st.subheader("Wykresy 3D – winequality-red.csv")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 3:
        st.warning("Za mało numerycznych kolumn do wykresu 3D.")
        return

    default_x = "alcohol" if "alcohol" in numeric_cols else numeric_cols[0]
    default_y = "volatile acidity" if "volatile acidity" in numeric_cols else numeric_cols[1]
    default_z = "sulphates" if "sulphates" in numeric_cols else numeric_cols[2]

    c1, c2, c3 = st.columns(3)
    with c1:
        x = st.selectbox("Oś X", numeric_cols, index=numeric_cols.index(default_x))
    with c2:
        y = st.selectbox("Oś Y", numeric_cols, index=numeric_cols.index(default_y))
    with c3:
        z = st.selectbox("Oś Z", numeric_cols, index=numeric_cols.index(default_z))

    color_by = "quality" if "quality" in df.columns else None
    n = len(df)
    max_points = st.slider("Maks. liczba punktów (dla wydajności)", min_value=500, max_value=min(10000, max(500, n)), value=min(3000, n), step=250)
    sample = df.sample(n=max_points, random_state=42) if n > max_points else df.copy()

    st.markdown("#### Scatter 3D (Plotly)")
    fig = px.scatter_3d(
        sample,
        x=x,
        y=y,
        z=z,
        color=color_by,
        opacity=0.75,
        height=650,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Optional: 3D surface-like view using binned averages (lightweight, avoids heavy interpolation)
    st.markdown("#### (Opcjonalnie) Mapa 3D: średnia quality w siatce X–Y")
    if "quality" in df.columns and pd.api.types.is_numeric_dtype(df["quality"]):
        bins = st.slider("Liczba koszyków (X i Y)", min_value=5, max_value=40, value=15)
        tmp = df[[x, y, "quality"]].dropna().copy()
        if len(tmp) < 50:
            st.info("Za mało danych po odrzuceniu braków do siatki 3D.")
            return

        tmp["x_bin"] = pd.cut(tmp[x], bins=bins)
        tmp["y_bin"] = pd.cut(tmp[y], bins=bins)

        grid = (
            tmp.groupby(["x_bin", "y_bin"], observed=True)["quality"]
            .mean()
            .reset_index()
        )

        # Convert bins to midpoints
        grid["x_mid"] = grid["x_bin"].apply(lambda i: (i.left + i.right) / 2)
        grid["y_mid"] = grid["y_bin"].apply(lambda i: (i.left + i.right) / 2)

        # Pivot to matrix for surface
        pivot = grid.pivot_table(index="y_mid", columns="x_mid", values="quality", aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)

        X = pivot.columns.values
        Y = pivot.index.values
        Z = pivot.values

        fig_surf = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig_surf.update_layout(height=650, scene=dict(xaxis_title=x, yaxis_title=y, zaxis_title="mean(quality)"))
        st.plotly_chart(fig_surf, use_container_width=True)
    else:
        st.info("Brak numerycznej kolumny 'quality' – pomijam mapę siatki 3D.")


# -----------------------------
# App layout
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    (red_src, red_up), (pair_src, pair_up) = _sidebar_dataset_paths()

    red = load_red(red_src, red_up)
    pairings = load_pairings(pair_src, pair_up)

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Nawigacja",
        options=[
            "EDA – oba datasety",
            "Filtrowanie + statystyki",
            "Rozkłady i porównania (winequality-red)",
            "Wykresy 3D (winequality-red)",
        ],
    )

    if page == "EDA – oba datasety":
        tab1, tab2 = st.tabs(["winequality-red.csv", "wine_food_pairings.csv"])
        with tab1:
            render_basic_eda(red, "Podstawowa eksploracja – winequality-red.csv")
        with tab2:
            render_basic_eda(pairings, "Podstawowa eksploracja – wine_food_pairings.csv")

    elif page == "Filtrowanie + statystyki":
        tab1, tab2 = st.tabs(["winequality-red.csv", "wine_food_pairings.csv"])
        with tab1:
            _ = filter_red(red)
        with tab2:
            _ = filter_pairings(pairings)

    elif page == "Rozkłady i porównania (winequality-red)":
        distributions_red(red)

    elif page == "Wykresy 3D (winequality-red)":
        plots_3d_red(red)

    st.sidebar.divider()
    st.sidebar.caption("Wskazówka: jeśli deployujesz na Streamlit Community Cloud, trzymaj pliki CSV w repozytorium obok wino.py (lub wgraj je przez uploader).")


if __name__ == "__main__":
    main()
