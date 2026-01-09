
# wino.py
# Streamlit app for exploratory analysis of two datasets:
# - winequality-red.csv
# - wine_food_pairings.csv

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_TITLE = "Wino – analiza datasetów (Streamlit)"

DEFAULT_RED_PATH = Path("winequality-red.csv")
DEFAULT_PAIRINGS_PATH = Path("wine_food_pairings.csv")

# Folders to search for datasets
SEARCH_DIRS = [
    Path("."),
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
    for base in SEARCH_DIRS:
        candidate = base / filename
        if candidate.exists():
            return candidate.resolve()

    for base in SEARCH_DIRS:
        if base.exists() and base.is_dir():
            for p in base.rglob(filename):
                try:
                    if len(p.relative_to(base).parts) <= max_depth + 1:
                        return p.resolve()
                except Exception:
                    return p.resolve()
    return None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")


@st.cache_data(show_spinner=False)
def load_dataset(default_path: Path, upload_key: str) -> pd.DataFrame:
    if default_path.exists():
        return _safe_read_csv(default_path)

    found = find_dataset_file(default_path.name)
    if found:
        st.sidebar.success(f"Znaleziono {default_path.name}: {found.as_posix()}")
        return _safe_read_csv(found)

    st.sidebar.warning(f"Nie znaleziono {default_path.name}. Wgraj plik ręcznie.")
    uploaded = st.sidebar.file_uploader(
        f"Wgraj {default_path.name}", type=["csv"], key=upload_key
    )
    if uploaded is None:
        st.stop()
    return pd.read_csv(uploaded)


def render_basic_eda(df: pd.DataFrame, title: str) -> None:
    st.subheader(title)

    c1, c2, c3 = st.columns(3)
    c1.metric("Wiersze", df.shape[0])
    c2.metric("Kolumny", df.shape[1])
    c3.metric("Duplikaty", df.duplicated().sum())

    st.markdown("**Podgląd danych**")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("**Typy danych**")
    st.dataframe(
        pd.DataFrame({"kolumna": df.columns, "typ": df.dtypes.astype(str)}),
        use_container_width=True,
    )

    st.markdown("**Braki danych**")
    na = df.isna().sum()
    na = na[na > 0]
    if na.empty:
        st.success("Brak brakujących wartości.")
    else:
        st.dataframe(
            pd.DataFrame(
                {"kolumna": na.index, "braki": na.values, "%": (na / len(df) * 100).round(2)}
            ),
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.sidebar.header("Źródła danych")

    red = load_dataset(DEFAULT_RED_PATH, "red_upload")
    pairings = load_dataset(DEFAULT_PAIRINGS_PATH, "pairings_upload")

    tab1, tab2 = st.tabs(["winequality-red.csv", "wine_food_pairings.csv"])
    with tab1:
        render_basic_eda(red, "Podstawowa eksploracja – winequality-red.csv")
    with tab2:
        render_basic_eda(pairings, "Podstawowa eksploracja – wine_food_pairings.csv")


if __name__ == "__main__":
    main()
