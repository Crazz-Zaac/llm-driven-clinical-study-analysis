import ast
from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_DIR = Path(__file__).parent / "results"

METRIC_COLS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
    "llm_context_precision_with_reference",
]

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_recall": "Context recall",
    "llm_context_precision_with_reference": "Context precision",
}

DEFAULT_CORRECTNESS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_recall",
]


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "query_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "query_id"})
        df["query_id"] = df["query_id"] + 1

    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_metrics = [col for col in METRIC_COLS if col in df.columns]
    df["mean_score"] = df[available_metrics].mean(axis=1, skipna=True)
    df["min_score"] = df[available_metrics].min(axis=1, skipna=True)
    if "correctness_score" in df.columns:
        df["correctness_score"] = pd.to_numeric(
            df["correctness_score"],
            errors="coerce",
        )
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(bool)
    df["short_query"] = df["user_input"].str.slice(0, 95)
    return df


def parse_contexts(value) -> list[str]:
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [str(value)]

    if isinstance(parsed, list):
        return [str(item) for item in parsed]

    return [str(parsed)]


def add_correctness_labels(
    df: pd.DataFrame,
    correctness_metrics: list[str],
    correctness_threshold: float,
) -> pd.DataFrame:
    labelled = df.copy()
    if "correctness_score" in labelled.columns and "is_correct" in labelled.columns:
        labelled["outcome"] = labelled["is_correct"].map(
            {True: "Correct", False: "Needs review"}
        )
        return labelled

    labelled["correctness_score"] = labelled[correctness_metrics].mean(
        axis=1,
        skipna=False,
    )
    labelled["is_correct"] = (
        labelled[correctness_metrics].ge(correctness_threshold).all(axis=1)
    )
    labelled["outcome"] = labelled["is_correct"].map(
        {True: "Correct", False: "Needs review"}
    )
    return labelled


def metric_summary(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    summary = df[metric_cols].describe().T
    summary.index = summary.index.map(lambda col: METRIC_LABELS.get(col, col))
    return summary


def show_score_pills(df: pd.DataFrame) -> None:
    correct_count = int(df["is_correct"].sum())
    total_count = len(df)
    review_count = total_count - correct_count
    accuracy = correct_count / total_count if total_count else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Queries", total_count)
    col2.metric("Correct", correct_count)
    col3.metric("Needs review", review_count)
    col4.metric("Derived accuracy", f"{accuracy:.0%}")


def show_query_outcome_chart(df: pd.DataFrame) -> None:
    chart_df = (
        df[["query_id", "correctness_score", "is_correct"]]
        .set_index("query_id")
        .rename(columns={"correctness_score": "Correctness score"})
    )
    st.bar_chart(
        chart_df,
        y="Correctness score",
        color="is_correct",
        height=320,
    )


def show_metric_heatmap(df: pd.DataFrame, metric_cols: list[str]) -> None:
    table = df[
        [
            "query_id",
            "outcome",
            "correctness_score",
            *metric_cols,
            "short_query",
        ]
    ].copy()
    table = table.rename(
        columns={
            "query_id": "Query",
            "outcome": "Outcome",
            "correctness_score": "Correctness",
            "short_query": "Question",
            **{col: METRIC_LABELS.get(col, col) for col in metric_cols},
        }
    )

    metric_labels = [METRIC_LABELS.get(col, col) for col in metric_cols]
    numeric_columns = ["Correctness", *metric_labels]
    table_height = min(520, 38 + (len(table) + 1) * 35)
    st.dataframe(
        table.style.format(
            {
                col: "{:.2f}"
                for col in numeric_columns
            },
            na_rep="n/a",
        ).background_gradient(
            subset=numeric_columns,
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
        ),
        column_config={
            "Query": st.column_config.NumberColumn("Query", width=80),
            "Outcome": st.column_config.TextColumn("Outcome", width=140),
            "Correctness": st.column_config.NumberColumn(
                "Correctness",
                width=130,
                format="%.2f",
            ),
            **{
                label: st.column_config.NumberColumn(
                    label,
                    width=150,
                    format="%.2f",
                )
                for label in metric_labels
            },
            "Question": st.column_config.TextColumn("Question", width=700),
        },
        height=table_height,
        width="stretch",
        hide_index=True,
    )


def show_example(row: pd.Series, metric_cols: list[str]) -> None:
    title = (
        f"Q{row['query_id']} | {row['outcome']} | "
        f"score {row['correctness_score']:.2f} | {row['user_input'][:90]}"
    )
    with st.expander(title):
        st.markdown("### Question")
        st.write(row["user_input"])

        st.markdown("### Generated response")
        st.write(row["response"])

        st.markdown("### Reference answer")
        st.write(row["reference"])

        st.markdown("### Retrieved contexts")
        contexts = parse_contexts(row["retrieved_contexts"])
        if contexts:
            for i, context in enumerate(contexts, start=1):
                st.markdown(f"**Context {i}**")
                st.write(context)
        else:
            st.caption("No retrieved contexts recorded.")

        st.markdown("### Scores")
        score_payload = {
            METRIC_LABELS.get(col, col): row[col]
            for col in metric_cols
        }
        score_payload.update(
            {
                "Correctness score": row["correctness_score"],
                "Outcome": row["outcome"],
                "Response model": row.get("response_model"),
                "Judge model": row.get("judge_model"),
                "Embedding model": row.get("embedding_model"),
            }
        )
        st.json(score_payload)


st.set_page_config(page_title="RAG Evaluation Explorer", layout="wide")
st.title("RAG Evaluation Explorer")

all_results_files = sorted(
    f for f in RESULTS_DIR.iterdir() if f.is_file() and f.suffix == ".csv"
)

if not all_results_files:
    st.error(f"No CSV result files found in {RESULTS_DIR}")
    st.stop()

st.sidebar.header("Results")
selected_file = st.sidebar.selectbox(
    "Results file",
    options=[f.name for f in all_results_files],
    index=0,
)

df = load_results(RESULTS_DIR / selected_file)
available_metric_cols = [col for col in METRIC_COLS if col in df.columns]

st.sidebar.header("Correctness")
correctness_metrics = st.sidebar.multiselect(
    "A query is correct when all selected metrics are at least the threshold",
    options=available_metric_cols,
    default=[
        col for col in DEFAULT_CORRECTNESS_METRICS if col in available_metric_cols
    ],
    format_func=lambda col: METRIC_LABELS.get(col, col),
)

if not correctness_metrics:
    st.sidebar.warning("Select at least one metric to label query correctness.")
    st.stop()

correctness_threshold = st.sidebar.slider(
    "Correctness threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.05,
)

df = add_correctness_labels(df, correctness_metrics, correctness_threshold)

st.sidebar.header("Filters")
outcome_filter = st.sidebar.multiselect(
    "Outcome",
    options=["Correct", "Needs review"],
    default=["Correct", "Needs review"],
)
selected_metric = st.sidebar.selectbox(
    "Sort by metric",
    available_metric_cols,
    format_func=lambda col: METRIC_LABELS.get(col, col),
)
sort_order = st.sidebar.radio("Order", ["Lowest first", "Highest first"])

score_range = st.sidebar.slider(
    "Correctness score range",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.05,
)

query_search = st.sidebar.text_input("Search questions")

filtered = df[df["outcome"].isin(outcome_filter)]
filtered = filtered[
    filtered["correctness_score"].between(score_range[0], score_range[1])
]

if query_search:
    filtered = filtered[
        filtered["user_input"].str.contains(query_search, case=False, na=False)
    ]

filtered = filtered.sort_values(
    selected_metric,
    ascending=(sort_order == "Lowest first"),
)

show_score_pills(df)

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Query-level correctness")
    show_query_outcome_chart(df)

with right:
    st.subheader("Metric averages")
    st.dataframe(
        metric_summary(df, available_metric_cols).style.format("{:.2f}"),
        width="stretch",
    )

st.subheader("All queries")
show_metric_heatmap(df, available_metric_cols)

st.subheader("Evaluation examples")
st.caption(f"Showing {len(filtered)} of {len(df)} queries after filters.")

for _, row in filtered.iterrows():
    show_example(row, available_metric_cols)
