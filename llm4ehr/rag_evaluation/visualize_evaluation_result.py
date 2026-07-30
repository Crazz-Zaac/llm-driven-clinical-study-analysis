import ast
from pathlib import Path

import pandas as pd
import streamlit as st


RESULTS_ROOT = Path(__file__).parent
DEFAULT_RESULTS_DIR = RESULTS_ROOT / "results_v1"

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


def discover_result_dirs(root: Path) -> list[Path]:
    result_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith("results")
        and any(file.suffix == ".csv" for file in path.iterdir() if file.is_file())
    )
    if DEFAULT_RESULTS_DIR in result_dirs:
        result_dirs.remove(DEFAULT_RESULTS_DIR)
        result_dirs.insert(0, DEFAULT_RESULTS_DIR)
    return result_dirs


def csv_files_by_dir(result_dirs: list[Path]) -> dict[str, list[Path]]:
    return {
        result_dir.name: sorted(
            file
            for file in result_dir.iterdir()
            if file.is_file() and file.suffix == ".csv"
        )
        for result_dir in result_dirs
    }


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


def summarize_run(df: pd.DataFrame, run_label: str, metric_cols: list[str]) -> dict:
    correct_count = int(df["is_correct"].sum())
    total_count = len(df)
    accuracy = correct_count / total_count if total_count else 0
    summary = {
        "Run": run_label,
        "Queries": total_count,
        "Correct": correct_count,
        "Needs review": total_count - correct_count,
        "Derived accuracy": accuracy,
        "Mean correctness": df["correctness_score"].mean(),
    }
    summary.update(
        {
            METRIC_LABELS.get(col, col): df[col].mean()
            for col in metric_cols
            if col in df.columns
        }
    )
    return summary


def prepare_comparison(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    compare_cols = [
        "query_id",
        "user_input",
        "outcome",
        "correctness_score",
        *metric_cols,
        "response",
        "reference",
        "retrieved_contexts",
    ]
    baseline_cols = [col for col in compare_cols if col in baseline_df.columns]
    candidate_cols = [col for col in compare_cols if col in candidate_df.columns]
    comparison = baseline_df[baseline_cols].merge(
        candidate_df[candidate_cols],
        on="query_id",
        how="inner",
        suffixes=("_baseline", "_candidate"),
    )

    comparison["Question"] = comparison["user_input_baseline"].fillna(
        comparison.get("user_input_candidate")
    )
    comparison["Baseline correctness"] = comparison["correctness_score_baseline"]
    comparison["Candidate correctness"] = comparison["correctness_score_candidate"]
    comparison["Correctness delta"] = (
        comparison["Candidate correctness"] - comparison["Baseline correctness"]
    )

    for col in metric_cols:
        comparison[f"{METRIC_LABELS.get(col, col)} delta"] = (
            comparison[f"{col}_candidate"] - comparison[f"{col}_baseline"]
        )

    comparison["Outcome change"] = (
        comparison["outcome_baseline"] + " -> " + comparison["outcome_candidate"]
    )
    return comparison


def show_single_run(
    df: pd.DataFrame,
    available_metric_cols: list[str],
    correctness_metrics: list[str],
    correctness_threshold: float,
) -> None:
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


def show_comparison(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    baseline_label: str,
    candidate_label: str,
    available_metric_cols: list[str],
    correctness_metrics: list[str],
    correctness_threshold: float,
) -> None:
    baseline_df = add_correctness_labels(
        baseline_df,
        correctness_metrics,
        correctness_threshold,
    )
    candidate_df = add_correctness_labels(
        candidate_df,
        correctness_metrics,
        correctness_threshold,
    )
    comparison = prepare_comparison(
        baseline_df,
        candidate_df,
        available_metric_cols,
    )

    if comparison.empty:
        st.error("No matching query_id values found between the selected runs.")
        return

    summary = pd.DataFrame(
        [
            summarize_run(baseline_df, baseline_label, available_metric_cols),
            summarize_run(candidate_df, candidate_label, available_metric_cols),
        ]
    )
    st.subheader("Run summary")
    st.dataframe(
        summary.style.format(
            {
                "Derived accuracy": "{:.0%}",
                "Mean correctness": "{:.2f}",
                **{
                    METRIC_LABELS.get(col, col): "{:.2f}"
                    for col in available_metric_cols
                },
            }
        ),
        width="stretch",
        hide_index=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Matched queries", len(comparison))
    col2.metric(
        "Accuracy delta",
        f"{summary.loc[1, 'Derived accuracy'] - summary.loc[0, 'Derived accuracy']:+.0%}",
    )
    col3.metric(
        "Mean correctness delta",
        f"{comparison['Correctness delta'].mean():+.2f}",
    )
    col4.metric(
        "Improved queries",
        int(comparison["Correctness delta"].gt(0).sum()),
    )

    st.subheader("Correctness by query")
    chart_df = comparison[
        [
            "query_id",
            "Baseline correctness",
            "Candidate correctness",
            "Correctness delta",
        ]
    ].set_index("query_id")
    st.line_chart(
        chart_df[["Baseline correctness", "Candidate correctness"]],
        height=300,
    )

    st.subheader("Query deltas")
    delta_columns = [
        "query_id",
        "Outcome change",
        "Baseline correctness",
        "Candidate correctness",
        "Correctness delta",
        *[f"{METRIC_LABELS.get(col, col)} delta" for col in available_metric_cols],
        "Question",
    ]
    display_df = comparison[delta_columns].sort_values(
        "Correctness delta",
        ascending=True,
    )
    numeric_columns = [
        "Baseline correctness",
        "Candidate correctness",
        "Correctness delta",
        *[f"{METRIC_LABELS.get(col, col)} delta" for col in available_metric_cols],
    ]
    delta_numeric_columns = [
        "Correctness delta",
        *[f"{METRIC_LABELS.get(col, col)} delta" for col in available_metric_cols],
    ]
    st.dataframe(
        display_df.style.format(
            {
                **{
                    col: "{:.2f}"
                    for col in numeric_columns
                    if col not in delta_numeric_columns
                },
                **{col: "{:+.2f}" for col in delta_numeric_columns},
            },
            na_rep="n/a",
        ).background_gradient(
            subset=["Correctness delta"],
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
        ),
        column_config={
            "query_id": st.column_config.NumberColumn("Query", width=80),
            "Outcome change": st.column_config.TextColumn(
                "Outcome change",
                width=190,
            ),
            "Question": st.column_config.TextColumn("Question", width=700),
        },
        width="stretch",
        height=min(520, 38 + (len(display_df) + 1) * 35),
        hide_index=True,
    )

    st.subheader("Side-by-side examples")
    selected_query = st.selectbox(
        "Query",
        options=comparison["query_id"].tolist(),
        format_func=lambda query_id: (
            f"Q{query_id} | "
            f"delta {comparison.loc[comparison['query_id'] == query_id, 'Correctness delta'].iloc[0]:+.2f}"
        ),
    )
    row = comparison[comparison["query_id"] == selected_query].iloc[0]
    st.markdown("### Question")
    st.write(row["Question"])

    left, right = st.columns(2)
    with left:
        st.markdown(f"### {baseline_label}")
        st.metric("Correctness", f"{row['Baseline correctness']:.2f}")
        st.write(row["response_baseline"])

    with right:
        st.markdown(f"### {candidate_label}")
        st.metric(
            "Correctness",
            f"{row['Candidate correctness']:.2f}",
            f"{row['Correctness delta']:+.2f}",
        )
        st.write(row["response_candidate"])


st.set_page_config(page_title="RAG Evaluation Explorer", layout="wide")
st.title("RAG Evaluation Explorer")

result_dirs = discover_result_dirs(RESULTS_ROOT)

if not result_dirs:
    st.error(f"No CSV result directories found in {RESULTS_ROOT}")
    st.stop()

results_by_dir = csv_files_by_dir(result_dirs)

st.sidebar.header("Results")
view_mode = st.sidebar.radio(
    "View mode",
    options=["Single run", "Compare runs"],
)

result_dir_names = list(results_by_dir.keys())

if view_mode == "Single run":
    selected_dir = st.sidebar.selectbox(
        "Results directory",
        options=result_dir_names,
        index=0,
    )
    selected_file = st.sidebar.selectbox(
        "Results file",
        options=[file.name for file in results_by_dir[selected_dir]],
        index=0,
    )
    selected_path = RESULTS_ROOT / selected_dir / selected_file
    df = load_results(selected_path)
    available_metric_cols = [col for col in METRIC_COLS if col in df.columns]
else:
    baseline_dir = st.sidebar.selectbox(
        "Baseline directory",
        options=result_dir_names,
        index=0,
    )
    candidate_default_index = 1 if len(result_dir_names) > 1 else 0
    candidate_dir = st.sidebar.selectbox(
        "Candidate directory",
        options=result_dir_names,
        index=candidate_default_index,
    )

    baseline_file = st.sidebar.selectbox(
        "Baseline file",
        options=[file.name for file in results_by_dir[baseline_dir]],
        index=0,
    )
    candidate_files = [file.name for file in results_by_dir[candidate_dir]]
    candidate_file_default = (
        candidate_files.index(baseline_file)
        if baseline_file in candidate_files
        else 0
    )
    candidate_file = st.sidebar.selectbox(
        "Candidate file",
        options=candidate_files,
        index=candidate_file_default,
    )

    baseline_path = RESULTS_ROOT / baseline_dir / baseline_file
    candidate_path = RESULTS_ROOT / candidate_dir / candidate_file
    baseline_df = load_results(baseline_path)
    candidate_df = load_results(candidate_path)
    available_metric_cols = [
        col
        for col in METRIC_COLS
        if col in baseline_df.columns and col in candidate_df.columns
    ]

if not available_metric_cols:
    st.error("No supported metric columns were found in the selected result file(s).")
    st.stop()

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

if view_mode == "Single run":
    st.caption(f"Viewing `{selected_dir}/{selected_file}`")
    show_single_run(
        df,
        available_metric_cols,
        correctness_metrics,
        correctness_threshold,
    )
else:
    baseline_label = f"{baseline_dir}/{baseline_file}"
    candidate_label = f"{candidate_dir}/{candidate_file}"
    st.caption(f"Comparing `{baseline_label}` against `{candidate_label}`")
    show_comparison(
        baseline_df,
        candidate_df,
        baseline_label,
        candidate_label,
        available_metric_cols,
        correctness_metrics,
        correctness_threshold,
    )
