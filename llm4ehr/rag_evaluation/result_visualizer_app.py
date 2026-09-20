import ast
from pathlib import Path
from textwrap import wrap

import pandas as pd
import streamlit as st


RESULTS_ROOT = Path(__file__).parent
DEFAULT_RESULTS_DIR = RESULTS_ROOT / "results_v1"
QUESTIONNAIRE_PATH = RESULTS_ROOT / "datasets" / "ML4EHR.csv"

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

JUDGE_COMPARISION_FILES = {
    "results_v3": {"retrieval_scope": "all_papers"},
    "results_v4": {"retrieval_scope": "all_papers"},
    "results_v5": {"retrieval_scope": "specific_paper"},
    "results_v6": {"retrieval_scope": "specific_paper"},
}

RETRIEVAL_SCOPE_LABELS = {
    "all_papers": "All papers",
    "specific_paper": "Paper specific",
}


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


def format_retrieval_scope(scope: str | None) -> str:
    if scope is None or pd.isna(scope):
        return "Unknown"
    return RETRIEVAL_SCOPE_LABELS.get(str(scope), str(scope).replace("_", " ").title())


def retrieval_scope_for_run(result_dir: str, df: pd.DataFrame) -> str:
    configured_scope = JUDGE_COMPARISION_FILES.get(result_dir, {}).get(
        "retrieval_scope"
    )
    if configured_scope:
        return format_retrieval_scope(configured_scope)

    if "retrieval_scope" not in df.columns:
        return "Unknown"

    scopes = df["retrieval_scope"].dropna().astype(str).unique()
    if len(scopes) == 1:
        return format_retrieval_scope(scopes[0])
    if len(scopes) > 1:
        return "Mixed"
    return "Unknown"


def load_questionnaire(path: Path = QUESTIONNAIRE_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.loc[:, [col for col in df.columns if not str(col).startswith("Unnamed")]]
    df = df.dropna(how="all")

    for col in ["article_id", "article_title", "field_id", "category"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    required_cols = ["field_id", "category", "question"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        missing = ", ".join(missing_cols)
        raise ValueError(f"Questionnaire is missing required columns: {missing}")

    questionnaire = df[required_cols].dropna(subset=required_cols).drop_duplicates()
    questionnaire = questionnaire.reset_index(drop=True)
    questionnaire["question_id"] = questionnaire.index + 1
    return questionnaire


def dot_quote(value: object) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{text}"'


def wrap_graph_label(text: object, width: int = 46) -> str:
    normalized = " ".join(str(text).split())
    return "\n".join(wrap(normalized, width=width, break_long_words=False))


def questionnaire_to_dot(questionnaire: pd.DataFrame) -> str:
    lines = [
        "digraph questionnaire {",
        "  graph [rankdir=LR, bgcolor=\"transparent\", pad=\"0.25\", nodesep=\"0.45\", ranksep=\"1.1\"];",
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=11, margin=\"0.10,0.07\", color=\"#d7dee8\"];",
        "  edge [color=\"#a9b4c2\", arrowsize=0.7];",
    ]

    for field_id in sorted(questionnaire["field_id"].unique()):
        node_id = f"field_{field_id}"
        lines.append(
            f"  {dot_quote(node_id)} [label={dot_quote(field_id)}, "
            'fillcolor="#dbeafe", color="#93c5fd", fontsize=13];'
        )

    categories = questionnaire[["field_id", "category"]].drop_duplicates()
    for _, row in categories.iterrows():
        field_node = f"field_{row['field_id']}"
        category_node = f"category_{row['category']}"
        category_label = str(row["category"]).replace(f"{row['field_id']}.", "")
        lines.append(
            f"  {dot_quote(category_node)} [label={dot_quote(category_label)}, "
            'fillcolor="#ecfdf5", color="#86efac"];'
        )
        lines.append(f"  {dot_quote(field_node)} -> {dot_quote(category_node)};")

    for _, row in questionnaire.iterrows():
        category_node = f"category_{row['category']}"
        question_node = f"question_{row['question_id']}"
        question_label = f"Q{row['question_id']}\n{wrap_graph_label(row['question'])}"
        lines.append(
            f"  {dot_quote(question_node)} [label={dot_quote(question_label)}, "
            'fillcolor="#fff7ed", color="#fdba74", fontsize=9];'
        )
        lines.append(f"  {dot_quote(category_node)} -> {dot_quote(question_node)};")

    lines.append("}")
    return "\n".join(lines)


def show_question_visualizer(display_mode: str = "Graph") -> None:
    st.caption(f"Questionnaire source: `{QUESTIONNAIRE_PATH.relative_to(RESULTS_ROOT)}`")

    try:
        questionnaire = load_questionnaire()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    field_options = sorted(questionnaire["field_id"].unique())
    selected_fields = st.sidebar.multiselect(
        "Question fields",
        options=field_options,
        default=field_options,
    )
    search = st.sidebar.text_input("Search questionnaire")

    filtered = questionnaire[questionnaire["field_id"].isin(selected_fields)]
    if search:
        search_mask = (
            filtered["field_id"].str.contains(search, case=False, na=False)
            | filtered["category"].str.contains(search, case=False, na=False)
            | filtered["question"].str.contains(search, case=False, na=False)
        )
        filtered = filtered[search_mask]

    if filtered.empty:
        st.warning("No questionnaire rows match the current filters.")
        return

    field_count = filtered["field_id"].nunique()
    category_count = filtered["category"].nunique()
    question_count = len(filtered)
    col1, col2, col3 = st.columns(3)
    col1.metric("Fields", field_count)
    col2.metric("Categories", category_count)
    col3.metric("Questions", question_count)

    if display_mode == "Graph":
        st.subheader("Question graph")
        st.graphviz_chart(questionnaire_to_dot(filtered), width="stretch")

    st.subheader("Grouped questionnaire")
    grouped = filtered.sort_values(["field_id", "category", "question_id"]).rename(
        columns={
            "question_id": "Question ID",
            "field_id": "Field",
            "category": "Category",
            "question": "Question",
        }
    )
    for field, field_df in grouped.groupby("Field", sort=True):
        with st.expander(f"{field} ({len(field_df)} questions)", expanded=True):
            st.dataframe(
                field_df[["Question ID", "Category", "Question"]],
                column_config={
                    "Question ID": st.column_config.NumberColumn(
                        "Question ID",
                        width=100,
                    ),
                    "Category": st.column_config.TextColumn(
                        "Category",
                        width=260,
                    ),
                    "Question": st.column_config.TextColumn(
                        "Question",
                        width=820,
                    ),
                },
                hide_index=True,
                width="stretch",
                height=min(360, 38 + (len(field_df) + 1) * 35),
            )


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
                "Retrieval scope": format_retrieval_scope(row.get("retrieval_scope")),
            }
        )
        st.json(score_payload)


def summarize_run(
    df: pd.DataFrame,
    run_label: str,
    metric_cols: list[str],
    retrieval_scope: str = "Unknown",
) -> dict:
    correct_count = int(df["is_correct"].sum())
    total_count = len(df)
    accuracy = correct_count / total_count if total_count else 0
    summary = {
        "Run": run_label,
        "Retrieval scope": retrieval_scope,
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
        "retrieval_scope",
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
    if "retrieval_scope_baseline" in comparison.columns:
        comparison["Baseline retrieval scope"] = comparison[
            "retrieval_scope_baseline"
        ].map(format_retrieval_scope)
    if "retrieval_scope_candidate" in comparison.columns:
        comparison["Candidate retrieval scope"] = comparison[
            "retrieval_scope_candidate"
        ].map(format_retrieval_scope)

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
    baseline_retrieval_scope: str,
    candidate_retrieval_scope: str,
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
            summarize_run(
                baseline_df,
                baseline_label,
                available_metric_cols,
                baseline_retrieval_scope,
            ),
            summarize_run(
                candidate_df,
                candidate_label,
                available_metric_cols,
                candidate_retrieval_scope,
            ),
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
    retrieval_scope_columns = [
        col
        for col in ["Baseline retrieval scope", "Candidate retrieval scope"]
        if col in comparison.columns
    ]
    delta_columns = [
        "query_id",
        "Outcome change",
        *retrieval_scope_columns,
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
        st.caption(f"Retrieval scope: {baseline_retrieval_scope}")
        st.metric("Correctness", f"{row['Baseline correctness']:.2f}")
        st.write(row["response_baseline"])

    with right:
        st.markdown(f"### {candidate_label}")
        st.caption(f"Retrieval scope: {candidate_retrieval_scope}")
        st.metric(
            "Correctness",
            f"{row['Candidate correctness']:.2f}",
            f"{row['Correctness delta']:+.2f}",
        )
        st.write(row["response_candidate"])


st.set_page_config(page_title="RAG Evaluation Explorer", layout="wide")
st.title("RAG Evaluation Explorer")

st.sidebar.header("Question visualizer")
question_view = st.sidebar.radio(
    "Question visualizer",
    options=["Graph", "Tabular"],
    label_visibility="collapsed",
)

st.sidebar.header("Results Visualizer")
results_view = st.sidebar.radio(
    "Results Visualizer",
    options=["Single run", "Compare runs"],
    index=None,
    label_visibility="collapsed",
)

view_mode = results_view or question_view

if view_mode in {"Graph", "Tabular"}:
    show_question_visualizer(view_mode)
    st.stop()

result_dirs = discover_result_dirs(RESULTS_ROOT)

if not result_dirs:
    st.error(f"No CSV result directories found in {RESULTS_ROOT}")
    st.stop()

results_by_dir = csv_files_by_dir(result_dirs)

st.sidebar.header("Results")
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
elif view_mode == "Compare runs":
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
else:
    st.error(f"Unknown view mode: {view_mode}")
    st.stop()

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
    selected_retrieval_scope = retrieval_scope_for_run(selected_dir, df)
    st.caption(
        f"Viewing `{selected_dir}/{selected_file}` | "
        f"Retrieval scope: {selected_retrieval_scope}"
    )
    show_single_run(
        df,
        available_metric_cols,
        correctness_metrics,
        correctness_threshold,
    )
else:
    baseline_label = f"{baseline_dir}/{baseline_file}"
    candidate_label = f"{candidate_dir}/{candidate_file}"
    baseline_retrieval_scope = retrieval_scope_for_run(baseline_dir, baseline_df)
    candidate_retrieval_scope = retrieval_scope_for_run(candidate_dir, candidate_df)
    st.caption(f"Comparing `{baseline_label}` against `{candidate_label}`")
    show_comparison(
        baseline_df,
        candidate_df,
        baseline_label,
        candidate_label,
        baseline_retrieval_scope,
        candidate_retrieval_scope,
        available_metric_cols,
        correctness_metrics,
        correctness_threshold,
    )
