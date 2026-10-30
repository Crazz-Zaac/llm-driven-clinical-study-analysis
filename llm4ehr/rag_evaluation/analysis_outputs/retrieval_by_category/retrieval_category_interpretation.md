#### Where retrieval coverage is stronger or weaker

- **V1, Llama 3.2 (3B):** highest mean recall in **Cohort** (0.505; precision 0.802; n=106 pairs); lowest in **Features** (0.131; precision 0.548; n=42 pairs).
- **V1, MedLlama 2 (7B):** highest mean recall in **Outcome** (0.506; precision 0.885; n=61 pairs); lowest in **Features** (0.136; precision 0.571; n=42 pairs).
- **V2, Llama 3.2 (3B):** highest mean recall in **Outcome** (0.757; precision 0.918; n=61 pairs); lowest in **Features** (0.523; precision 0.548; n=42 pairs).
- **V2, MedLlama 2 (7B):** highest mean recall in **Outcome** (0.708; precision 0.951; n=61 pairs); lowest in **Features** (0.498; precision 0.619; n=42 pairs).

**Detailed review priorities:** The table below lists the three lowest mean-recall queries per run among categories with at least 5 valid papers. The zero-recall fraction counts scores equal to zero, not human-confirmed failures. All categories, including smaller samples, remain in the heatmap and full rankings.

**Interpretation:** Inspect low-recall categories for missing reference statements in the evaluated evidence. A high precision score does not rule out incomplete coverage. The configured context-character cap can omit evidence present in longer retrieved material, so these scores alone cannot distinguish retrieval omissions from evaluation truncation.

**Scope:** The rankings are descriptive, not significance tests. They concern only reported references and the evaluated context representation, not system-wide retrieval efficiency. Because these metrics do not use generated answers, differing answers do not explain their score differences when the retrieval-evaluation inputs are identical.
