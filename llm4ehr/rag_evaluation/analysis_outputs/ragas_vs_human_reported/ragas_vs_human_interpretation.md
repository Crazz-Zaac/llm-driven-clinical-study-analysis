### What the RAGAS-human comparison shows

- **gpt-5.6-luna / Llama 3.2 (3B):** answer correctness AUC 0.85 (median 0.61 for human-correct vs 0.31 for failures; 21.8% of human-correct answers reach 0.75). Context recall AUC 0.84 (median 0.50 when the context contains the answer vs 0.00 when it lacks it).
- **gpt-5.6-luna / MedLlama 2 (7B):** answer correctness AUC 0.85 (median 0.59 for human-correct vs 0.33 for failures; 22.6% of human-correct answers reach 0.75). Context recall AUC 0.86 (median 0.50 when the context contains the answer vs 0.00 when it lacks it).
- **ministral-8b-2512 / Llama 3.2 (3B):** answer correctness AUC 0.73 (median 0.57 for human-correct vs 0.43 for failures; 8.7% of human-correct answers reach 0.75). Context recall AUC 0.79 (median 0.75 when the context contains the answer vs 0.33 when it lacks it).
- **ministral-8b-2512 / MedLlama 2 (7B):** answer correctness AUC 0.74 (median 0.61 for human-correct vs 0.48 for failures; 11.9% of human-correct answers reach 0.75). Context recall AUC 0.80 (median 0.73 when the context contains the answer vs 0.33 when it lacks it).

**Interpretation:** AUC measures ranking agreement with the human diagnosis, independent of any threshold. The share of human-correct answers reaching the threshold shows how much of the judge's flagging comes from the threshold rather than the ranking. Context results cover human-diagnosed failures only.
