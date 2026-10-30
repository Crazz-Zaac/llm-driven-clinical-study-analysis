### What the comparison shows

- **gpt-5.6-luna / Llama 3.2 (3B):** judge accuracy 66.3%; it flagged 77.9% of correct answers and passed 0.9% of failures. Answer correctness AUC 0.88; only 22.1% of correct answers reach 0.75. Context recall AUC 0.83.
- **gpt-5.6-luna / MedLlama 2 (7B):** judge accuracy 72.3%; it flagged 79.5% of correct answers and passed 0.4% of failures. Answer correctness AUC 0.84; only 20.5% of correct answers reach 0.75. Context recall AUC 0.86.
- **ministral-8b-2512 / Llama 3.2 (3B):** judge accuracy 60.3%; it flagged 89.8% of correct answers and passed 1.4% of failures. Answer correctness AUC 0.76; only 10.2% of correct answers reach 0.75. Context recall AUC 0.79.
- **ministral-8b-2512 / MedLlama 2 (7B):** judge accuracy 70.0%; it flagged 87.5% of correct answers and passed 1.2% of failures. Answer correctness AUC 0.75; only 12.5% of correct answers reach 0.75. Context recall AUC 0.80.

**Interpretation:** AUC measures ranking agreement with the manual diagnosis, independent of any threshold. The share of correct answers reaching the threshold shows how much of the judge's flagging comes from the threshold rather than from the ranking. Context results cover human-diagnosed failures only.
