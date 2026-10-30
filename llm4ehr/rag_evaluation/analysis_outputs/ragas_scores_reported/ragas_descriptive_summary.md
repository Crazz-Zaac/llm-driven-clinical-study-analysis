#### What the RAGAS scores show

- **Faithfulness:** V1: Llama higher by 0.107 (0.715 vs 0.607, Llama vs MedLlama); V2: Llama higher by 0.069 (0.746 vs 0.677, Llama vs MedLlama).
- **Answer relevancy:** V1: MedLlama higher by 0.158 (0.456 vs 0.614, Llama vs MedLlama); V2: MedLlama higher by 0.119 (0.408 vs 0.527, Llama vs MedLlama).
- **Context recall:** V1: MedLlama higher by 0.005 (0.396 vs 0.401, Llama vs MedLlama); V2: Llama higher by 0.010 (0.633 vs 0.623, Llama vs MedLlama).
- **Context precision:** V1: MedLlama higher by 0.010 (0.762 vs 0.773, Llama vs MedLlama); V2: MedLlama higher by 0.003 (0.815 vs 0.817, Llama vs MedLlama).
- **Answer correctness:** V1: Llama higher by 0.007 (0.431 vs 0.424, Llama vs MedLlama); V2: MedLlama higher by 0.025 (0.469 vs 0.495, Llama vs MedLlama).

**Reading the pattern:** No single metric establishes an overall winner. Answer correctness is a graded RAGAS score, not the proportion of correct answers. High faithfulness or context precision does not establish high reference coverage or answer correctness; the separate scores and distributions should be read together.

**Limits:** These are descriptive mean differences, not significance tests or human-confirmed failure diagnoses. V1/V2 use different judges and have differing saved responses, so changes cannot be interpreted as a controlled judge-only comparison. The bootstrap intervals are exploratory with only 12 papers and do not account for variability across a new set of questions.
