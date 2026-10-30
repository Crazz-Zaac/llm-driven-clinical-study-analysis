### What the manual diagnosis shows

- **gpt-5.6-luna / Llama 3.2 (3B):** 142/383 correct (37.1%: 31 confirmed_correct, 111 judge_false_negative); 121 retrieval and 120 generation failures. Judge accuracy 69.2% over 383 comparable rows; it flagged 78.2% of correct answers and passed 2.9% of failures.
- **gpt-5.6-luna / MedLlama 2 (7B):** 115/383 correct (30.0%: 26 confirmed_correct, 89 judge_false_negative); 117 retrieval and 151 generation failures. Judge accuracy 76.2% over 383 comparable rows; it flagged 77.4% of correct answers and passed 0.7% of failures.
- **ministral-8b-2512 / Llama 3.2 (3B):** 149/383 correct (38.9%: 13 confirmed_correct, 136 judge_false_negative); 122 retrieval and 112 generation failures. Judge accuracy 62.7% over 383 comparable rows; it flagged 91.3% of correct answers and passed 3.0% of failures.
- **ministral-8b-2512 / MedLlama 2 (7B):** 118/383 correct (30.8%: 14 confirmed_correct, 104 judge_false_negative); 119 retrieval and 146 generation failures. Judge accuracy 71.5% over 383 comparable rows; it flagged 88.1% of correct answers and passed 1.9% of failures.

**Denominators and interpretation:** Correct answers combine confirmed_correct (blank label + judge ok) and judge_false_negative (model_ok on a flagged row). Manual failure labels always override ok. These are descriptive proportions for the reviewed reported-reference cohort, not a causal comparison of judges or a measurement of full-context retrieval recall.
