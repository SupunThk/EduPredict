# Zero grade scope decision

On 11 October 2026, the main analysis changed to retain all 395 Mathematics students, including 38 recorded zero final grades. The zeros are not confirmed missing values or errors, and their causes remain unknown.

This replaces the earlier 357-student positive-grade-only scope. Scores from the two populations are not directly comparable. Exported historical result files were removed at the user's request; current scores and explanations remain inside the model notebooks.

Every model uses the same 316/79 split with random_state=42 and five training folds. The main scenario excludes G1/G2; the separate later-stage comparison includes both. There are 130 recorded grades below 10, including all 38 zeros.

Final evaluation stays in `notebooks/04_evaluation_recommendations.ipynb`. The test set has already been inspected and must not guide new tuning. Notebook 01 explored the full dataset before splitting. Feature availability, especially absences, remains unresolved. Do not interpret zeros as confirmed withdrawals or dropout.
