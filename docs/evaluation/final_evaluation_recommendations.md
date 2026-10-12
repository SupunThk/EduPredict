# Final evaluation and recommendations

## Evidence from the held-out set

Both frozen Random Forest models were evaluated on the same 79 students; 27 had G3 < 10. Model choice and tuning used only the 316 training students. All 395 students, including 38 recorded zero grades, remain in scope.

- **Main model, without G1/G2:** MAE **3.095**, RMSE **3.871**, R² **0.269**. Relative MAE reduction versus the training-median baseline is **14.5%** (negative means worse). An MAE is an average, not a guarantee for an individual student.
- **Later-stage model, with G1/G2:** MAE **1.170**, RMSE **1.975**, R² **0.810**. Its MAE is higher than the simple G2-unchanged benchmark (**1.000**). These inputs require a later prediction time; this is a separate scenario, not an early-warning accuracy claim.
- **Main risk rule (predicted G3 < 10):** flags **21** students, correctly flags **10**, misses **17** at-risk students, and produces **11** false positives. Precision is **47.6%** and recall is **37.0%**.
- **Later-stage risk rule:** flags **37**, correctly flags **27**, misses **0**, and produces **10** false positives. Precision is **73.0%** and recall is **100.0%**.
- **Recorded zero-grade errors:** Main: without G1/G2: MAE 8.071 across 5 students; Later stage: with G1/G2: MAE 4.606 across 5 students. These observations are retained, and their causes are not established by this dataset.

### Error patterns and simple alternatives

The main-model grade-group table shows overprediction at the low end (negative mean residuals for G3=0 and G3=1–9) and underprediction at the high end (positive mean residual for G3=15–20). Predictions tend toward the middle of the observed range. This helps explain why many students below 10 are missed despite improvement over a constant baseline. The later-stage model still has large errors among the five zero-grade test students.

Simply flagging G2 < 10 misses **1** at-risk student(s) and produces **6** false positives. The later-stage forest catches **1** additional at-risk student(s) with **4** additional false positives. It also has worse MAE than using G2 unchanged, although its RMSE is lower. It is therefore not an unqualified improvement over that simple reference. This comparison does not retrospectively replace the frozen model selection. Perfect observed recall in a small sample does not guarantee perfect recall on future students.

The descriptive audit shows different main-model recall across recorded sex groups and schools. Counts are small, especially the 12-student MS group with four observed at-risk students; investigate these signals using larger representative data before drawing conclusions.

## Practical recommendations

1. Use the main model only as an exploratory support signal. Its 17 missed at-risk students mean an unflagged prediction must not exclude a student from assistance. Combine teacher observations, student requests and academic evidence with model flags.
2. For flagged students, have a staff member check current performance and data accuracy, speak with the student, and agree on appropriate support such as tutoring, study planning or attendance assistance. A predicted low grade is not a diagnosis or evidence of lack of effort.
3. If G1 and G2 are available, consider the later-stage scenario for follow-up after the second grading period. Compare its value with simply reviewing G2, including support workload and missed students, before adopting a complex model.
4. Keep the grade-10 rule fixed for this report. If a school needs greater recall or a different review workload, choose a new operating rule using training/validation data and a stated cost tradeoff, then evaluate it on new untouched data. Do not tune on these 79 test outcomes.
5. Before practical use, define a prediction date and verify every input is available then, particularly absences. Run a prospective pilot on a new cohort, check subgroup performance with adequate sample sizes, and monitor missed support cases and unnecessary reviews. Keep access limited to authorized support staff and allow students to correct inaccurate data.

## Limitations and responsible use

- The dataset has only 395 Mathematics students from two schools. One 79-student random holdout does not establish performance in other subjects, schools or future cohorts; a new temporal or external evaluation is needed.
- The full dataset was explored during EDA before splitting, so the holdout was not fully blind to exploratory analysis. Training CV scores also informed model selection and are not independent estimates of final performance.
- Input availability remains unresolved. Excluding G1/G2 alone does not prove that the main model can be used early; whole-period absences or other later-collected inputs could be unavailable at the intended date.
- Recorded zero grades are valid observations for this analysis, but their meaning is unknown. The model cannot distinguish their causes or justify relabeling them.
- Subgroup checks are descriptive and small; they do not establish fairness. Personal and family attributes require an appropriateness and privacy review before operational use.
- Regression outputs are point estimates, not probabilities or individual uncertainty intervals. Error averages and a fixed threshold cannot ensure reliable decisions near grade 10.
- Associations and prediction accuracy do not demonstrate that a particular intervention will improve grades. Assess support outcomes separately.
- No automatic disciplinary action, exclusion, grading decision or denial of support should follow a model prediction. Staff should review context, communicate uncertainty and retain responsibility.

## Conclusion

The final test results quantify both grade error and the consequences of the risk flag. Treat this as an educational modelling study and a candidate for a supervised support pilot, not a validated automatic decision system. Preserve these results as the final evaluation of the frozen models; further development needs new evaluation data.
