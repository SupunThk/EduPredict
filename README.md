# EduPredict: Student Performance Prediction and Early Academic Support

IT3091 Machine Learning · Group 2026-AI-33 · Guided Data Track (code 3: Education)

##  Overview

EduPredict estimates a student's final Mathematics grade (G3, on the Portuguese 0–20 scale) from information a school already holds, and marks students whose estimate is below 10 as *potentially at risk* so that staff can review them. The project compares five regression models, selects one for each of two prediction stages, evaluates them on held-out students, and wraps the selected models in a small local app.

The predictions support human judgement. They are not meant to decide grades, discipline or access to help.

##  Problem Statement

Teachers and academic support staff need a way to prioritise students for further review while also considering current classroom evidence and what students ask for. The decision question is what final grade a student may achieve, and whether that estimate warrants a supportive conversation.

| Element | Definition |
|---|---|
| Primary task | Supervised regression predicting G3, the recorded final Mathematics grade |
| Unit of analysis | One student record: 395 rows, 33 raw columns |
| Secondary output | *Potentially at risk* flag when the continuous predicted G3 is below 10 |
| Primary stakeholders | Teachers and academic support staff; students are affected stakeholders |

##  Project Objectives

1. Frame the business problem, decision lens and exact prediction task.
2. Understand the data, its quality issues and what is known at prediction time.
3. Build leakage-safe, reusable preprocessing.
4. Compare a baseline and five regression models with the same cross-validation folds.
5. Evaluate the selected models honestly on held-out students, including missed at-risk students.
6. Recommend staff-reviewed use and document the limits.

##  Dataset

- **Source:** UCI Machine Learning Repository, Student Performance (Cortez, 2008). https://archive.ics.uci.edu/dataset/320/student+performance
- **File used:** `data/student-mat.csv` (Mathematics, semicolon separated). `data/student.txt` holds the source descriptions. The Portuguese-language file is not used.
- **Size:** 395 students from two Portuguese schools, 33 columns, no missing cells and no duplicate rows.
- **Target:** G3, the final grade (0–20). There are 38 recorded zero grades. Their causes are unknown, so they are kept.
- **Prior grades:** G1 and G2 (first and second period grades) are only known later in the year, so they are used only in the later-stage scenario.

The full variable definitions are in the [Data Dictionary](docs/group/IT3091_2026-AI-33_04_Data_Dictionary.pdf).

##  Machine Learning Approach

- **Two scenarios:** the *main* scenario uses 30 inputs without G1/G2. The *later-stage* scenario uses 32 inputs including both.
- **Split:** one shared 80/20 split (316 training, 79 test students, `random_state=42`) and five shuffled training folds.
- **Preprocessing** (`src/preprocessing.py`): 13 binary features mapped to 0/1, four nominal features one-hot encoded, ten ordinal codes kept as numbers, a search over original versus `log1p` absences, and scaling only for the linear models. All learned steps are fitted inside each fold.
- **Models:** training mean and median baselines, Linear Regression, Ridge, Decision Tree, Random Forest and Gradient Boosting, each with its own tuning search.
- **Selection:** lowest mean training-CV MAE, separately for each scenario. RMSE and R² support interpretation. The at-risk flag is evaluated with accuracy, precision, recall, F1, specificity and confusion counts.

##  Project Workflow

```
Problem framing and data understanding (Notebook 01)
        ↓
Common 316/79 split and preprocessing (Notebook 02, src/preprocessing.py)
        ↓
Five model notebooks, both scenarios (03a–03e)
        ↓
Compare on training CV and export the selected pipelines (Notebook 03 → models/)
        ↓
Final evaluation on the 79 held-out students (Notebook 04)
        ↓
Staff-reviewed recommendations, a prediction notebook (05) and the local app (app/)
```

See the [Workflow Diagram](docs/group/IT3091_2026-AI-33_02_Workflow_Diagram.pdf) and the [Decision Log](docs/group/IT3091_2026-AI-33_03_Decision_Log.pdf).

##  Evaluation

Random Forest had the lowest training-CV MAE in both scenarios (2.964 without G1/G2, 0.938 with them). On the 79 held-out test students:

| Predictor | MAE | RMSE | R² | Below-10 students found | Extra flags |
|---|---|---|---|---|---|
| Training median baseline | 3.620 | 4.534 | -0.003 | – | – |
| Main model (without G1/G2) | 3.095 | 3.871 | 0.269 | 10 of 27 | 11 |
| Later-stage model (with G1/G2) | 1.170 | 1.975 | 0.810 | 27 of 27 | 10 |
| G2 unchanged (later stage only) | 1.000 | 2.072 | 0.791 | 26 of 27 | 6 |

The test set has now been inspected, so it must not guide any further tuning or threshold choice.

##  Key Findings

- Without prior grades, the main model beats the baselines but misses 17 of the 27 test students who scored below 10. Its flags cannot be relied on alone.
- G1 and G2 carry most of the signal. Once they are known, errors fall sharply.
- At the later stage, simply reusing G2 has a lower MAE than the forest, and the G2 < 10 rule finds 26 of 27 with fewer extra flags. Model complexity has to be justified against this simple reference.
- The recommended use is a staff-reviewed pilot on a new cohort, where flags are combined with classroom evidence and an unflagged student can still receive support.

The full argument is in the [Business Problem](docs/group/IT3091_2026-AI-33_01_Business_Problem.pdf) document and Notebook 04.

##  Technologies Used

- Python 3.11, Jupyter notebooks
- pandas, NumPy, SciPy, scikit-learn, joblib
- Matplotlib and seaborn for the notebooks' charts
- A plain HTML, CSS and JavaScript app served by Python's built-in web server

##  Project Structure

```
EduPredict/
├── app/                 Local prediction app (server.py, static/ page, tests)
├── data/                student-mat.csv (used), student-por.csv, student.txt
├── docs/
│   ├── assignment/      IT3091 assignment descriptor
│   ├── group/           Business problem, workflow, decision log, data dictionary (PDF)
│   ├── individual/      One-page learning journey per member (PDF)
│   └── notes/           Model workflow and zero-grade scope notes (Markdown + PDF)
├── models/              The two selected pipelines + metadata; experiments/ holds all fitted candidates
├── notebooks/           01 EDA → 02 preprocessing → 03a–03e models → 03 comparison → 04 evaluation → 05 prediction
├── src/                 preprocessing.py, model_utils.py, evaluation_utils.py
└── requirements.txt     Notebook dependencies
```

##  Installation & Setup

The saved models must be loaded with the same library versions they were fitted with, so use Python 3.11 and the pinned versions in `app/requirements.txt`.

```bash
git clone https://github.com/SupunThk/EduPredict.git
cd EduPredict
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r app/requirements.txt
pip install -r requirements.txt    # notebook extras (Jupyter, plotting)
```

##  Usage

**Local app.** Run this from the project root, and a browser tab opens at http://127.0.0.1:8000.

```bash
python app/server.py
```

- *Student:* describe one student. The estimate updates live on a 0–20 grade ruler. Choose "G1 and G2 known" to use the later-stage model.
- *Class list:* upload a CSV of students and download the predictions.
- *Models:* the selection rule, test results and limitations.

To check that the app reproduces the notebook results, run `python -m unittest discover -s app`. More detail is in [app/README.md](app/README.md).

**Notebooks.** Open them in order from `notebooks/`. The run order is in [notebooks/README.md](notebooks/README.md). Notebook 05 predicts a single student inside Jupyter.

Use fictional or de-identified students only.

##  Team Members

| Member | IT Number | Name | Responsibility |
|---|---|---|---|
| 1 | IT24102032 | Thinila N.A.D.I.D. | Problem Framing & Data Understanding |
| 2 | IT24102008 | Withanan W.Y.P. | Preprocessing & Feature Engineering |
| 3 | IT24102088 | Maduwatta U.S.T. | Regression Modelling |
| 4 | IT24102885 | Karunarathna H.R.A. | Evaluation & Recommendation |

Personal learning journeys are in [docs/individual/](docs/individual/).

##  Limitations

- The sample is small and comes from two Portuguese schools, so results may not transfer to other schools.
- Exploratory analysis looked at the full dataset before splitting, and the selection folds were reused, so development was not fully blind.
- The causes of the 38 zero grades are unknown, and the source description of the `failures` codes conflicts with the data.
- When each input becomes available, especially absences, is not confirmed, so excluding G1/G2 alone does not prove early-warning ability.
- Subgroup checks are descriptive, and the predictions show associations rather than the effect of any intervention.
- Known issue: `src/evaluation_utils.py` compares the preprocessing file's fingerprint byte for byte. The fingerprint was recorded on a Windows checkout with CRLF line endings, so Notebooks 04 and 05 can stop with an `AssertionError` on macOS or Linux even though the code is identical. The app's loader ignores line endings and is not affected.

##  References

1. Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
2. Cortez, P. and Silva, A. (2008). Using data mining to predict secondary school student performance. In *Proceedings of the 5th Future Business Technology Conference (FUBUTEC 2008)*, pp. 5–12. Porto, Portugal: EUROSIS.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. IT3091 Machine Learning Group Assignment Descriptor, [docs/assignment/IT3091_Assignment_Descriptor_V1.pdf](docs/assignment/IT3091_Assignment_Descriptor_V1.pdf).
