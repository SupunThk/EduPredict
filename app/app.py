"""EduPredict local prediction app.

Run from the project root:  streamlit run app/app.py
"""

import pandas as pd
import streamlit as st

import model_service as ms

st.set_page_config(page_title="EduPredict – Grade Prediction", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading and verifying the selected models…")
def get_models():
    return ms.load_models()


try:
    models, dataset = get_models()
except ms.ModelLoadError as error:
    st.error("The saved models could not be loaded safely.")
    st.write(str(error))
    st.info("Install the pinned versions with `pip install -r app/requirements.txt` "
            "(Python 3.11), then restart the app.")
    st.stop()


def fmt_pct(x):
    return "–" if pd.isna(x) else f"{x:.0%}"


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Selected models")
    st.caption("Chosen in Notebook 03 by the lowest training-CV MAE in each scenario. "
               "Both scenarios selected a Random Forest.")
    for m in models.values():
        t, cv = m.test_metrics, m.metadata["selection_cv_scores"]["MAE"]
        st.markdown(f"**{m.label}**  \n{m.estimator_name}, {len(m.features)} inputs")
        st.markdown(f"- Training-CV MAE: **{cv:.3f}**\n"
                    f"- Held-out test MAE: **{t['MAE']:.3f}**\n"
                    f"- Found **{t['TP']} of {t['TP'] + t['FN']}** at-risk test students "
                    f"({t['FP']} extra flags)")
    st.success("Model, data and code fingerprints verified.")
    st.caption("Use fictional or de-identified students only. Nothing you enter is saved.")

# ---------------------------------------------------------------- header
st.title("🎓 EduPredict – Student Grade Prediction")
st.write("Estimate a student's final Mathematics grade (G3, 0–20) with the project's selected models. "
         f"A predicted grade below **{ms.THRESHOLD}** is shown as *potentially at risk* so staff can review "
         "the student. This is a support signal, not a decision about grades, discipline or access to help.")

single, batch, about = st.tabs(["Single student", "Batch (CSV)", "About the models"])


def field_input(name, container):
    _, label, _, rule, default = ms.FIELD_BY_NAME[name]
    if isinstance(rule, dict):
        codes = list(rule)
        return container.selectbox(label, codes, index=codes.index(default),
                                   format_func=rule.get, key=name)
    low, high = rule
    return container.number_input(label, min_value=low, max_value=high, value=default, step=1, key=name)


# ---------------------------------------------------------------- single student
with single:
    has_prior = st.toggle("G1 and G2 are already known (later in the school year)", value=False)
    scenario = ms.choose_scenario(has_prior)
    model = models[scenario]
    st.caption(f"Using **{model.label}**. "
               + ("Only use this once both period grades exist." if has_prior
                  else "Switch on the toggle when both period grades are available."))

    with st.form("student"):
        groups = ["Student and family", "Study and support", "Lifestyle and attendance"]
        if has_prior:
            groups.append("Prior grades")
        values = {}
        for group in groups:
            st.subheader(group)
            cols = st.columns(3)
            names = [f[0] for f in ms.FIELDS if f[2] == group]
            for i, name in enumerate(names):
                values[name] = field_input(name, cols[i % 3])
        submitted = st.form_submit_button("Predict final grade", type="primary")

    if submitted:
        clean, errors = ms.validate_student(values, scenario)
        if errors:
            st.error("Please fix: " + "; ".join(errors))
        else:
            result = ms.predict(model, [clean])[0]
            grade, t = result["predicted_G3"], model.test_metrics
            c1, c2 = st.columns([1, 2])
            c1.metric("Predicted final grade (G3)", f"{grade:.2f} / 20")
            if result["flag"]:
                c2.warning(f"**Potentially at risk.** The prediction is below {ms.THRESHOLD}. "
                           "Consider a supportive conversation alongside current classroom evidence.")
            else:
                c2.success(f"**Not flagged.** The prediction is {ms.THRESHOLD} or above. "
                           "This does not mean the student needs no help.")
            st.info(f"On the 79 held-out test students this model was off by **{t['MAE']:.2f} grade points** "
                    f"on average and found **{t['TP']} of {t['TP'] + t['FN']}** students who scored below "
                    f"{ms.THRESHOLD}. Treat the number as an estimate, not a guarantee.")
            if clean["absences"] > ms.OBSERVED_MAX_ABSENCES:
                st.caption("Note: absences is above the maximum seen in the training data (75), "
                           "so this prediction is less reliable.")

# ---------------------------------------------------------------- batch
with batch:
    st.write("Upload a CSV with one student per row using the raw dataset codes (comma or semicolon separated). "
             "Rows that include both **G1** and **G2** use the later-stage model; other rows use the main model.")
    template = pd.DataFrame([{**ms.DEFAULT_STUDENT, "G1": "", "G2": ""}])
    st.download_button("Download CSV template", template.to_csv(index=False).encode(),
                       "edupredict_template.csv", "text/csv")
    upload = st.file_uploader("Student CSV", type="csv")
    if upload is not None:
        try:
            table, had_target = ms.read_student_csv(upload.getvalue())
        except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
            st.error("Could not read this file. Upload a UTF-8 CSV like the template.")
            st.stop()
        if had_target:
            st.caption("The G3 column was ignored: it is the value being predicted.")
        results = ms.predict_batch(models, table)
        valid = results["predicted_G3"].notna()
        k1, k2, k3 = st.columns(3)
        k1.metric("Students", len(results))
        k2.metric("Potentially at risk", int((results["status"] == "Potentially at risk").sum()))
        k3.metric("Invalid rows", int((~valid).sum()))
        st.dataframe(results, hide_index=True, width="stretch")
        st.download_button("Download predictions", results.to_csv(index=False).encode(),
                           "edupredict_predictions.csv", "text/csv")

# ---------------------------------------------------------------- about
with about:
    st.subheader("How the model is chosen")
    st.write("Notebook 03 compared Linear Regression, Ridge, Decision Tree, Random Forest and Gradient Boosting "
             "on the same 316 training students with five-fold cross-validation, and selected the lowest mean MAE "
             "separately for each scenario. The app loads those two saved pipelines and picks between them only "
             "by what you know: with both G1 and G2 it uses the later-stage model, otherwise the main model.")
    rows = []
    for m in models.values():
        t = m.test_metrics
        rows.append({"Model": m.label, "Estimator": m.estimator_name, "Inputs": len(m.features),
                     "Absences": "log1p" if m.metadata["log_absences"] else "original",
                     "Training-CV MAE": round(m.metadata["selection_cv_scores"]["MAE"], 3),
                     "Test MAE": round(t["MAE"], 3), "Test RMSE": round(t["RMSE"], 3), "Test R²": round(t["R2"], 3),
                     "Recall (G3<10)": fmt_pct(t["Recall"]), "Precision": fmt_pct(t["Precision"]),
                     "Missed": t["FN"], "Extra flags": t["FP"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Test metrics are recomputed from the saved models on the same 79 held-out students "
               "and match Notebook 04. That test set has already been inspected, so it must not be used to tune further.")
    st.subheader("Limits to keep in mind")
    st.markdown(
        "- The main model misses **17 of 27** at-risk test students. An unflagged student can still need support.\n"
        "- At the later stage, simply reusing G2 has a lower MAE (1.000) than the forest, so complexity must be justified.\n"
        "- Data come from two Portuguese schools; results may not transfer to other schools.\n"
        "- The 38 zero final grades have unknown causes, and the `failures` code meaning is unconfirmed.\n"
        "- Predictions are associations, not causal effects of any intervention.")
