"""Streamlit demo.  Run from the repo root:  python -m streamlit run app.py

Needs only files that are committed to the repo: models/lgbm_{B,C}[_compliant].joblib
and demo_data/ (see src.export_demo). No dataset or training required.
"""
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src import serve
from src.serve import build_row, make_raw, predict, top_contributions

st.set_page_config(page_title="Loan repayment risk", layout="wide")


# ----------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    return serve.load_models()


@st.cache_resource
def load_schema():
    return serve.load_schema()


@st.cache_resource(show_spinner="Loading held-out applicants...")
def load_heldout(sample):
    return serve.load_heldout(sample)


ALL_MODELS = load_models()
if not ALL_MODELS:
    st.error("No complete model pair found in models/. Run the pipeline first "
             "(src.features, src.aggregate, then src.tune --set B and --set C, "
             "optionally with --compliant).")
    st.stop()
try:
    SCHEMA = load_schema()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run python -m src.export_demo (or src.features) first.")
    st.stop()


# ----------------------------------------------------------------- helpers
def explanation_fig(art, row, k=8):
    labels, values = top_contributions(art, row, k)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(labels[::-1], values[::-1],
            color=["#d62728" if x > 0 else "#1f77b4" for x in values[::-1]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("effect on log-odds of default (red = raises risk)")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def show_decision(p, thr):
    c1, c2, c3 = st.columns(3)
    c1.metric("P(default)", f"{p:.1%}")
    c2.metric("Reject if at least", f"{thr:.1%}")
    (c3.error if p >= thr else c3.success)("**REJECT**" if p >= thr else "**APPROVE**")


def options(col, default=None, blank=False):
    opts = list(SCHEMA[col].cat.categories)
    if blank:
        opts = ["(not given)"] + opts
    return opts, (opts.index(default) if default in opts else 0)


# ----------------------------------------------------------------- sidebar
LABELS = {"research": "With gender (statistics)",
          "compliant": "Without gender (compliance)"}  # first = default
st.sidebar.header("Model variant")
VARIANT = st.sidebar.radio("Variant", [k for k in LABELS if k in ALL_MODELS],
                           format_func=LABELS.get, label_visibility="collapsed")
MODELS = ALL_MODELS[VARIANT]
RESEARCH = VARIANT == "research"
st.sidebar.header("Decision rule")
R = st.sidebar.slider("A missed defaulter costs this many times a rejected good borrower",
                      1, 20, 5)
THR = 1 / (1 + R)
st.sidebar.markdown(f"**Reject if P(default) ≥ {THR:.1%}**")
st.sidebar.caption(
    "The model's probabilities are well calibrated (mean predicted 8.0% vs "
    "actual 8.1%), so the cost-minimising threshold is 1/(1+R). On held-out "
    "data the empirically best thresholds matched this closely (e.g. 0.170 vs "
    "0.167 at R=5). The usual 0.5 would reject almost nobody.")

st.title("Loan repayment risk: thin-file vs full-history models")
st.caption(f"Model variant: **{LABELS[VARIANT]}**")
tab1, tab2 = st.tabs(["Assess a new applicant (no credit history)",
                      "Replay a held-out applicant"])

# -------------------------------------------------------- tab 1: the form
with tab1:
    st.write("Fill in what an applicant with **no credit-bureau history** would "
             "supply. Everything not asked here (gender, region, documents, "
             "external scores...) is passed to the model as *missing*.")
    with st.form("applicant"):
        c1, c2, c3 = st.columns(3)
        income = c1.number_input("Annual income", 10_000, 10_000_000, 150_000, 5_000)
        credit = c1.number_input("Loan amount", 10_000, 5_000_000, 500_000, 10_000)
        annuity = c1.number_input("Annual repayment (annuity)", 1_000, 500_000, 25_000, 1_000)
        goods = c1.number_input("Price of goods financed", 10_000, 5_000_000, 450_000, 10_000)
        gender = None
        if RESEARCH:
            go = ["(not given)"] + list(SCHEMA["CODE_GENDER"].cat.categories)
            gender = c1.selectbox("Gender (research variant only)", go)
        age = c2.slider("Age", 20, 69, 35)
        employed = c2.checkbox("Currently employed", True)
        years = c2.slider("Years in current job", 0.0, 40.0, 5.0, 0.5)
        children = c2.number_input("Children", 0, 10, 0)
        family = c2.number_input("Family members", 1, 12, 2)
        o, i = options("NAME_EDUCATION_TYPE", "Secondary / secondary special")
        education = c3.selectbox("Education", o, i)
        o, i = options("NAME_INCOME_TYPE", "Working")
        income_type = c3.selectbox("Income type", o, i)
        o, i = options("NAME_FAMILY_STATUS", "Married")
        family_status = c3.selectbox("Family status", o, i)
        o, i = options("NAME_HOUSING_TYPE", "House / apartment")
        housing = c3.selectbox("Housing", o, i)
        o, i = options("OCCUPATION_TYPE", blank=True)
        occupation = c3.selectbox("Occupation", o, i)
        o, i = options("NAME_CONTRACT_TYPE", "Cash loans")
        contract = c3.selectbox("Loan type", o, i)
        o, i = options("FLAG_OWN_CAR", "N")
        own_car = c3.selectbox("Owns a car", o, i)
        car_age = c3.number_input("Car age (years, if owned)", 0, 60, 5)
        o, i = options("FLAG_OWN_REALTY", "Y")
        own_realty = c3.selectbox("Owns property", o, i)
        st.form_submit_button("Assess applicant")

    raw = make_raw(income, credit, annuity, goods, age, employed, years,
                   children, family, education, income_type, family_status,
                   housing, occupation, contract, own_car, car_age, own_realty,
                   gender=gender)
    row = build_row(SCHEMA, raw)
    p = predict(MODELS["C"], row)
    st.subheader("Thin-file model (no credit-history features)")
    show_decision(p, THR)
    st.pyplot(explanation_fig(MODELS["C"], row))
    st.caption("Bars show how each input moved this applicant's risk relative "
               "to the average applicant (TreeSHAP).")

# --------------------------------------------- tab 2: held-out applicants
with tab2:
    st.write("Pick a real applicant the models never saw in training and "
             "compare the thin-file model with the full-history model.")
    if st.toggle("Load held-out applicants"):
        X_te, y_te = load_heldout(MODELS["B"]["sample"])
        kind = st.radio("Draw", ["any applicant", "a defaulter", "a repaid loan"],
                        horizontal=True)
        if "seed" not in st.session_state:
            st.session_state.seed = 0
        if st.button("Draw another"):
            st.session_state.seed += 1
        mask = {"any applicant": np.ones(len(y_te), bool),
                "a defaulter": (y_te.to_numpy() == 1),
                "a repaid loan": (y_te.to_numpy() == 0)}[kind]
        pos = np.random.default_rng(st.session_state.seed).choice(np.flatnonzero(mask))
        xrow = X_te.iloc[[pos]]
        outcome = "DEFAULTED" if y_te.iloc[pos] == 1 else "repaid"
        st.markdown(f"**Actual outcome: {outcome}**")
        left, right = st.columns(2)
        for col, key, title in [(left, "C", "Thin-file model"),
                                (right, "B", "Full model (with credit history)")]:
            with col:
                st.subheader(title)
                show_decision(predict(MODELS[key], xrow), THR)
                st.pyplot(explanation_fig(MODELS[key], xrow))
