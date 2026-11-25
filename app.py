"""
Amazon review sentiment dashboard (Streamlit).

Run from the project folder:
    streamlit run app.py
"""

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

BLUE, ORANGE = "#3b82f6", "#f97316"

st.set_page_config(page_title="Review sentiment", page_icon="💬", layout="wide")

EXAMPLES = {
    "Positive": "Great taste and arrived fast. My kids love these, will buy again!",
    "Negative": "Terrible. Stale and bland, a total waste of money. Very disappointed.",
    "Mixed": "The flavor is delicious, but the package arrived damaged and half empty.",
    "Negation": "Not good at all. I would not recommend this to anyone.",
}


@st.cache_resource
def load_model():
    """Load the vectorizer, classifier and threshold once."""
    return joblib.load("models/sentiment.joblib")


def explain(pack, text, top=8):
    """
    Contribution of each word = TF-IDF value * model weight.
    Positive contribution pushes toward NEGATIVE, negative toward POSITIVE.
    """
    X = pack["vectorizer"].transform([text])
    names = pack["vectorizer"].get_feature_names_out()
    weights = pack["model"].coef_[0]
    contrib = X.data * weights[X.indices]
    order = np.argsort(contrib)
    words = [(names[X.indices[i]], contrib[i]) for i in order]
    return words[::-1][:top], words[:top]  # (toward negative, toward positive)


def set_example(label):
    st.session_state["review"] = EXAMPLES[label]


pack = load_model()

st.title("Amazon Review Sentiment")
st.caption("TF-IDF + logistic regression trained on Amazon Fine Food Reviews. "
           "Label = stars (4-5 positive, 1-2 negative); 3-star reviews were dropped.")

st.write("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, label in zip(cols, EXAMPLES):
    col.button(label, on_click=set_example, args=(label,), use_container_width=True)

text = st.text_area("Your review (English)", key="review", height=140,
                    placeholder="Type or paste a product review...")

if text.strip():
    p_neg = float(pack["model"].predict_proba(pack["vectorizer"].transform([text]))[0, 1])
    negative = p_neg >= pack["threshold"]

    verdict = "NEGATIVE" if negative else "POSITIVE"
    color = ORANGE if negative else BLUE
    st.markdown(
        f"<div style='padding:18px 22px;border-radius:16px;border:1px solid #22304a;"
        f"background:linear-gradient(145deg,#131c2e,#0f1829);'>"
        f"<span style='color:#8fa3c4'>Verdict</span><br>"
        f"<span style='font-size:2.2rem;font-weight:700;color:{color}'>{verdict}</span>"
        f"<span style='color:#8fa3c4'> &nbsp; probability of negative: {p_neg:.0%}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    toward_neg, toward_pos = explain(pack, text)
    items = [(w, -c, BLUE) for w, c in toward_pos if c < 0] + \
            [(w, c, ORANGE) for w, c in toward_neg if c > 0]
    st.subheader("Which words drove the verdict")
    if items:
        items.sort(key=lambda x: x[1])
        fig = go.Figure(go.Bar(
            x=[i[1] * (1 if i[2] == ORANGE else -1) for i in items],
            y=[i[0] for i in items], orientation="h",
            marker_color=[i[2] for i in items]))
        fig.update_layout(height=360, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6edf7"),
                          xaxis_title="← pushes to positive | pushes to negative →",
                          margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No known words found in this text.")

st.info("The model reads words, not meaning: sarcasm and mixed reviews can fool it. "
        "Labels come from star ratings, which are sometimes inconsistent with the text.")