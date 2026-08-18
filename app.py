"""Streamlit interface for the trained protein function classifier."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from protfunc.model import load_artifact, predict_sequences
from protfunc.validation import SequenceValidationError, parse_fasta

st.set_page_config(page_title="ProtFunc ML", page_icon="🧬", layout="centered")

MODEL_PATH = Path(os.getenv("PROTFUNC_MODEL", "models/protfunc.joblib"))


@st.cache_resource
def get_model(path: str):
    """Cache a trusted local model artifact."""

    return load_artifact(path)


st.title("🧬 ProtFunc ML")
st.caption("Sequence-based prediction of the seven top-level enzyme classes")

st.info(
    "This is an educational baseline, not a replacement for experimental annotation, "
    "homology search, or expert review."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model not found at `{MODEL_PATH}`. Run the fetch and train commands "
        "from the README first."
    )
    st.stop()

sequence_text = st.text_area(
    "Paste a protein sequence or FASTA record",
    height=240,
    placeholder=">protein_id\nMSTNPKPQRKTKRNTNRRPQDVKFPGG...",
)
top_k = st.slider("Number of predictions", min_value=1, max_value=7, value=3)

if st.button("Predict function", type="primary", use_container_width=True):
    try:
        records = parse_fasta(sequence_text)
        if len(records) > 25:
            raise SequenceValidationError("The web demo accepts at most 25 FASTA records at once.")
        artifact = get_model(str(MODEL_PATH))
        predictions = predict_sequences(
            artifact, [record.sequence for record in records], top_k=top_k
        )
    except (SequenceValidationError, ValueError, OSError) as exc:
        st.error(str(exc))
    else:
        for record, ranked in zip(records, predictions, strict=True):
            st.subheader(record.identifier)
            best = ranked[0]
            st.metric("Top prediction", best["name"], f"{best['probability']:.1%} confidence")
            st.write(best["description"])
            chart = pd.DataFrame(
                {
                    "Function": [item["name"] for item in ranked],
                    "Probability": [item["probability"] for item in ranked],
                }
            ).set_index("Function")
            st.bar_chart(chart)
            with st.expander("Prediction details"):
                st.dataframe(
                    pd.DataFrame(ranked).assign(
                        probability=lambda frame: frame["probability"].map(
                            lambda value: f"{value:.2%}"
                        )
                    ),
                    hide_index=True,
                    use_container_width=True,
                )

st.divider()
st.caption("Model: amino-acid k-mer TF-IDF + physicochemical descriptors + logistic regression")
