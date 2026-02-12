import streamlit as st
from src.search import ContractSearchEngine

# Page config
st.set_page_config(page_title="Contract Clause Search Engine", layout="wide")

st.title(" Semantic Contract Clause Search")
st.write("Search across 19,000+ legal clauses using semantic similarity.")

# Load engine once
@st.cache_resource
def load_engine():
    return ContractSearchEngine()

engine = load_engine()

# User input
query = st.text_input("Enter your query:", placeholder="e.g., termination clause")

top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        results = engine.search(query, top_k=top_k)

        st.subheader(" Results")

        for i, r in enumerate(results, 1):
            st.markdown(f"### {i}. {r['contract']}")
            st.write(f"**Similarity Score:** {r['score']:.3f}")
            st.write(r['text'])
            st.markdown("---")
