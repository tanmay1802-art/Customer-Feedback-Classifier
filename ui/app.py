import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Customer Feedback Classifier", page_icon="✈️")
st.title("Customer Feedback Classifier")
st.caption("Classifies a tweet, then drafts a reply for confident negative ones.")

tweet = st.text_area("Customer tweet", height=120, max_chars=1000)

if st.button("Analyze", type="primary", disabled=not tweet.strip()):
    try:
        with st.spinner("Analyzing... (first reply can take 10-30 seconds)"):
            resp = requests.post(f"{API_URL}/analyze", json={"text": tweet}, timeout=150)
    except requests.RequestException:
        st.error("Cannot reach the API. Is uvicorn running on port 8000?")
        st.stop()

    if resp.status_code == 400:
        st.warning(resp.json().get("detail", "Invalid text"))
        st.stop()
    if resp.status_code != 200:
        st.error(f"API error {resp.status_code}")
        st.stop()

    data = resp.json()
    col1, col2 = st.columns(2)
    col1.metric("Sentiment", data["label"])
    col2.metric("Confidence", f"{data['confidence']:.0%}")
    st.progress(data["confidence"], text=f"Threshold: {data['threshold']:.0%}")
    st.bar_chart(data["probabilities"])

    action = data["action"]
    if action == "draft_reply":
        st.success("Draft reply (review before sending)")
        st.code(data["reply"], language=None)
    elif action == "human_review":
        st.warning("Human review needed: the model is not confident enough.")
        if data.get("reply_error"):
            st.info(data["reply_error"])
    else:
        st.info("No reply needed.")
