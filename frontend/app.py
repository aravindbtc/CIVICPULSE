import sys
import os
import streamlit as st
import requests
import pandas as pd
import json
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import report, viz
print("Current Working Directory:", os.getcwd())
print("Python Path:", sys.path)
print("utils Folder Exists:", os.path.exists("utils"))
print("viz.py Exists:", os.path.exists("utils/viz.py"))
print("report.py Exists:", os.path.exists("utils/report.py"))
try:
    from utils.viz import (
        sentiment_pie, sentiment_trend, top_keywords_bar, keyword_sentiment_heatmap,
        sentiment_wordcloud, summary_cards, section_sentiment_stacked, cluster_bubble
    )
    from utils.report import generate_pdf_report, generate_excel_report
    print("Successfully imported utils.viz and utils.report")
except Exception as e:
    print(f"Failed to import utils: {e}")
    st.error(f"Failed to import utils: {e}")
    st.stop()

st.title("e-Consultation Analysis Platform")

# File Upload
uploaded_file = st.file_uploader("Upload Comments CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file:
    with st.spinner("Processing..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            response = requests.post("http://localhost:8000/upload", files=files)
            if response.status_code == 200:
                st.success("File processed successfully!")
            else:
                st.error(f"Upload failed: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to backend. Ensure FastAPI is running on http://localhost:8000.")
            st.stop()

# Text Input for Sentiment Analysis
st.subheader("Analyze Single Comment")
comment_text = st.text_area("Paste your comment here:", height=100)
comment_language = st.selectbox("Comment Language", ["en", "hi"])
if st.button("Analyze Comment"):
    if comment_text:
        with st.spinner("Analyzing..."):
            payload = {"comment": comment_text, "language": comment_language}
            try:
                response = requests.post("http://localhost:8000/analyze-text", json=payload)
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']}%)")
                    st.write(f"Summary: {result['summary']}")
                    st.write(f"Keywords: {', '.join(result['keywords'])}")
                else:
                    st.error(f"Analysis failed: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to backend. Ensure FastAPI is running on http://localhost:8000.")
    else:
        st.warning("Please enter a comment.")

# Filters for File-Based Analysis
st.subheader("Filter Analysis")
draft_version = st.selectbox("Draft Version", ["All", "v1", "v2"])
section = st.selectbox("Section", ["All", "Section1", "Section2", "Section3"])
if draft_version == "All": draft_version = None
if section == "All": section = None

# Get Data for Visualizations
try:
    data_response = requests.get("http://localhost:8000/analysis", params={"draft_version": draft_version, "section": section})
    if data_response.status_code == 200:
        data = json.loads(data_response.content)
        df = pd.DataFrame(data)
        
        if not df.empty:
            st.subheader("Sentiment Distribution")
            st.plotly_chart(sentiment_pie(df))
            
            st.subheader("Sentiment Trends Over Time")
            st.plotly_chart(sentiment_trend(df))
            
            st.subheader("Top Keywords")
            st.plotly_chart(top_keywords_bar(df))
            
            st.subheader("Keyword vs Sentiment Heatmap")
            st.plotly_chart(keyword_sentiment_heatmap(df))
            
            st.subheader("Sentiment Word Cloud")
            st.pyplot(sentiment_wordcloud(df))
            
            st.subheader("Summary Cards")
            summary_cards(df)
            
            st.subheader("Sentiment per Section")
            st.plotly_chart(section_sentiment_stacked(df))
            
            st.subheader("Topic Clusters")
            st.plotly_chart(cluster_bubble(df))
            
            st.subheader("Stakeholder Contributions")
            if 'stakeholder' in df.columns:
                df['contribution_score'] = df['original_comment'].str.len()
                top_stakeholders = df.groupby('stakeholder')['contribution_score'].sum().nlargest(5)
                st.table(top_stakeholders)
                for sh, score in top_stakeholders.items():
                    badge = "Gold" if score > 1000 else "Silver" if score > 500 else "Bronze"
                    st.write(f"{sh}: {badge} Badge")
            
            st.subheader("Overall Summary")
            overall_summary = "Overall summary generated from all comments."
            st.write(overall_summary)
            
            if st.button("Generate PDF Report"):
                pdf_path = generate_pdf_report(df)
                with open(pdf_path, "rb") as f:
                    st.download_button("Download PDF", f, file_name="report.pdf")
            
            if st.button("Generate Excel Report"):
                excel_path = generate_excel_report(df)
                with open(excel_path, "rb") as f:
                    st.download_button("Download Excel", f, file_name="report.xlsx")
    else:
        st.error(f"Analysis failed: {data_response.text}")
except requests.exceptions.ConnectionError:
    st.error("Cannot connect to backend. Ensure FastAPI is running on http://localhost:8000.")
    st.stop()