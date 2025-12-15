import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np

def sentiment_pie(df):
    counts = df['sentiment'].value_counts()
    fig = px.pie(values=counts.values, names=counts.index, title="Sentiment Distribution")
    return fig

def sentiment_trend(df):
    df['date'] = pd.to_datetime(df['date'])
    trend = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
    fig = px.line(trend, title="Sentiment Trends")
    return fig

def top_keywords_bar(df):
    all_keywords = [kw for kws in df['keywords'] for kw in kws]
    top = pd.Series(all_keywords).value_counts().head(10)
    fig = px.bar(x=top.index, y=top.values, title="Top Keywords")
    return fig

def keyword_sentiment_heatmap(df):
    exploded = df.explode('keywords')
    pivot = exploded.pivot_table(index='keywords', columns='sentiment', aggfunc='size', fill_value=0)
    fig, ax = plt.subplots()
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)
    return fig

def sentiment_wordcloud(df):
    pos = ' '.join([' '.join(kws) for _, row in df[df['sentiment']=='Positive'].iterrows() for kws in row['keywords']])
    neg = ' '.join([' '.join(kws) for _, row in df[df['sentiment']=='Negative'].iterrows() for kws in row['keywords']])
    neu = ' '.join([' '.join(kws) for _, row in df[df['sentiment']=='Neutral'].iterrows() for kws in row['keywords']])
    wc = WordCloud(width=800, height=400).generate(pos + neg + neu)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    return fig

def summary_cards(df):
    pos_insights = df[df['sentiment']=='Positive']['summary'].head(3).to_list()
    neg_insights = df[df['sentiment']=='Negative']['summary'].head(3).to_list()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Positive Insights")
        for insight in pos_insights:
            st.write(insight)
    with col2:
        st.subheader("Negative Insights")
        for insight in neg_insights:
            st.write(insight)

def section_sentiment_stacked(df):
    pivot = df.pivot_table(index='section', columns='sentiment', aggfunc='size', fill_value=0)
    fig = px.bar(pivot, barmode='stack', title="Sentiment per Section")
    return fig

def cluster_bubble(df):
    if 'cluster' in df.columns and 'embedding' in df.columns:
        from sklearn.decomposition import PCA
        embeddings = np.array(df['embedding'].tolist())
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        df['x'] = reduced[:,0]
        df['y'] = reduced[:,1]
        fig = px.scatter(df, x='x', y='y', color='cluster', size='confidence_score', hover_data=['summary'], title="Topic Clusters")
        return fig
    return None