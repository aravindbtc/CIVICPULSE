from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from backend.ai import analyze_comment, get_recommendations, get_overall_summary, translate_to_english, get_embedding
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def process_single(row):
    original = row.get('comment', '')
    lang = row.get('language', 'en')
    translated = translate_to_english(original, lang)
    sentiment, confidence, summary, keywords = analyze_comment(translated)
    embedding = get_embedding(translated)
    priority = "High" if sentiment == "Negative" and confidence > 70 else "Normal"
    return {
        "original_comment": original,
        "translated_comment": translated,
        "sentiment": sentiment,
        "confidence": float(confidence),  # Ensure float
        "summary": summary,
        "keywords": ','.join(keywords) if keywords else '',
        "section": row.get('section', 'Unknown'),
        "priority": priority,
        "policy_recommendations": [],
        "draft_version": row.get('draft_version', 'v1'),
        "date": row.get('date', 'Unknown'),
        "stakeholder": row.get('stakeholder', ''),
        "embedding": embedding
    }

def process_comments_batch(df, batch_size=100):
    required_cols = ['comment', 'language', 'section', 'draft_version', 'date', 'stakeholder']
    for col in required_cols:
        if col not in df.columns:
            df[col] = 'Unknown' if col in ['section', 'date'] else 'en' if col == 'language' else 'v1' if col == 'draft_version' else ''
    with Pool(processes=4) as pool:
        batches = [df[i:i + batch_size] for i in range(0, len(df), batch_size)]
        results = []
        for batch in batches:
            batch_results = pool.map(process_single, batch.to_dict('records'))
            results.extend(batch_results)

    embeddings = np.array([eval(r['embedding']) for r in results])  # Convert string to list
    if len(embeddings) > 1:
        kmeans = KMeans(n_clusters=min(5, len(embeddings)), random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        for i, cluster in enumerate(clusters):
            results[i]['cluster'] = int(cluster)
    else:
        for r in results:
            r['cluster'] = None

    negative_comments = [r['translated_comment'] for r in results if r['sentiment'] == 'Negative']
    recommendations = get_recommendations(negative_comments)
    for r in results:
        if r['sentiment'] == 'Negative':
            r['policy_recommendations'] = recommendations

    all_summaries = [r['summary'] for r in results]
    overall_summary = get_overall_summary(all_summaries)
    logger.debug("Overall summary: %s", overall_summary)

    return results

def process_single_comment(comment, language='en', section='Unknown', draft_version='v1', date='Unknown', stakeholder=''):
    row = {
        'comment': comment,
        'language': language,
        'section': section,
        'draft_version': draft_version,
        'date': date,
        'stakeholder': stakeholder
    }
    processed = process_single(row)
    processed['cluster'] = None
    processed['policy_recommendations'] = get_recommendations([processed['translated_comment']]) if processed['sentiment'] == 'Negative' else []
    return processed