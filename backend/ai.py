from ollama import Client
import re
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

client = Client(host='http://localhost:11434')

def translate_to_english(text, language):
    return text  # Placeholder

def analyze_comment(comment: str):
    comments = [c.strip() for c in comment.split('\n') if c.strip()]
    if not comments:
        return 'Neutral', 50, 'No valid comment provided', []

    sentiments, confidences, summaries, all_keywords = [], [], [], []
    for single_comment in comments:
        prompt = f"""
        Analyze the following comment and return ONLY a valid JSON object with four fields:
        - sentiment: "Positive", "Negative", or "Neutral"
        - confidence: Integer from 0 to 100
        - summary: A one-sentence summary of the comment
        - keywords: A list of 2-5 keywords
        Comment: '{single_comment}'
        Example: {{"sentiment": "Positive", "confidence": 85, "summary": "The comment is positive.", "keywords": ["policy", "excellent"]}}
        """
        try:
            logger.debug("Sending prompt to LLaMA: %s", prompt)
            response = client.chat(model='llama3:8b', messages=[{'role': 'user', 'content': prompt}])
            result = response['message']['content'].strip()
            logger.debug("LLaMA response: %s", result)
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                sentiment = data.get('sentiment', 'Neutral')
                confidence = float(data.get('confidence', 50))  # Ensure float
                summary = data.get('summary', 'No summary')
                keywords = data.get('keywords', [])
                if not isinstance(keywords, list):
                    keywords = [k.strip() for k in str(keywords).split(',') if k.strip()]
            else:
                raise json.JSONDecodeError("No JSON found", result, 0)
            sentiments.append(sentiment)
            confidences.append(confidence)
            summaries.append(summary)
            all_keywords.extend(keywords)
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error: %s", str(e))
            sentiments.append('Neutral')
            confidences.append(50.0)
            summaries.append('Parsing failed')
            all_keywords.extend([])
        except Exception as e:
            logger.error("Error: %s", str(e))
            sentiments.append('Neutral')
            confidences.append(50.0)
            summaries.append('Analysis error')
            all_keywords.extend([])

    sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for s in sentiments:
        sentiment_counts[s] += 1
    aggregated_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    aggregated_confidence = sum(confidences) / len(confidences) if confidences else 50.0
    aggregated_summary = f"Multiple comments express {'positive' if aggregated_sentiment == 'Positive' else 'negative' if aggregated_sentiment == 'Negative' else 'neutral'} sentiments."
    keyword_freq = {k: all_keywords.count(k) for k in set(all_keywords)}
    aggregated_keywords = sorted(keyword_freq, key=keyword_freq.get, reverse=True)[:5]

    return aggregated_sentiment, aggregated_confidence, aggregated_summary, aggregated_keywords

def get_sentiment(comment: str):
    sentiment, confidence, _, _ = analyze_comment(comment)
    return sentiment, confidence

def get_summary(comment: str):
    _, _, summary, _ = analyze_comment(comment)
    return summary

def get_keywords(comment: str):
    _, _, _, keywords = analyze_comment(comment)
    return keywords

def get_embedding(comment: str):
    import numpy as np
    np.random.seed(abs(hash(comment)) % (10 ** 8))
    return ','.join(map(str, np.random.rand(128).tolist()))

def get_recommendations(negative_comments):
    if not negative_comments:
        return []
    combined = ' '.join(negative_comments)[:2000]
    prompt = f"""
    Suggest 1-3 actionable policy recommendations based on: {combined}
    Return ONLY a valid JSON list, e.g., ["Rec 1", "Rec 2"].
    """
    try:
        response = client.chat(model='llama3:8b', messages=[{'role': 'user', 'content': prompt}])
        result = response['message']['content'].strip()
        logger.debug("Recommendations response: %s", result)
        try:
            return json.loads(result)[:3] if isinstance(json.loads(result), list) else []
        except json.JSONDecodeError:
            logger.warning("JSON parse failed for recommendations")
            return [r.strip() for r in result.split('\n') if r.strip()][:3]
    except Exception as e:
        logger.error("Recommendations error: %s", str(e))
        return []

def get_overall_summary(summaries):
    if not summaries:
        return "No summaries."
    combined = ' '.join(summaries)[:2000]
    prompt = f"""
    Summarize: {combined}
    Return ONLY a concise paragraph.
    """
    try:
        response = client.chat(model='llama3:8b', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'].strip()
    except Exception as e:
        logger.error("Summary error: %s", str(e))
        return "Summary generation failed."