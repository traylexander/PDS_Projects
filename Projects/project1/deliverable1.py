import requests
import re
from sentence_transformers import SentenceTransformer, util
from googlesearch import search
from bs4 import BeautifulSoup

# Load BERT model for text similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

# Moz API credentials (replace with your credentials)
MOZ_ACCESS_ID = "your_moz_ID"
MOZ_SECRET_KEY = "your_moz_secret_key"

def get_domain_authority(url):
    """
    Fetch domain authority using Moz API (alternative: Majestic, Ahrefs).
    """
    endpoint = f"https://lsapi.seomoz.com/v2/url_metrics"
    headers = {"Authorization": f"Bearer {MOZ_SECRET_KEY}"}
    data = {"targets": [url]}

    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()["results"][0]["domain_authority"]
    return 0  # If API fails

def get_page_text(url):
    """
    Extract main content from the webpage using BeautifulSoup.
    """
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract readable text
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text() for p in paragraphs])

        return text[:2000]  # Limit text to 2000 chars for efficiency
    except:
        return ""

def semantic_similarity(user_query, retrieved_text):
    """
    Compute similarity between user query and webpage content.
    """
    embedding1 = model.encode(user_query, convert_to_tensor=True)
    embedding2 = model.encode(retrieved_text, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding1, embedding2)
    return similarity.item()  # Return similarity score

from transformers import pipeline

def factual_verification(claim):
    """
    Use Hugging Face zero-shot classification to verify claim validity.
    """
    fact_checker = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    labels = ["entailment", "contradiction", "neutral"]
    result = fact_checker(claim, candidate_labels=labels)

    # Assign scores
    scores = {label: result["scores"][i] for i, label in enumerate(labels)}
    return scores["entailment"]  # Higher means more factual


def stance_detection(user_query, retrieved_text):
    """
    Use Hugging Face zero-shot classification for stance detection.
    """
    stance_classifier = pipeline("zero-shot-classification", model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")

    labels = ["entailment", "contradiction", "neutral"]
    result = stance_classifier(f"{user_query} [SEP] {retrieved_text}", candidate_labels=labels)

    return result["scores"][0]  # Entailment score (higher means agreement)


import requests
from transformers import pipeline

def rate_url(user_query, url):
    """
    Rates a given URL based on credibility, factual alignment, and relevance to the user's query.
    Returns a JSON object with a credibility score and explanation.
    """

    # Step 1: Domain Authority Check (Replace with Moz API)
    domain_authority = get_domain_authority(url)  # Correct function

    # Step 2: Get Content from URL
    page_text = get_page_text(url)  # Corrected function name

    if not page_text:
        return {"score": 0, "explanation": "Failed to retrieve webpage content."}

    # Step 3: Measure Relevance to User Query
    relevance = semantic_similarity(user_query, page_text)

    # Step 4: Fact-Checking the Content
    factuality = factual_verification(page_text)

    # Step 5: Stance Agreement (Does it support the user’s question?)
    stance_score = stance_detection(user_query, page_text)

    # Step 6: Weighted Final Score
    final_score = (0.3 * domain_authority) + (0.3 * relevance) + (0.2 * factuality) + (0.2 * stance_score)

    # Step 7: Explanation
    explanation_parts = []
    if domain_authority > 0.7:
        explanation_parts.append("The source is from a highly authoritative domain.")
    elif domain_authority > 0.4:
        explanation_parts.append("The source has moderate credibility.")
    else:
        explanation_parts.append("The source has low domain authority.")

    if factuality > 0.6:
        explanation_parts.append("The content is well-aligned with factual sources.")
    elif factuality > 0.3:
        explanation_parts.append("The factuality of the content is questionable.")
    else:
        explanation_parts.append("The content lacks strong factual backing.")

    if stance_score > 0.6:
        explanation_parts.append("The article strongly supports the user’s question.")
    elif stance_score > 0.3:
        explanation_parts.append("The article is neutral or has mixed support.")
    else:
        explanation_parts.append("The article contradicts the user’s question.")

    # Step 8: Return Final Output
    result = {
        "score": round(final_score, 2),
        "explanation": " ".join(explanation_parts)
    }

    return result

# Example Usage:
user_prompt = "I just got off an international flight can I come back home to my 1-month-old newborn?"
url = "https://www.bhtp.com/blog/when-safe-to-travel-with-newborn/"

rating = rate_url(user_prompt, url)
print(rating)
