"""
=============================================================================
  AI FOR BUSINESS — Week 04
  Topic : AI & Marketing — Personalized Content Recommendation System
  File  : 04.py

  Approach : Hybrid Recommendation Engine
    1. Content-Based Filtering  (TF-IDF cosine similarity)
    2. Collaborative Filtering  (user-item matrix cosine similarity)
    3. Hybrid Scoring            final = 0.6 * content + 0.4 * collab

  Dependencies : numpy, scikit-learn  (pip install numpy scikit-learn)
=============================================================================
"""

import sys
import numpy as np
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── 1. CONTENT CATALOG ────────────────────────────────────────────────────────

ARTICLES = [
    {"id": "A1", "category": "AI",        "title": "Generative AI for Business",        "text": "generative AI large language models GPT content automation business productivity"},
    {"id": "A2", "category": "AI",        "title": "Machine Learning in Analytics",     "text": "machine learning predictive analytics decision tree random forest data science"},
    {"id": "A3", "category": "AI",        "title": "NLP for Customer Support",          "text": "natural language processing chatbot sentiment analysis customer support NLP"},
    {"id": "A4", "category": "Marketing", "title": "Personalisation at Scale with AI",  "text": "AI personalisation e-commerce recommendation marketing customer segmentation"},
    {"id": "A5", "category": "Marketing", "title": "Email Marketing Automation",        "text": "email marketing automation CRM segmentation open rate click conversion campaign"},
    {"id": "A6", "category": "Finance",   "title": "AI-Driven Algorithmic Trading",     "text": "algorithmic trading AI quantitative finance stock market reinforcement learning"},
    {"id": "A7", "category": "Finance",   "title": "FinTech and Mobile Banking",        "text": "fintech mobile banking digital payments neo-bank financial inclusion blockchain"},
    {"id": "A8", "category": "Health",    "title": "AI in Medical Imaging",             "text": "AI medical imaging radiology deep learning cancer detection diagnosis healthcare"},
    {"id": "A9", "category": "Health",    "title": "Wearables and Preventive Care",     "text": "wearable smartwatch health data fitness tracker heart rate prevention biosensor"},
    {"id": "A10","category": "Travel",    "title": "AI-Powered Travel Planning",        "text": "AI travel planning personalisation booking recommendation itinerary tourism"},
]

# ─── 2. USER PROFILES ─────────────────────────────────────────────────────────

USERS = [
    {"id": "U1", "name": "Priya  (Data Scientist)",  "browsed": ["A1", "A2"],       "ratings": {"A1": 5, "A2": 4}},
    {"id": "U2", "name": "Rahul  (Mkt. Manager)",    "browsed": ["A4", "A5"],       "ratings": {"A4": 5, "A5": 3}},
    {"id": "U3", "name": "Ananya (Med. Student)",    "browsed": ["A8", "A9"],       "ratings": {"A8": 5, "A9": 4}},
    {"id": "U4", "name": "Karan  (Fin. Analyst)",    "browsed": ["A6", "A7"],       "ratings": {"A6": 5, "A7": 4}},
    {"id": "U5", "name": "Neha   (Travel Blogger)",  "browsed": ["A10", "A4"],      "ratings": {"A10": 5, "A4": 3}},
]

# ─── 3. BUILD TF-IDF MATRIX ────────────────────────────────────────────────────

corpus = [a["text"] + " " + a["category"] for a in ARTICLES]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)          # shape: (10 articles, n_features)
article_idx  = {a["id"]: i for i, a in enumerate(ARTICLES)}

# ─── 4. BUILD USER-ITEM MATRIX ─────────────────────────────────────────────────

ui_matrix = np.zeros((len(USERS), len(ARTICLES)))
for u_i, user in enumerate(USERS):
    for art_id, rating in user["ratings"].items():
        ui_matrix[u_i, article_idx[art_id]] = rating / 5.0   # normalise to [0, 1]

# ─── 5. RECOMMENDATION FUNCTIONS ──────────────────────────────────────────────

def content_scores(user):
    """Cosine similarity between the user's read articles and all articles."""
    indices = [article_idx[aid] for aid in user["browsed"] if aid in article_idx]
    profile = np.asarray(tfidf_matrix[indices].mean(axis=0))           # avg TF-IDF vector
    return cosine_similarity(profile, tfidf_matrix).flatten()

def collab_scores(user_index):
    """Weighted average of other users' ratings (similarity-weighted)."""
    user_vec = ui_matrix[user_index].reshape(1, -1)
    sims     = cosine_similarity(user_vec, ui_matrix).flatten()
    sims[user_index] = 0                                                # exclude self
    if sims.max() == 0:
        return np.zeros(len(ARTICLES))
    return sims.dot(ui_matrix) / (sims.sum() + 1e-9)

def recommend(user, user_index, top_n=3):
    """Hybrid score = 0.6 × content + 0.4 × collab, exclude seen articles."""
    scores  = 0.6 * content_scores(user) + 0.4 * collab_scores(user_index)
    seen    = set(user["browsed"])
    ranked  = sorted(
        [(scores[i], i, a) for i, a in enumerate(ARTICLES) if a["id"] not in seen],
        key=lambda x: x[0],
        reverse=True
    )
    return [(s, a) for s, _, a in ranked[:top_n]]

# ─── 6. DEMO OUTPUT ───────────────────────────────────────────────────────────

print("=" * 60)
print("  AI Content Recommendation System — Week 04")
print("=" * 60)

for u_i, user in enumerate(USERS):
    print(f"\n  User : {user['name']}")
    print(f"  Read : {', '.join(user['browsed'])}")
    print(f"  {'#':<3} {'Article ID':<6} {'Score':<8} Category    Title")
    print(f"  {'-' * 55}")
    for rank, (score, art) in enumerate(recommend(user, u_i), 1):
        print(f"  {rank:<3} {art['id']:<6} {score:.4f}   {art['category']:<11} {art['title']}")

print("\n" + "=" * 60)
print("  Hybrid formula: 0.6 × Content-Based + 0.4 × Collaborative")
print("=" * 60)
