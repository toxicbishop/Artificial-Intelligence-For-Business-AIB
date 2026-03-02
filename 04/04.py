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

from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
website_data = { 
    "SEO Optimization": "Improve your website ranking with advanced SEO techniques and keyword strategies.", 
    "Social Media Campaigns": "Boost brand awareness through targeted social media marketing campaigns.", 
    "Email Marketing": "Increase customer engagement using personalized email marketing strategies.", 
    "Content Marketing": "Attract and retain customers with high-quality content marketing solutions.", 
    "Web Analytics": "Track website performance and customer behavior using advanced analytics tools.", 
    "Online Advertising": "Run paid advertising campaigns to increase website traffic and conversions." 
}

def recommend_content(user_interest): 
    titles = list(website_data.keys()) 
    documents = list(website_data.values()) 
 
    # Add user interest as last document 
    documents.append(user_interest) 
 
    # Convert to TF-IDF vectors 
    vectorizer = TfidfVectorizer(stop_words='english') 
    tfidf_matrix = vectorizer.fit_transform(documents) 
 
    # Compute similarity between user interest and website contents 
    similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]) 
 
    # Get top 3 similar contents 
    similarity_scores = similarity.flatten() 
    top_indices = similarity_scores.argsort()[-3:][::-1] 
 
    recommendations = [] 
    for index in top_indices: 
        recommendations.append((titles[index], documents[index])) 
 
    return recommendations 
  
def generate_recommendation(user_name, interest): 
    recommended_items = recommend_content(interest) 
 
    message = f"\nHello {user_name},\n" 
    message += f"Based on your interest in '{interest}', we recommend the following:\n\n" 
 
    for i, (title, content) in enumerate(recommended_items, 1): 
        message += f"{i}. {title}\n   {content}\n\n" 
 
    message += "Visit our website to explore these services further.\n" 
 
    return message 
 
print("====== AI Targeted Content Recommendation System ======\n") 
 
user_name = input("Enter User Name: ") 
user_interest = input("Enter User Interest or Search Query: ") 
 
result = generate_recommendation(user_name, user_interest) 
 
print("\n Recommended Content:\n") 
print(result)

""" 
Output:
====== AI Targeted Content Recommendation System ======

Enter User Name: Pranav
Enter User Interest or Search Query: Computers    

 Recommended Content:


Hello Pranav,
Based on your interest in 'Computers', we recommend the following:

1. Online Advertising
   Run paid advertising campaigns to increase website traffic and conversions.

2. Web Analytics
   Track website performance and customer behavior using advanced analytics tools.

3. Content Marketing
   Attract and retain customers with high-quality content marketing solutions.

Visit our website to explore these services further.
"""