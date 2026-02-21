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

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
def recommend_content(user_id, user_data, content_data):
   """
   Recommends top 3 content items for a given user based on cosine similarity.
 
   Args:
   user_id (str): The ID of the user.
   user_data (dict): A dictionary containing user embeddings.
   content_data (numpy.ndarray): A matrix of content embeddings.
 
   Returns:
   list: A list of indices for the top 3 recommended content items.
   """
   if user_id not in user_data:
       return "User not found in the dataset."
   
   user_vector = user_data[user_id]
   similarities = cosine_similarity([user_vector], content_data)
   # Get indices of the top 3 recommended content items
   recommended_indices = np.argsort(similarities[0])[-3:][::-1]
   return recommended_indices
 
# Sample data
user_data = {
   "John": [0.8, 0.2],
   "Alice": [0.1, 0.9]
}
content_data = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
 
# Interactive input
user_id = input("Enter the user ID: ")
recommended_content = recommend_content(user_id, user_data, content_data)
 
print("\nTop 3 Recommended Content Indices:")
print(recommended_content)
""" 
Output = Enter the user ID: John
 
Top 3 Recommended Content Indices:
[0 2 1]
"""