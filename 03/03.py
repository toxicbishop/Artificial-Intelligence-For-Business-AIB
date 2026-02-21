"""
=============================================================================
  AI FOR BUSINESS — Week 03
  Topic : AI & Marketing — Data-Driven Website Content & SEO Optimisation
  File  : 03.py

  Approach : AI-Driven Marketing System
    1. Generate structured website content for a given business
    2. Run TF-IDF SEO analysis to surface top keywords (ngram 1-2)
    3. Produce personalised promotional emails per customer interest

  Dependencies : scikit-learn  (pip install scikit-learn)
=============================================================================
"""

"""AI and Marketing: Develop data-driven content for a given business organization (Web site). 
Optimize Website content for search engines. Send emails to customers with personalized content/activity."""
import asyncio.exceptions
import asyncio.exceptions
from sklearn.feature_extraction.text import TfidfVectorizer 

# 1️. Generate Website Content
def generate_website_content(business_name, industry, services): 
    content = f""" 
    Welcome to {business_name}! 
 
    We are a leading {industry} company dedicated to delivering high-quality services. 
    Our core services include {services}. 
 
    We help businesses grow through innovative strategies, 
    customer-focused solutions, and data-driven marketing techniques. 
 
    Contact us today to boost your brand visibility and achieve business success. 
    """ 
    return content

# 2️. SEO Optimization using TF-IDF 

def seo_optimization(text): 
    vectorizer = TfidfVectorizer( 
        stop_words='english', 
        max_features=20, 
        ngram_range=(1,2) 
    ) 
 
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out() 
 
    sorted_keywords = sorted( 
        keywords, 
        key=lambda x: X[0, vectorizer.vocabulary_[x]], 
        reverse=True 
    ) 
    return sorted_keywords[:10]

# 3. Personalized Email Generator 
def generate_personalized_email(customer_name, interest, business_name): 
    email = f""" 
    Subject: Special Offer Just for You, {customer_name}! 
 
    Dear {customer_name}, 
 
    We noticed your interest in {interest}. 
    At {business_name}, we provide customized solutions to help you achieve your goals. 
 
    As a valued customer, we are offering exclusive benefits tailored to your interests. 
 
    Visit our website to explore more exciting opportunities. 
 
    Best Regards, 
    {business_name} Team 
    """ 
    return email

# 4️. Main Program (Interactive) 
print("====== AI Driven Marketing System ======\n") 
business_name = input("Enter Business Name: ") 
industry = input("Enter Industry Type: ") 
services = input("Enter Main Services (comma separated): ") 
# Generate Website Content 
website_content = generate_website_content(business_name, industry, services) 
print("\n Generated Website Content:\n") 
print(website_content)

# SEO Optimization 

keywords = seo_optimization(website_content) 
print("\n Top SEO Keywords:\n") 
for i, word in enumerate(keywords, 1): 
    print(f"{i}. {word}") 
# Personalized Email Section 
print("\n====== Personalized Email Section ======\n") 
customer_name = input("Enter Customer Name: ") 
interest = input("Enter Customer Interest: ") 
email = generate_personalized_email(customer_name, interest, business_name) 
print("\n Generated Personalized Email:\n") 
print(email)

"""Output:
====== AI Driven Marketing System ======

Enter Business Name: Google
Enter Industry Type: Technology
Enter Main Services (comma separated): Search, YouTube, Android, Chrome, Gmail 

Generated Website Content:

 
Welcome to Google! 
 
We are a leading Technology company dedicated to delivering high-quality services. 
Our core services include Search, YouTube, Android, Chrome, Gmail. 
 
We help businesses grow through innovative strategies, 
customer-focused solutions, and data-driven marketing techniques.

Contact us today to boost your brand visibility and achieve business success.


Top SEO Keywords:

1. services
2. achieve
3. achieve business
4. android
5. android chrome
6. boost
7. boost brand
8. brand
9. brand visibility
10. business

====== Personalized Email Section ======

Enter Customer Name: Pranav
Enter Customer Interest: Search & Information

Generated Personalized Email:


Subject: Special Offer Just for You, Pranav!

Dear Pranav,

We noticed your interest in Search & Information.
At Google, we provide customized solutions to help you achieve your goals.

As a valued customer, we are offering exclusive benefits tailored to your interests.

Visit our website to explore more exciting opportunities.

Best Regards,
Google Team """