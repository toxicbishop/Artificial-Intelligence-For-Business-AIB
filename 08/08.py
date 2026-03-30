"""
=============================================================================
  AI FOR BUSINESS — Week 08
  Topic : AI Product Search & Content Marketing
  File  : 08.py

  Description:
    1. Content Marketing using AI (Keyword-based title suggestions).
    2. Text-Based E-Commerce Product Search.
    3. Customer Service – Voice Search (Speech to Text).
    4. Customer Service – Visual Search (Image Path Matching).

  Dependencies : 
    pip install SpeechRecognition Pillow pyaudio
=============================================================================
"""

"""
Demonstrate content development using an AI-powered 
language platform. Customer Service and AI: Demonstrate intelligent product searches 
and discoveries possible across text, voice, and visual searches on e-commerce websites 
using AI tools. 
"""

# Part 1: Content Marketing using AI 
print("--- Part 1: Content Marketing using AI ---")
content_list = [ 
    "How AI is transforming digital marketing", 
    "Top 5 AI tools for content writers", 
    "Benefits of AI in SEO and keyword optimization", 
    "Using Chatbots for customer engagement", 
    "Creating personalized content using AI", 
    "Natural language processing in ad copy writing", 
    "AI trends shaping future of content marketing", 
    "Building brand identity using generative AI", 
    "AI in content curation for social media", 
    "How predictive analytics drives content strategy", 
    "Creating viral content using AI-powered insights" 
] 
user_topic = input("Enter a topic: ").lower() 
print("\n Suggested Content Titles:") 
matched = False 
for content in content_list: 
    if any(word in content.lower() for word in user_topic.split()): 
        print("*", content) 
        matched = True 
if not matched: 
    print("No matching content found") 

# Part 2: Text-Based E-Commerce Product Search 
print("\n--- Part 2: Text-Based E-Commerce Product Search ---")
products = ["Smartphone", "Headphones", "Laptop", "Smartwatch", "Camera", 
            "Refrigerator", "Air Conditioner", "Washing Machine", 
            "Printer", "Router", "Hard Drive", "Smart Light", "Drone"] 
search_input = input("Enter product: ").lower() 
print("\n Matching Products:") 
matched = False 
for product in products: 
    if search_input in product.lower(): 
        print("Correct", product) 
        matched = True 
if not matched: 
    print("No match found") 

# Part 3: Customer Service – Voice Search 
print("\n--- Part 3: Customer Service – Voice Search ---")
import speech_recognition as sr 
recognizer = sr.Recognizer() 
try: 
    with sr.Microphone() as source: 
        print("Speak now...") 
        audio = recognizer.listen(source) 
        voice_input = recognizer.recognize_google(audio) 
        print("You said:", voice_input) 
        
        products_voice = ["Smartphone", "Headphones", "Laptop", "Smartwatch", "Camera"] 
        found = False 
        for product in products_voice: 
            if voice_input.lower() in product.lower(): 
                print("Matched:", product) 
                found = True 
        if not found: 
            print("No match found") 
except Exception as e: 
    print("Error:", e) 

# Part 4: Customer Service – Visual Search 
print("\n--- Part 4: Customer Service – Visual Search ---")
from PIL import Image 
import os 
product_images = { 
    "smartphone": "images/smartphone.jpg", 
    "laptop": "images/laptop.jpg", 
    "camera": "images/camera.jpg" 
} 
uploaded_path = input("Enter image path (e.g., images/smartphone.jpg): ") 
if os.path.exists(uploaded_path): 
    img = Image.open(uploaded_path) 
    img.show()
    print("\n Simulated AI Tags: smartphone, laptop, camera") 
    found = False 
    for product in product_images: 
        if product in uploaded_path.lower(): 
            print("Matched Product:", product) 
            found = True 
    if not found: 
        print("No match found") 
else: 
    print(" File not found")