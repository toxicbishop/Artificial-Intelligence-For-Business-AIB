"""
=============================================================================
  AI FOR BUSINESS — Week 08
  Topic : Content Marketing and AI
  File  : 08.py

  Approach : intelligent product searches
    1. Content Marketing using AI (String matching).
    2. Text-Based E-Commerce Product Search.
    3. Customer Service – Voice Search.
    4. Customer Service – Visual Search.

  Dependencies : SpeechRecognition, Pillow, pyaudio
=============================================================================
"""
# Part 1: Content Marketing using AI
print("Part 1: Content Marketing using AI: -")
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

user_topic = input("Enter a topic you're interested in (e.g., 'AI'): ").lower()
matched = False
print("\nSuggested Content Titles:")
for content in content_list:
    if any(word in content.lower() for word in user_topic.split()):
        print("👉", content)
        matched = True
if not matched:
    print("❌ Sorry, no matching content found for your topic.")

# Part 2: Text-Based E-Commerce Product Search
print("\nPart 2: Text-Based E-Commerce Product Search: -")
products = ["Smartphone", "Headphones", "Laptop", "Smartwatch", "Camera", "Refrigerator", 
            "Air Conditioner", "Washing Machine","Printer", "Router", "Hard Drive", "Smart Light", "Drone"]

search_input = input("Enter product to search: ").lower()
matched = False
print("\n🔎 Matching Products:")
for product in products:
    if search_input in product.lower():
        print("✅", product)
        matched = True
if not matched:
    print("❌ No matching products found.")

# Part 3: Customer Service – Voice Search
print("\nPart 3: Customer Service – Voice Search: -")
import speech_recognition as sr
import time

products = ["Smartphone", "Headphones", "Laptop", "Smartwatch", "Camera"]
recognizer = sr.Recognizer()

class MockAudio: pass
def mock_listen(source): return MockAudio()
def mock_recognize(audio): return "smartphone"
recognizer.listen = mock_listen
recognizer.recognize_google = mock_recognize

class MockMicrophone:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): pass
sr.Microphone = MockMicrophone

try:
    with sr.Microphone() as source:
        print("🎤 Speak your product search query...")
        time.sleep(1)
        audio = recognizer.listen(source)
        voice_input = recognizer.recognize_google(audio)
        print("✅ You said:", voice_input)
        
        found = False
        for product in products:
            if voice_input.lower() in product.lower():
                print("🛒 Matched Product:", product)
                found = True
        if not found:
            print("❌ No matching product found.")
except Exception as e:
    print(f"⚠️ Voice input failed. Try again or check your mic. Error: {e}")

# Part 4: Customer Service – Visual Search
print("\nPart 4: Customer Service – Visual Search: -")
from PIL import Image
import os

product_images = {
    "smartphone": "../images/smartphone.jpg",
    "laptop": "../images/laptop.jpg",
    "camera": "../images/camera.jpg"
}

uploaded_path = input("📁 Enter path to your image (e.g., '../images/smartphone.jpg'): ")

actual_path = uploaded_path
if not os.path.exists(actual_path) and os.path.exists(uploaded_path.replace("../", "")):
    actual_path = uploaded_path.replace("../", "")

if os.path.exists(actual_path):
    try:
        img = Image.open(actual_path)
        img.show()
    except:
        pass
    
    simulated_tags = ["smartphone", "laptop", "camera"]
    print("\n🧠 Simulated AI Detected Tags:", simulated_tags)
    
    found_product = False
    for product, path in product_images.items():
        if product in uploaded_path.lower():
            print("🛒 Matched Product:", product)
            found_product = True
    
    if not found_product:
        print("❌ No product matched visual tags.")
else:
    print("❌ Image file not found! Please check the path.")

"""Output:

Part 1: Content Marketing using AI: -
Enter a topic you're interested in (e.g., 'AI'): AI

Suggested Content Titles:
👉 How AI is transforming digital marketing
👉 Top 5 AI tools for content writers
👉 Benefits of AI in SEO and keyword optimization
👉 Creating personalized content using AI
👉 AI trends shaping future of content marketing
👉 Building brand identity using generative AI
👉 AI in content curation for social media
👉 Creating viral content using AI-powered insights


Part 2: Text-Based E-Commerce Product Search: -
Enter product to search: Laptop

🔎 Matching Products:
✅ Laptop

Part 3: Customer Service – Voice Search: -
🎤 Speak your product search query...
✅ You said: smartphone
🛒 Matched Product: Smartphone

Part 4: Customer Service – Visual Search: -
📁 Enter path to your image (e.g., '../images/smartphone.jpg'): ../images/camera.jpg

🧠 Simulated AI Detected Tags: ['smartphone', 'laptop', 'camera']
🛒 Matched Product: camera
"""