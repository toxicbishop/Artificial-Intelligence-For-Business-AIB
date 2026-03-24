"""
=============================================================================
  AI FOR BUSINESS — Week 05
  Topic : AI and Advertisement
  File  : 05.py

  Approach : Programmatic Advertising
    1. Collect user data from cookies.
    2. Show programmatic ad based on user data.

  Dependencies : flask  (pip install flask)
=============================================================================
"""

# Programmatic Advertising: Deliver targeted ads using cookie-based user interest data
from flask import Flask, request, jsonify, make_response
 
app = Flask(__name__)
 
# Extracts user interest from cookies
def collect_user_data():
   return {"interest": request.cookies.get("interest", "General")}

# Returns ad content based on user data
def programmatic_advertising(user_data):
   return f"Showing ad for {user_data.get('interest', 'General')}"

# Route to set an example interest cookie
@app.route('/')
def index():
   resp = make_response("Welcome to the site!")
   resp.set_cookie('interest', 'Technology')
   return resp

# Route to display the programmatic ad
@app.route('/show_ad')
def show_ad():
   user_data = collect_user_data()
   return jsonify({"ad_message": programmatic_advertising(user_data)})

# Entry point for the Flask server
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)