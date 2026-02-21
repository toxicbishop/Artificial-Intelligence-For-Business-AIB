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
from flask import Flask, request, jsonify, make_response
 
app = Flask(__name__)
 
# Collect user data from cookies
def collect_user_data():
   cookies = request.cookies
   user_data = {
       "interest": cookies.get("interest", "General")  # Default to "General" if no cookie is found
   }
   return user_data
 
# Show programmatic ad based on user data
def programmatic_advertising(user_data):
   return f"Showing ad for {user_data.get('interest', 'General')}"
 
@app.route('/')
def index():
   # Example of setting a cookie (usually this would happen after a user action)
   resp = make_response("Welcome to the site!")
   resp.set_cookie('interest', 'Technology')  # Setting a cookie with 'interest' = 'Technology'
   return resp
 
@app.route('/show_ad')
def show_ad():
   user_data = collect_user_data()
   ad_message = programmatic_advertising(user_data)
   return jsonify({"ad_message": ad_message})
 
# Run the Flask app and allow access over the local network
if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
 