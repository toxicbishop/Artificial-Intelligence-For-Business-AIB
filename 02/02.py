from transformers import pipeline

# Load GPT-2 Medium
text_generator = pipeline("text-generation", model="gpt2-medium")

user_preferences = {
    "John": "Technology",
    "Alice": "Health",
    "Emma": "Finance",
    "Michael": "Education"
}

def generate_personalized_email(user, purpose, your_name):
    interest = user_preferences.get(user, "Technology")  # Default to Technology

    prompt = (
        f"Subject: Exciting Insights on {purpose}!\n\n"
        f"Dear {user},\n\n"
        f"I hope you're having a great day! I recently came across some interesting trends in {purpose} "
        f"that align with your interest in {interest}. For example, did you know that {purpose} is "
        f"evolving rapidly with new developments every day?\n\n"
        f"I'd love to hear your thoughts on this. Have you come across any recent articles or insights?\n\n"
        f"Looking forward to your response!\n\n"
        f"Best regards,\n{your_name}"
    )

    generated_email = text_generator(
        prompt,
        max_length=200,
        num_return_sequences=1,
        temperature=0.7,
        top_k=40,
        top_p=0.9
    )[0]['generated_text']

    return generated_email

if __name__ == "__main__":
    # --- Example usage ---
    user = input("Enter the recipient's name: ")
    purpose = input("Enter the purpose of the email (e.g., product launch, thank you): ")
    your_name = input("Enter your Name: ")

    generated_email = generate_personalized_email(user, purpose, your_name)
    print("\nGenerated Email:\n")
    print(generated_email)

    #Output
    #Generated Email:
    #Subject: Exciting Insights on product launch!
    #Dear John,
    #I hope you're having a great day! I recently came across some interesting trends in product launch that align with your interest in Technology. For example, did you know that product launch is evolving rapidly with new developments every day?
    #I'd love to hear your thoughts on this. Have you come across any recent articles or insights?
    #Looking forward to your response!
    #Best regards,
    #John