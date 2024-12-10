import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL setup for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Define the chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Initialize CSV for storing conversation history if not exists
history_file = 'chat_log.csv'
if not os.path.exists(history_file):
    with open(history_file, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

# Main function for the Streamlit app
def main():
    st.title("Intents of Chatbot using NLP")

    # Sidebar menu options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Home Menu: User input and chatbot interaction
    if choice == "Home":
        st.write("Welcome to the chatbot. Please type a message and press Enter to start the conversation.")

        global counter
        counter = 0  # Reset counter on app start
        user_input = st.text_input("You:", key=f"user_input_{counter}")

        if user_input:
            response = chatbot(user_input)
            st.text_area("Chatbot:", value=response, height=120, max_chars=None, key=f"chatbot_response_{counter}")

            # Log the user input, response, and timestamp in CSV file
            timestamp = datetime.datetime.now().strftime(f"%Y-%m-%d %H:%M:%S")
            with open(history_file, 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Exit the conversation if response is goodbye
            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu: Displaying previous conversations
    elif choice == "Conversation History":
        st.header("Conversation History")
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.text(f"User: {row[0]}")
                    st.text(f"Chatbot: {row[1]}")
                    st.text(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.info("No conversation history found.")

    elif choice == "About":
        st.write("This chatbot uses NLP techniques with Logistic Regression to identify the user's intent and provide appropriate responses. The model is trained with various intents and patterns.")

if __name__ == '__main__':
    main()
