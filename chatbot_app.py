import nltk
import streamlit as st
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab') # <-- This is the crucial one for the current error

# Load and preprocess the Grimm's Fairy Tales text
with open('grimm_tales.txt', 'r', encoding='utf-8') as f:
    data = f.read()

# Split the text into fairy tale sections
tale_sections = {}
current_tale = None
lines = data.split('\n')
for line in lines:
    line = line.strip()
    if line.isupper() and line not in ['CONTENTS:', '*** START OF THE PROJECT GUTENBERG EBOOK GRIMMS\' FAIRY TALES ***', '*** END OF THE PROJECT GUTENBERG EBOOK GRIMMS\' FAIRY TALES ***']:
        current_tale = line
        tale_sections[current_tale] = []
    elif current_tale and line:
        tale_sections[current_tale].append(line)

# Join sentences for each tale and create a list of sentences
sentences = []
tale_mapping = {}
for tale, lines in tale_sections.items():
    tale_text = ' '.join(lines).replace('\n', ' ')
    tale_sentences = sent_tokenize(tale_text)
    sentences.extend(tale_sentences)
    for sentence in tale_sentences:
        tale_mapping[sentence] = tale

# Custom stop words for fairy tales
custom_stopwords = set(stopwords.words('english')).union({
    'king', 'princess', 'prince', 'queen', 'said', 'went', 'came', 'upon',
    'fairy', 'tale', 'grimm', 'brothers', 'one', 'day', 'night'
})

# Preprocess function
def preprocess(text):
    words = word_tokenize(text)
    # Keep only nouns and verbs for better semantic matching
    tagged = nltk.pos_tag(words)
    words = [word.lower() for word, pos in tagged if pos.startswith(('NN', 'VB'))]
    words = [word for word in words if word not in string.punctuation and word not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

# Preprocess all sentences
corpus = [preprocess(sentence) for sentence in sentences]
corpus_joined = [' '.join(words) for words in corpus]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus_joined)

# Historical context
historical_context = """
The Brothers Grimm, Jacob (1785-1863) and Wilhelm (1786-1859), were German scholars who collected over 200 folk tales, published in 1812 and 1814 as 'Nursery and Household Tales.' Their goal was to preserve German cultural heritage, but their stories, like 'The Golden Bird' and 'Hans in Luck,' became beloved worldwide. Translated by Edgar Taylor in 1823, the tales became popular for children, despite their original scholarly intent.
"""

# Moral explanations for some tales
morals = {
    'THE GOLDEN BIRD': "Listen to wise advice and avoid greed to achieve your goals.",
    'HANS IN LUCK': "Value what you have and find happiness in simplicity, not material wealth.",
    'JORINDA AND JORINDEL': "Perseverance and love can overcome even the most difficult challenges.",
    'THE TRAVELLING MUSICIANS': "Teamwork and creativity can lead to unexpected success.",
    'OLD SULTAN': "Loyalty and cleverness are rewarded, even in old age.",
    'THE STRAW, THE COAL, AND THE BEAN': "Unity and cooperation can prevent misfortune.",
    'BRIAR ROSE': "Patience and destiny can lead to a happy resolution.",
    'CAT-SKIN': "Inner beauty and resilience shine through despite hardships.",
    'SNOW-WHITE AND ROSE-RED': "Kindness and bravery are rewarded with love and freedom."
}

# Function to find the most relevant sentence or tale
def get_most_relevant_sentence(query):
    query = query.lower()
    # Check for specific tale request
    for tale in tale_sections:
        if tale.lower() in query:
            return ' '.join(tale_sections[tale][:50]) + " ... (full tale available)"
    # Check for moral request
    if 'moral' in query:
        for tale in morals:
            if tale.lower() in query:
                return f"The moral of {tale} is: {morals[tale]}"
    # Check for historical context
    if 'history' in query or 'brothers grimm' in query:
        return historical_context
    # Preprocess query
    query_processed = ' '.join(preprocess(query))
    query_vec = vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    max_similarity_idx = similarities.argmax()
    if similarities[max_similarity_idx] > 0.1:  # Threshold for relevance
        return sentences[max_similarity_idx]
    return None

# Chatbot function
def chatbot(question):
    response = get_most_relevant_sentence(question)
    if response:
        tale = tale_mapping.get(response, "General")
        return f"From '{tale}': {response}"
    return "I'm sorry, I couldn't find a relevant response. Try asking about a specific fairy tale, its moral, or the history of the Brothers Grimm!"

# Streamlit app
def main():
    st.title("Grimm's Fairy Tales Chatbot")
    st.write("Welcome to the magical world of Grimm's Fairy Tales! Ask me to tell a story, explain a moral, or share the history of the Brothers Grimm.")
    
    # Query type buttons
    query_type = st.radio("What would you like to do?", 
                         ("Ask about a specific tale", "Learn a moral", "Explore historical context"))
    
    # User input
    question = st.text_input("Your question:")
    
    # Submit button
    if st.button("Submit"):
        if question:
            response = chatbot(question)
            st.write(f"**Chatbot**: {response}")
        else:
            st.write("Please enter a question!")
    
    # Example questions
    with st.expander("Example Questions"):
        st.write("- Tell me about The Golden Bird")
        st.write("- What's the moral of Hans in Luck?")
        st.write("- What's the history of Grimm's Fairy Tales?")

if __name__ == "__main__":
    main()