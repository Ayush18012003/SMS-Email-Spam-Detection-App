import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [word for word in text if word.isalnum()]
    y = [word for word in y if word not in stopwords.words('english') and word not in string.punctuation]
    y = [ps.stem(word) for word in y]

    return " ".join(y)

# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit UI
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

# App Title
st.markdown("<h1 style='text-align: center; color: #90CAF9;'>📧 Email/SMS Spam Classifier</h1>", unsafe_allow_html=True)

# Input Section
st.markdown("<h3 style='text-align: center; color: #7986CB;'>Enter your message below to check if it's Spam or Not!</h3>", unsafe_allow_html=True)
input_sms = st.text_area("", placeholder="Type your message here...", height=150)

# Predict Button
if st.button('🚀 Predict'):
    if input_sms.strip():
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        # Vectorize the input
        vector_input = tfidf.transform([transformed_sms])
        # Make prediction
        result = model.predict(vector_input)[0]

        # Display Result
        if result == 1:
            st.markdown(
                "<div style='text-align: center; padding: 20px; background-color: #EF9A9A; color: white; border-radius: 10px;'>"
                "<h2>🚨 Spam Message Detected!</h2>"
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align: center; padding: 20px; background-color: #81C784; color: white; border-radius: 10px;'>"
                "<h2>✅ Not Spam</h2>"
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Please enter a message before clicking Predict.")

# Footer
st.markdown(
    """
    <hr>
    <div style='text-align: center; color: grey;'>
        <p>Created with ❤️ using Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("**Built by Ayush Gupta**")
st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/ayush-gupta-352287261/) or [GitHub](https://github.com/Ayush18012003).")