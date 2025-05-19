import pandas as pd
import re
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

# ----------------------------
# Clean Tweets
# ----------------------------
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet)
    tweet = re.sub(r'\@\w+|\#','', tweet)
    tweet = re.sub(r"[^a-zA-Z0-9\s]", '', tweet)
    return tweet

# ----------------------------
# Analyze Sentiment
# ----------------------------
def analyze_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# ----------------------------
# Load Sample Dataset
# ----------------------------
def load_sample_tweets():
    df = pd.read_csv("Sentiment.csv")  # Make sure this file exists in the same folder
    df = df[['text', 'label']]
    df.columns = ['Tweet', 'Sentiment']
    df['Cleaned_Tweet'] = df['Tweet'].apply(clean_tweet)
    df['Sentiment'] = df['Cleaned_Tweet'].apply(analyze_sentiment)
    return df

# ----------------------------
# Streamlit App UI
# ----------------------------
st.set_page_config(page_title="Dual Mode Sentiment Analyzer", layout="centered")
st.title("🧠 Dual Mode Sentiment Analyzer")
st.markdown("Analyze custom input or a batch of tweets from a dataset.")

# ----------------------------
# Mode Selection
# ----------------------------
mode = st.radio("Choose Mode", ["📥 Type Text", "📂 Analyze Sample Tweets"])

# ----------------------------
# Mode 1: Real-time Sentiment Input
# ----------------------------
if mode == "📥 Type Text":
    user_input = st.text_area("💬 Enter text, tweet, or hashtag...", height=100, placeholder="e.g., I love Streamlit!")
    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            sentiment = analyze_sentiment(user_input)
            polarity = TextBlob(user_input).sentiment.polarity
            st.markdown(f"**Sentiment:** {sentiment}")
            st.markdown(f"**Polarity Score:** `{polarity}`")

# ----------------------------
# Mode 2: Analyze Dataset
# ----------------------------
else:
    if st.button("Analyze Sample Dataset"):
        with st.spinner("Analyzing tweets..."):
            df = load_sample_tweets()
            st.success("Analysis Complete!")

            # 📊 Sentiment Distribution
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            st.bar_chart(sentiment_counts.set_index('Sentiment'))

            # ☁️ Word Clouds for Positive and Negative
            st.subheader("☁️ Word Clouds")

            # Positive Word Cloud
            st.markdown("**Positive Tweets Word Cloud**")
            pos_text = " ".join(tweet for tweet in df[df["Sentiment"] == "Positive"]["Cleaned_Tweet"])
            if pos_text:
                pos_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(pos_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(pos_wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.write("No positive tweets found.")

            # Negative Word Cloud
            st.markdown("**Negative Tweets Word Cloud**")
            neg_text = " ".join(tweet for tweet in df[df["Sentiment"] == "Negative"]["Cleaned_Tweet"])
            if neg_text:
                neg_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(neg_wordcloud, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.write("No negative tweets found.")

            # 📌 Sample Table
            st.subheader("📌 Sample Tweets")
            st.dataframe(df[["Tweet", "Sentiment"]].head(10))
