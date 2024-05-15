import streamlit as st
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


# Setup NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


page = st.sidebar.selectbox("Choose a page",["Our Data","News Analysis", "Sentiment Analysis", "Economic Impact"])

if page == "Our Data":
    data = {
        "Source": ["Reddit_Comments", "Facebook_Comments", "Instagram_Comments", "News_Articles"],
        "Count": [304, 815, 800, 204]
    }
    df = pd.DataFrame(data)

    # Streamlit app layout
    def main():
        st.title("Comments and Articles Count")

        # Display DataFrame
        
        

        # Plot countplot
        st.subheader("Countplot of our Data")
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='Source', y='Count', ax=ax)
        ax.set_ylabel('Count')
        ax.set_xlabel('Source')
        ax.set_title('Count of Comments and Articles by Source')
        ax.tick_params(axis='x', rotation=45)  # Rotate x-axis labels for better readability
        st.pyplot(fig)

    if __name__ == "__main__":
        main()

if page == "News Analysis":

    def main():
        st.title("Pandoro-Gate: a data-driven analysis")

    if __name__ == "__main__":
        main()

    def main():
        st.write("## News Topic Modeling with LDA")
        
        # Path to the dataset
        file_path = 'NewsArticles_df'
        
        # Read the dataset
        news = pd.read_csv(file_path)

        news = news.dropna()
        
        # Process data
        process_data(news)

    def process_data(news):
        newstexts = news['Testo'].tolist()
        stop_words = set(stopwords.words('italian'))
        tokenized_comments = [preprocess_comment(comment, stop_words) for comment in newstexts]

        # Topic modeling
        dictionary, corpus, lda_model = perform_lda(tokenized_comments)
        
        # Visualization
        visualize_topics(lda_model, corpus, dictionary)

    def preprocess_comment(comment, stop_words):
        tokens = word_tokenize(comment)
        tokens = [token for token in tokens if token not in string.punctuation]
        tokens = [token for token in tokens if token.lower() not in stop_words and token.isalnum()]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def perform_lda(tokenized_comments):
        dictionary = corpora.Dictionary(tokenized_comments)
        dictionary.filter_extremes(no_below=10, no_above=0.5)
        corpus = [dictionary.doc2bow(comment) for comment in tokenized_comments]
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=3, passes=20, random_state=25)
        return dictionary, corpus, lda_model

    def visualize_topics(lda_model, corpus, dictionary):
        top_n_terms = 20
        vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary, R=top_n_terms)
        # Set width and height to larger values if necessary, and enable scrolling
        html_string = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(html_string, width=1300, height=770, scrolling=False)

    if __name__ == "__main__":
        main()




    options = {
        "Topic 1 - Commercial Theme": {"image": "topic1.png", "description": "This topic is the largest one,contains 39.4% of tokens. The topic includes terms like”sales”, ”branded” and ”Antitrust” and ”Fenice Srl”(licensor of Chiara Ferragni brands),which suggest the focus on the commercial activities ofMiss Ferragni."},
        "Topic 2 - Social Media and Public Life Theme": {"image": "topic2.png", "description": "This topic contains the 33.8% of tokens. The presence of terms such as ”Fedez”,”video”, ”follower”, ”Instagram” and ”communication”,indicates a topic centered around social media and there-fore her public life."},
        "Topic 3 - Legal Theme": {"image": "topic3.png", "description": "The smallest topic with 26.9% of tokens. Terms like ”fraud”, ”Prosecutor’s Office”, the name of the prosecutor ”Eugenio Fusco”, ”crime”, ”Codacons” and ”suspects”, suggest the related legal matters the case involved."},
    }

    # Function to display image and description
    def display_option(option):
        selected_option = options[option]
        st.image(selected_option["image"], use_column_width=True)
        st.write(selected_option["description"])

    # Streamlit app
    def main():
        st.write("## Topic's wordclouds")
        
        # Selectbox to choose an option
        selected_option = st.selectbox("Choose an option", list(options.keys()))

        # Display image and description for the selected option
        if selected_option:
            display_option(selected_option)

    if __name__ == "__main__":
        main()



if page == "Sentiment Analysis":
    def main():
        st.write('## Sentiment Distribution from Facebook Articles in Percentage')

        # Display an image from local storage
        image_path = "corriere.png"
        st.image(image_path, use_column_width=True)

        st.write("""
                        
    This bar chart represents a sentiment analysis distributed across three different articles 
                    from Italian news sources. The sentiments are categorized into three groups:

    Against Chiara Ferragni or Happy for the Fine
    Pro Chiara Ferragni
    Neutral
    The chart shows the percentage of comments that fall into each sentiment category for each article. Notably, "Il Giornale" has a higher percentage of comments that are against Chiara Ferragni or happy for the fine, whereas "Corriere della Sera" and "La Repubblica" show a majority of neutral comments, with a very small percentage showing a pro-Chiara Ferragni sentiment.
                    """)


    if __name__ == "__main__":
        main()


    def main():
        # Load CSS styles
        st.markdown("""
            <style>
                {% include 'styles.css' %}
            </style>
        """, unsafe_allow_html=True)

        # Streamlit content
        st.write('## Comments Sentiment on IG Posts')

        image_path = "sentiment per comments.png"
        st.image(image_path, use_column_width=True)

        st.write("""
        In the sentiment analysis, three categories—Positive, Neutral, and Negative—were tracked over time, as represented on the x-axis which marks specific dates of data collection. Key events, notably the 'Day of the Scandal' and the 'Fazio Interview', 
        are highlighted by two vertical dashed lines and appear to be pivotal moments with a significant impact on public sentiment.

        Here’s what we can takeaway from the graph:

        - **Prior to the scandal**: the sentiment was largely positive.
        - **On the day of the scandal**: there was a drastic shift. Positive sentiment collapsed while negative sentiment spiked sharply.
        - **Following the scandal**: the positive sentiment began to recover, suggesting a possible resilience or short-term impact of the scandal.
        - **The "Fazio Interview"**: appears to have had an impact on the sentiment distribution as well. After this interview, negative sentiment dropped,
        while positive sentiment increased again. Maybe people started to empathize with her.
        """)
    if __name__ == "__main__":
        main()


    def main():
        st.write("## Sentiment analysis on Reddit's Posts")

        image_path = "reddit.png"

        st.image(image_path, use_column_width=True)

        st.write("""
                 Results are really similar to those gathered for Facebook. Here we notice an higher number of neutral comments since they were usually more descriptive instead of giving straight judgments. However, what is interesting for our analysis is that comments against Ferragni are much higher than those in favor""")

    if __name__ == "__main__":
        main()



if page == "Economic Impact":
    def main():
        st.write("""## Economic consequences for Chiara Ferragni
                
                """)
        
        

        image_path = "number of posts.png"
        st.image(image_path, use_column_width=True)

        st.write(""" 
    The decrease in posts is evident. The substantial reduction in content output has dis-
    rupted follower habits, consequently diminishing the account’s
    success, which prompts the algorithm to penalize the reach of
    the content.""")
        
        image_path = "gain loss.png"
        st.image(image_path, use_column_width=True)

        st.write(""" 
    The financial implications of her silence strategy are noteworthy, as
    each sponsored post potentially yields €93,000 in value. Based on this, it is possible to estimate
    that by reducing her monthly sponsored post count to one in
    March 2024, from an average of 23 in September 2023 (before
    the scandal), Ferragni’s potential earnings dropped from over
    €2 million to around 90K""")


    if __name__ == "__main__":
        main()