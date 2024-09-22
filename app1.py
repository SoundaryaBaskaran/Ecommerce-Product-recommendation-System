import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# Recommendations function
def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        st.write(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating','Description']]
    return recommended_items_details

def display_recommendations(recommendations):
    if recommendations.empty:
        st.write("No recommendations available.")
        return

    st.write("### Recommended Products")
    for index, row in recommendations.iterrows():
        if pd.isna(row['ImageURL']) or not row['ImageURL'].strip():
            image_url = 'https://via.placeholder.com/150?text=Image+Not+Available'
        else:
            image_url = row['ImageURL']

        st.image(image_url, caption=row['Name'], width=150)
        st.write(f"**Name:** {row['Name']}")
        st.write(f"**Brand:** {row['Brand']}")
        st.write(f"**Review Count:** {row['ReviewCount']}")
        st.write(f"**Reviews:** {row['ReviewCount']}")
        st.write(f"**Description**: {row['Description']}")

# Streamlit app
st.title("E-Commerce Product Recommendation System")

# Company details section on the sidebar
with st.sidebar:
    st.image("static/img/Beauty (4).png", width=220) 
    st.markdown("<h2 style='text-align: center;'>About Us</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-style: italic;'>At Beauty Products, we believe that beauty is more than skin deep. Our mission is to empower you to feel confident and radiant every day. We are passionate about creating high-quality beauty products that enhance your natural beauty and bring out your best self.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Contact Us</h2>", unsafe_allow_html=True)

    st.write("**Company Name:** Beauty Products")
    st.write("**Address:** 123 SpicyUp, Bangalore")  
    st.write("**Country:** India")
    st.write("**Phone Number:** +1234567890")  
    st.write("**Email:** beauty@gmail.com")

# Video display
video_file = open('static/Beauty products.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)

# Layout containers
col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Find Recommendations")
    product_name = st.text_input("Enter Product Name")
    num_recommendations = st.slider("Number of Recommendations", min_value=1, max_value=20, value=5)

    if st.button("🔍 Get Recommendations"):
        if product_name:
            recommendations = content_based_recommendations(train_data, product_name, top_n=num_recommendations)
            display_recommendations(recommendations)
        else:
            st.write("Please enter a product name.")

with col2:
    # Container to align images to the right
    with st.container():
        st.write("### Beauty Products")
        st.image('static/img/img_1.jpg', caption='Lipsticks', use_column_width=True)
        st.image('static/img/img_2.jpg', caption='Compact Powder', use_column_width=True)
        st.image('static/img/img_3.jpg', caption='Sandals', use_column_width=True)
        st.image('static/img/img_4.jpg', caption='Hand bags', use_column_width=True)
        st.image('static/img/img_5.jpg', caption='Perfumes', use_column_width=True)
        st.image('static/img/img_6.jpg', caption='Body Lotion', use_column_width=True)
        st.image('static/img/img_7.jpg', caption='Oils', use_column_width=True)
        st.image('static/img/img_8.jpg', caption='Makeup Sets', use_column_width=True)
        st.image('static/img/img_9.jpg', caption='Shampoo', use_column_width=True)

