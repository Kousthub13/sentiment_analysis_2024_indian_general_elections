import streamlit as st
import pandas as pd
from PIL import Image

st.title("2024 Indian General Election")

df = pd.read_csv('2024/IndianElection24TwitterData.csv')
st.write(df.head())


image1 = Image.open('2024/imgs/barplotCountModi.png')
st.image(image1, caption='Count Of Emotion Of Tweets About Modi')

image2=Image.open('2024/imgs/barplotCountRahul.png')
st.image(image2, caption='Count Of Emotion Of Tweets About Rahul Gandhi')

image3=Image.open('2024/imgs/sentimentScoreModi.png')
st.image(image3, caption='Sentiment scores of tweets about Modi')

image4=Image.open('2024/imgs/sentimentScoreRahul.png')
st.image(image4, caption='Sentiment scores of tweets about Rahul Gandhi')


image5=Image.open('2024/imgs/darkPolarityModi.png')
st.image(image5, caption='Polarity Changes Of Tweets Mentioning Modi')

image6=Image.open('2024/imgs/darkPolarityRahul.png')
st.image(image6, caption='Polarity Changes Of Tweets Mentioning Rahul Gandhi')


image7=Image.open('2024/imgs/lightPolarityModi.png')
st.image(image7, caption='Polarity Changes Of Tweets Mentioning Modi')

image8=Image.open('2024/imgs/lightPolarityRahul.png')
st.image(image8, caption='Polarity Changes Of Tweets Mentioning Rahul Gandhi')
