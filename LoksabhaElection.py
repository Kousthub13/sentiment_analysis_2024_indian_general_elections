import warnings, re,math, datetime, time
start_time = time.time()
from datetime import timedelta
warnings.filterwarnings("ignore")
import math
import pandas as pd
import csv, io
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import nltk, string
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import geopandas as gpd
from geopy import geocoders
gn = geocoders.GeoNames(username = "idselection")
import shapely
from shapely.geometry import Point, LineString, Polygon

import streamlit as st   #  STREAMLIT NITIALISED
from PIL import Image 

st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache_data
def long_code_runner():
    #df = pd.read_csv(io.BytesIO(uploaded['Elections2019.csv']))
    df = pd.read_csv("Elections2019.csv")
    st.header('Twitter Sentiment Analysis 2024')
    st.title('***Election Dataset Initially***')

    st.write(df.head())  #df printed

    df.shape
    df.isnull().sum()
    #df.dtypes



    df = df[pd.notnull(df['created_at'])]
    df.dropna(subset=["full_text"], inplace = True)
    df.drop("District", axis = 1, inplace =True)
    df.drop("user_location", axis=1, inplace = True)
    df.shape

    usa_terms = ["usa", "united states", "trump", "america", "democrat","cia", "fbi", "nsa"]
    for i in range(0,46043):
        try:
            flag = True
            x = df.loc[i,"full_text"]
            #print(x)
            for j in usa_terms:
                if j in str(x).lower():
                    flag = False
                    break
            if flag==False:
                df = df.drop(i)
        except:
            pass
    df = df[df["Country"] != "Others"]

    df.shape

    df["Country"].fillna("India", inplace=True)
    df["retweet_count"].fillna(0, inplace=True)
    df["quote_count"].fillna(0, inplace=True)
    df["reply_count"].fillna(0, inplace=True)
    df["favorite_count"].fillna(0, inplace=True)
    df["State"].fillna("", inplace=True)
    df["hashtags"].fillna("", inplace=True)
    df["user_mentions_screen_name"].fillna("", inplace=True)
    df["Country"]=df["Country"].replace("0","India") 
    df["Country"]=df["Country"].replace("1","India")
    df["Country"]=df["Country"].replace("2","India")
    df["State"]=df["State"].replace("0","")
    df["State"]=df["State"].replace("1","")
    df["State"]=df["State"].replace("2","")
    df["City"].fillna("", inplace=True) 



    lastupdate, created = [], []
    for i in df["last_updated"]:
        lastupdate.append(datetime.datetime.strptime(i, "%d-%m-%Y %H:%M"))
    for i in df["created_at"]:
        created.append(datetime.datetime.strptime(i, "%d-%m-%Y %H:%M"))

    df["created_at"] = created
    df["last_updated"] = lastupdate
    df.drop(df[df['last_updated'] > datetime.datetime(2019,5,23,0,0)].index, inplace = True) 
    df.drop(df[df['created_at'] > datetime.datetime(2019,5,23,0,0)].index, inplace = True) 


    df = df.sort_values(by=['last_updated'])

    df.shape
    df.isnull().sum()
    return df
df=long_code_runner()

st.write(df.head())   # df printed

capitals = dict(zip(
    ['Delhi', 'Karnataka', 'Andhra Pradesh', 'Maharashtra',
    'Punjab', 'Uttar Pradesh', 'Gujarat', 'Pondicherry', 'Jharkhand',
    'Chandigarh', 'Madhya Pradesh', 'West Bengal', 'Assam',
    'Jammu & Kashmir', 'Bihar', 'Haryana', 'Tamil Nadu', 'Orissa',
    'Chhattisgarh', 'Rajasthan', 'Uttaranchal', 'Telangana',
    'Daman & Diu', 'Meghalaya', 'Tripura', 'Kerala', 'Goa',
    'Himachal Pradesh', 'Manipur', 'Arunachal Pradesh', 'Lakshadweep',
    'Andaman & Nicobar Islands', 'Mizoram', 'Nagaland', 'Sikkim'],
                    
    ['New Delhi','Bangalore','Hyderabad','Mumbai','Chandigarh','Lucknow', 'Gandhinagar', 'Pondicherry',"Ranchi","Chandigarh",
        "Bhopal", "Kolkata", "Dispur", "Srinagar", "Patna", "Chandigarh", "Chennai", "Bhubaneswar", "Raipur",
        "Jaipur", "Dehradun", "Hyderabad", "Daman", "Shillong", "Agartala", "Trivandrum", "Panaji", "Shimla",
        "Imphal", "Itanagar", "Kavaratti", "Port Blair", "Aizawl", "Kohima","Gangtok"]))

for i in range(0,46044):
    try:
        if df["City"][i]=='':
            df["City"][i] = capitals[df["State"][i]]
    except:
        pass
df.head()


d = {}
for i in df["City"]:
    #print("Checking for ",i)
    if i not in d:
        d[i]=set()
    for j in df["City"]:
        if(i in j and i!=''):
            d[i].add(j)
for i in d:
    if len(d[i])>1:
        print(i,d[i])

df["City"]=df["City"].replace("Navi Mumbai","Mumbai")
df["City"]=df["City"].replace("Greater Mumbai","Mumbai")
df["City"]=df["City"].replace("Bokaro Steel City","Bokaro")
df["City"]=df["City"].replace("L.B. Nagar","Hyderabad")
df["City"]=df["City"].replace("Delhi Cantt.","New Delhi")
df["City"]=df["City"].replace("New Delhi Municipal Council","New Delhi")
df["City"]=df["City"].replace("Hyderabad M.Corp", "Hyderabad")
df["City"]=df["City"].replace("Tambaur-cum-Ahmadabad", "Ahmedabad")
df["City"]=df["City"].replace("Ahmedabad Cantonment", "Ahmedabad")
df["City"]=df["City"].replace('S.A.S. Nagar (Mohali)', "Mohali")
df["City"]=df["City"].replace('Panchkula Urban Estate', "Panchkula")
df["City"]=df["City"].replace('Mirzapur-cum-Vindhyachal', "Mirzapur")
df["City"]=df["City"].replace('Jawaharnagar (Gujarat Refinery)', "Jawaharnagar")
df["City"]=df["City"].replace('Jemari  (J.K. Nagar Township)', "Jemari")
df["City"]=df["City"].replace('Jam Jodhpur', "Jodhpur")
df["City"]=df["City"].replace('Bodh Gaya', "Gaya")
df["City"]=df["City"].replace('Chamoli Gopeshwar', "Gopeshwar")
df["City"]=df["City"].replace('English Bazar', "Malda")
df["City"]=df["City"].replace('G.C.F Jabalpur', "Jabalpur")
df["City"]=df["City"].replace('Ambikapur Part-X', "Ambikapur")
df["City"]=df["City"].replace('Azhikode South', "Azikode")
df["City"]=df["City"].replace('Bhagalpur (M.Corp)', "Bhagalpur")
df["City"]=df["City"].replace('Bad', "Badgam")
df["City"]=df["City"].replace('Badgam', "Budgam")
df["City"]=df["City"].replace('Nagar', "Anand Nagar")
df["City"]=df["City"].replace('Kalambe Turf Thane', "Thane")
df["City"]=df["City"].replace('Anand', "Anand Nagar")
df["City"]=df["City"].replace('North Lakhimpur', "Lakhimpur")
df["City"]=df["City"].replace('Chhota Udaipur', "Udaipur")
df["City"]=df["City"].replace('Balod', "Baloda")
df["City"]=df["City"].replace('Chandrapur Bagicha', "Chandrapur")

set(df["City"]).intersection(set(df["State"]))

for i in range(1,46044):
    try:
        if df["City"][i] == "Bihar":
            if df["City"][i] == "" or df["City"][i] not in capitals:
                df["City"][i] = "Patna"
                df["State"][i] = "Bihar"
            else:
                df["City"][i] = capitals[df["State"][i]]
    except:
        pass

df = df.reset_index()
df.drop("index", axis=1, inplace = True)


  # df session created 


st.write('***Final Cleaned Dataset***')
st.write(df.head())



#df=st.session_state.df                                   #session variable used or we can say stored again in df


import nltk
nltk.download("vader_lexicon")
nltk.download("stopwords")
stemmer = SnowballStemmer('english')
vectorizer = TfidfVectorizer(use_idf = True, tokenizer = nltk.word_tokenize,stop_words='english', smooth_idf = True)
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
sid = SentimentIntensityAnalyzer()
stopw = set(stopwords.words('english'))

contraction = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have", "bharatiya janata party" : "bjp",
                   "inc" : "congress","@narendramodi":"modi", "pappu":"rahul gandhi","gandhi":"rahul gandhi", "@rahulgandhi":"Rahul Gandhi"}


@st.cache_data
def clean(text):
    text = text.lower()
    temp = ""
    for i in text.split():
        if i not in stopw:
            try:
                temp+=contraction[i]+' '
            except:
                temp+= i+' '
    text = temp.strip()
    text = re.sub(r'http\S+','', text)
    text = text.lower().translate(remove_punctuation_map)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.replace("bhartiya janata party",'bjp')
    text = text.replace("indian national congress", 'congress')
    text = text.replace("aam aadmi party", 'aap')
    text = text.replace("narendra modi", 'modi')
    text = text.replace("rahulgandhi", 'rahul gandhi')
    temp=''
    for i in text:
        if i.isdigit()==False:
            temp+=i
    text = temp
    text = text.split()
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)
    return text.strip()


@st.cache_data
def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1.strip(), text2.strip()])
    return ((tfidf * tfidf.T).A)[0, 1]



preprocessed = []
for i in df["full_text"]:                    # session df used 
    preprocessed.append(clean(i.strip().lower()))
#preprocessed

df.insert(4,"processed_tweet",preprocessed,True)
#df.head()

df["user_mentions_screen_name"].value_counts()[:9]

sentiment = {"Positive":[], "Negative" :[], "Neutral":[], "Compound":[]}
for i in df["processed_tweet"]:
    sentiment["Positive"].append(sid.polarity_scores(i)['pos'])
    sentiment["Negative"].append(sid.polarity_scores(i)['neg'])
    sentiment["Neutral"].append(sid.polarity_scores(i)['neu'])
    sentiment["Compound"].append(sid.polarity_scores(i)['compound'])

df.insert(9,"Neutral",sentiment['Neutral'],True)
df.insert(10,"Positive",sentiment['Positive'],True)
df.insert(11,"Negative",sentiment['Negative'],True)
df.insert(12,"Compound",sentiment['Compound'],True)

df.loc[:,"quote_count":"retweet_count"].describe()

imp = []
for i in range(0,39874):
    if df["retweet_count"][i]>=70 or sum([df["retweet_count"][i],df["quote_count"][i],df["reply_count"][i]])>=50:
        imp.append(1)
    else:
        imp.append(0)

df.insert(8,"Importance",imp,True)
df.head()

df["Importance"].value_counts()

plt.pie(df["Importance"].value_counts())
#st.pyplot(plt.show())                                         #we have to do bar plot here


df_numeric = df.select_dtypes(include=['number'])
dfcorr = df_numeric.corr(method="pearson")


#st.write(dfcorr)

m11 = dfcorr["Compound"]["quote_count"]
m22 = dfcorr["Compound"]["reply_count"]
m33 = dfcorr["Compound"]["retweet_count"]
m44 = dfcorr["Compound"]["favorite_count"]
m55 = dfcorr["Compound"]["Importance"]

#Positive or compound?

m1 = dfcorr["Positive"]["quote_count"]
m2 = dfcorr["Positive"]["reply_count"]
m3 = dfcorr["Positive"]["retweet_count"]
m4 = dfcorr["Positive"]["favorite_count"]
m5 = dfcorr["Positive"]["Importance"]

score = lambda x1,x2,x3,x4,x5: m1*x1+m2*x2+m3*x3+m4*x4+x5
#score1 = lambda x1,x2,x3,x4,x5: m11*x1+m22*x2+m33*x3+m44*x4+x5

#Random test case
score(10,20,30,10,0)

score(10,20,30,10,1) 

scores = []
for i in range(0,39874):
    qt = df["quote_count"][i]
    reply = df["reply_count"][i]
    rt = df["retweet_count"][i]
    fav = df["favorite_count"][i]
    imp = df["Importance"][i]
    scores.append(score(qt,reply,rt,fav,imp))

df["Score"] = scores

compare = ["vijay rajnath ravishankar  yudhvirsethi patra narendra modi vijayvargiyah sadhvi bjp arun manoj reddy sushma rsprasad taneja maneka udhavthackeray gautam gambhir piyush goyal nitin gadkari gadkariji rss singh vasundhraraje bjpbengal smriti kailash gautamgambhir swamy udhav sushmaswaraj sadhvi pragya vasundhra sambitpatra shivraj arunjaitley manohar parikar subramanianswamy naredra modi manojtiwari amit modiji yogi adityanath sushma swaraj nitin vivekreddy shivrajsinghchouhan vijayrupani amit shah narendramodi pragya arun jaitley thackeray sunny deol bharatiya janata party kailashvijayvargiyah adityanath yogi jaitley piyush gadkari sambit smritiirani rajnathsingh irani swaraj gautam parikar nirmala bhartiya janta party ram nirmala sitaraman modi shivrajsingh nititngadkari manohar rammadhav smriti irani yedyurappa madhav gambhir narendra rajnath singh subrmanian goyal chouhan amitshah sitaraman manoharparikar ravishankar prasad rupani rao shah ravishankarprasad narsimha vivek vijay rupani prasad bhartiya janata party giriraj chowkidar",
          "congress rahul gandhi sonia pappu manish tiwari mani shankar aiyar amrinder singh navjot sidhu pilot sachin jyotiraditya scindia ashok gehlot ajay makhan makhen chidambaram raj babbar sheila dikshit kamal nath digvijay singh sanjay kaul ashok chavan prithviraj randeep surjaewala hooda deepender kapil sibal manmohan ahmed patil natwar gaurav vallabh pawan khera taneja reddy george antony venugopal rao raman gogoi lalu prasad yadav akhilesh ravat urmila milind deora siddaramiah shivkumar dks sandeep ashok tanwar prakash jha"]
documents = df["processed_tweet"]
party = []
for i in documents:
    l = len(i.split())
    freq_bjp = 0
    freq_cong = 0
    for j in compare[0].split():
        freq_bjp+=i.count(j.strip())
    for j in compare[1].split():
        freq_cong+=i.count(j.strip())
    #print(freq_cong, freq_bjp)
    if freq_bjp>freq_cong:
        party.append("BJP")
    elif freq_cong>freq_bjp:
        party.append("Congress")
    else:
        party.append("Other")

party.count("Other")

party.count("BJP")
party.count("Congress")

plt.rcParams["figure.figsize"] = [6,6]
labels = ["BJP","Congress","Other"]
count = [party.count("BJP"),party.count("Congress"),party.count("Other")]
explode = (0.05,0.05,0.05)
patches, texts,autotexts = plt.pie(count,labels=labels, explode = explode,colors = ["orange","lightgreen","lightblue"], startangle=90, autopct='%1.1f%%',shadow = False, pctdistance=0.85)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle


plt.axis('equal')  
plt.tight_layout()
st.title('Checking Party Counts')
st.pyplot(plt.show())




df.insert(5,"Party",party)

df["Party"].value_counts()

time_sec = list(set(df["last_updated"]))
time_sec.sort()

time_day = set()
time_month = set()
for i in time_sec:
    time_day.add(i.date())
    time_month.add(i.month)
time_day = list(time_day)
time_day.sort()
time_month = list(time_month)
time_month.sort()

time_week = set()
time_fortnight = set()
result=time_day[0]
while result<= time_day[-1]:
    time_week.add(result)
    result+=timedelta(days=7)
time_week = list(time_week)
time_week.sort()

result = time_day[0]
while result<= time_day[-1]:
    time_fortnight.add(result)
    result+=timedelta(days=15)
time_fortnight = list(time_fortnight)
time_fortnight.sort()

all_hour = []
result = time_sec[0]
while result<= time_sec[-1]:
    all_hour.append((result.date(),result.hour))
    result+=timedelta(hours=1)

all_min = []
result = time_sec[0]
while result<= time_sec[-1]:
    all_min.append((result.date(),result.hour,result.minute))
    result+=timedelta(minutes=1)



@st.cache_data                       # used caching
def compute_scores(df):
    score_sec = {}
    for t in df["last_updated"]:
        score_sec[t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        t = df["last_updated"][i]
        score_sec[t][df["Party"][i]] += df["Compound"][i]

    sec_total_bjp, sec_total_cong, sec_total_other = [], [], []
    for i in score_sec:
        try:
            sec_total_bjp.append(score_sec[i]["BJP"] + sec_total_bjp[-1])
        except IndexError:
            sec_total_bjp.append(score_sec[i]["BJP"] + 0)

    for i in score_sec:
        try:
            sec_total_cong.append(score_sec[i]["Congress"] + sec_total_cong[-1])
        except IndexError:
            sec_total_cong.append(score_sec[i]["Congress"] + 0)

    for i in score_sec:
        try:
            sec_total_other.append(score_sec[i]["Other"] + sec_total_other[-1])
        except IndexError:
            sec_total_other.append(score_sec[i]["Other"] + 0)

    return sec_total_bjp, sec_total_cong, sec_total_other

st.title('Lok Sabha Analysis')
# Compute scores
sec_total_bjp, sec_total_cong, sec_total_other = compute_scores(df)

# Plotting
st.subheader("Lok Sabha Analysis")
st.line_chart({
    'BJP': sec_total_bjp,
    'Congress': sec_total_cong,
    'Other': sec_total_other
})


@st.cache_data
def compute_minute_scores(df):
    score_min = {}
    for t in all_min:
        score_min[t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        t = df["last_updated"][i]
        score_min[(t.date(), t.hour, t.minute)][df["Party"][i]] += df["Compound"][i]

    min_total_bjp, min_total_cong, min_total_other = [], [], []
    for i in score_min:
        try:
            min_total_bjp.append(score_min[i]["BJP"] + min_total_bjp[-1])
        except IndexError:
            min_total_bjp.append(score_min[i]["BJP"] + 0)

    for i in score_min:
        try:
            min_total_cong.append(score_min[i]["Congress"] + min_total_cong[-1])
        except IndexError:
            min_total_cong.append(score_min[i]["Congress"] + 0)

    for i in score_min:
        try:
            min_total_other.append(score_min[i]["Other"] + min_total_other[-1])
        except IndexError:
            min_total_other.append(score_min[i]["Other"] + 0)

    return min_total_bjp, min_total_cong, min_total_other

 # Compute minute-wise scores
min_total_bjp, min_total_cong, min_total_other = compute_minute_scores(df)

# Plotting
st.subheader("Lok Sabha Analysis (Minute-wise)")
st.line_chart({
    'BJP': min_total_bjp,
    'Congress': min_total_cong,
    'Other': min_total_other
})    


# Expensive computation function for hour-wise scores
@st.cache_resource
def compute_hour_scores(df):
    score_hour = {}
    for t in all_hour:
        score_hour[t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        t = df["last_updated"][i]
        score_hour[(t.date(), t.hour)][df["Party"][i]] += df["Compound"][i]

    hour_total_bjp, hour_total_cong, hour_total_other = [], [], []
    for i in score_hour:
        try:
            hour_total_bjp.append(score_hour[i]["BJP"] + hour_total_bjp[-1])
        except IndexError:
            hour_total_bjp.append(score_hour[i]["BJP"] + 0)

    for i in score_hour:
        try:
            hour_total_cong.append(score_hour[i]["Congress"] + hour_total_cong[-1])
        except IndexError:
            hour_total_cong.append(score_hour[i]["Congress"] + 0)

    for i in score_hour:
        try:
            hour_total_other.append(score_hour[i]["Other"] + hour_total_other[-1])
        except IndexError:
            hour_total_other.append(score_hour[i]["Other"] + 0)

    return hour_total_bjp, hour_total_cong, hour_total_other

# Compute hour-wise scores
hour_total_bjp, hour_total_cong, hour_total_other = compute_hour_scores(df)

# Plotting
st.subheader("Lok Sabha Analysis (Hour-wise)")
st.line_chart({
    'BJP': hour_total_bjp,
    'Congress': hour_total_cong,
    'Other': hour_total_other
})

# Expensive computation function for daily scores
@st.cache_resource
def compute_daily_scores(df, time_day):
    score_day = {}
    for t in time_day:
        score_day[t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        t = df["last_updated"][i]
        score_day[t.date()][df["Party"][i]] += df["Compound"][i]

    day_total_bjp, day_total_cong, day_total_other = [], [], []
    for i in score_day:
        try:
            day_total_bjp.append(score_day[i]["BJP"] + day_total_bjp[-1])
        except IndexError:
            day_total_bjp.append(score_day[i]["BJP"] + 0)

    for i in score_day:
        try:
            day_total_cong.append(score_day[i]["Congress"] + day_total_cong[-1])
        except IndexError:
            day_total_cong.append(score_day[i]["Congress"] + 0)

    for i in score_day:
        try:
            day_total_other.append(score_day[i]["Other"] + day_total_other[-1])
        except IndexError:
            day_total_other.append(score_day[i]["Other"] + 0)

    return day_total_bjp, day_total_cong, day_total_other


# Compute daily scores
time_day = set()
for i in df["last_updated"]:
    time_day.add(i.date())
time_day = list(time_day)
time_day.sort()
day_total_bjp, day_total_cong, day_total_other = compute_daily_scores(df, time_day)

# Plotting
st.subheader("Lok Sabha Analysis (Daily)")
st.line_chart({
    'BJP': day_total_bjp,
    'Congress': day_total_cong,
    'Other': day_total_other
})


#per day popularity over 100 days
# Expensive computation function for daily scores

 #Per Day Popularity over 100 days
score_day = {}
all_days = []
result=time_day[0]
while result<= time_day[-1]:
    all_days.append(result)
    result+=timedelta(days=1)
#print(all_days)
for t in all_days:
    score_day[t] = {"BJP":0,"Congress":0,"Other":0}
for i in range(0,39874):
    t = df["last_updated"][i]
    score_day[t.date()][df["Party"][i]]+=df["Score"][i]

day_total_bjp, day_total_cong, day_total_other = [],[],[]
for i in score_day:
    try:
        day_total_bjp.append(score_day[i]["BJP"]+day_total_bjp[-1])
    except IndexError:
        day_total_bjp.append(score_day[i]["BJP"]+0)

for i in score_day:
    try:
        day_total_cong.append(score_day[i]["Congress"]+day_total_cong[-1])
    except IndexError:
        day_total_cong.append(score_day[i]["Congress"]+0)

for i in score_day:
    try:
        day_total_other.append(score_day[i]["Other"]+day_total_other[-1])
    except IndexError:
        day_total_other.append(score_day[i]["Other"]+0)


plt.rcParams["figure.figsize"] = [20,15]
plt.figure(num ='Lok Sabha')
plt.plot([i  for i in range(0,98)],day_total_bjp,label = 'BJP',color = 'orange')
plt.plot([i  for i in range(0,98)],day_total_cong,label = 'Congress',color = 'green')
plt.plot([i  for i in range(0,98)],day_total_other,label = 'Other',color = 'black')
plt.style.use('ggplot')
ax = plt.gca()
plt.grid(True)
plt.legend()
plt.show()
plt.close()

#weekly popularity over 14 weeks 

# Expensive computation function for weekly scores
@st.cache_resource
def compute_weekly_scores(time_week, time_day, day_total_bjp, day_total_cong, day_total_other):
    score_week = {}
    last = time_day[0]
    ctr = 0
    while last <= time_day[-1]:
        if last in time_week:
            score_week[last] = {
                "BJP": day_total_bjp[ctr],
                "Congress": day_total_cong[ctr],
                "Other": day_total_other[ctr]
            }
        last += timedelta(days=1)
        ctr += 1

    week_total_bjp = [score_week[i]["BJP"] for i in score_week]
    week_total_cong = [score_week[i]["Congress"] for i in score_week]
    week_total_other = [score_week[i]["Other"] for i in score_week]

    return week_total_bjp, week_total_cong, week_total_other

# Extracting unique days
time_day = sorted(set(df["last_updated"].dt.date))

# Extracting unique weeks
time_week = set()
result = time_day[0]
while result <= time_day[-1]:
    time_week.add(result)
    result += timedelta(days=7)
time_week = sorted(time_week)

# Compute weekly scores
week_total_bjp, week_total_cong, week_total_other = compute_weekly_scores(
    time_week, time_day, day_total_bjp, day_total_cong, day_total_other)

# Plotting
st.subheader("Weekly Popularity over 14 Weeks")
st.line_chart({
    'BJP': week_total_bjp[:14],  # considering only first 14 weeks
    'Congress': week_total_cong[:14],
    'Other': week_total_other[:14]
})

# Fortnightly Popularity over 7 fortnights
# Expensive computation function for fortnightly scores
@st.cache_resource
def compute_fortnightly_scores(time_fortnight, time_day, day_total_bjp, day_total_cong, day_total_other):
    score_fn = {}
    last = time_day[0]
    ctr = 0
    while last <= time_day[-1]:
        if last in time_fortnight:
            score_fn[last] = {
                "BJP": day_total_bjp[ctr],
                "Congress": day_total_cong[ctr],
                "Other": day_total_other[ctr]
            }
        last += timedelta(days=1)
        ctr += 1

    fn_total_bjp = [score_fn[i]["BJP"] for i in score_fn]
    fn_total_cong = [score_fn[i]["Congress"] for i in score_fn]
    fn_total_other = [score_fn[i]["Other"] for i in score_fn]

    return fn_total_bjp, fn_total_cong, fn_total_other

# Extracting unique days
time_day = sorted(set(df["last_updated"].dt.date))

# Extracting unique fortnights
time_fortnight = set()
result = time_day[0]
while result <= time_day[-1]:
    time_fortnight.add(result)
    result += timedelta(days=15)
time_fortnight = sorted(time_fortnight)

# Compute fortnightly scores
fn_total_bjp, fn_total_cong, fn_total_other = compute_fortnightly_scores(
    time_fortnight, time_day, day_total_bjp, day_total_cong, day_total_other)

# Plotting
st.subheader("Fortnightly Popularity over 7 Fortnights")
st.line_chart({
    'BJP': fn_total_bjp[:7],  # considering only first 7 fortnights
    'Congress': fn_total_cong[:7],
    'Other': fn_total_other[:7]
})

# Monthly Popularity over 4 months
# Expensive computation function for monthly scores
@st.cache_resource
def compute_monthly_scores(time_month, df):
    score_month = {}
    for t in time_month:
        score_month[t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        month = df["last_updated"][i].month
        score_month[month][df["Party"][i]] += df["Compound"][i]

    month_total_bjp, month_total_cong, month_total_other = [], [], []
    for i in score_month:
        try:
            month_total_bjp.append(score_month[i]["BJP"] + month_total_bjp[-1])
        except IndexError:
            month_total_bjp.append(score_month[i]["BJP"] + 0)

    for i in score_month:
        try:
            month_total_cong.append(score_month[i]["Congress"] + month_total_cong[-1])
        except IndexError:
            month_total_cong.append(score_month[i]["Congress"] + 0)

    for i in score_month:
        try:
            month_total_other.append(score_month[i]["Other"] + month_total_other[-1])
        except IndexError:
            month_total_other.append(score_month[i]["Other"] + 0)

    return month_total_bjp, month_total_cong, month_total_other


# Extracting unique months
time_month = sorted(set(df["last_updated"].dt.month))

# Compute monthly scores
month_total_bjp, month_total_cong, month_total_other = compute_monthly_scores(time_month, df)

# Plotting
st.subheader("Monthly Popularity over 4 Months")
st.line_chart({
    'BJP': month_total_bjp[:4],  # considering only first 4 months
    'Congress': month_total_cong[:4],
    'Other': month_total_other[:4]
})



@st.cache_resource
def compute_state_popularity(df, all_days):
    state_day_score = {}
    for i in set(list(df["State"])):
        if list(df["State"]).count(i) > 500 and i != '':
            state_day_score[i] = dict()
            for t in all_days:
                state_day_score[i][t] = {"BJP": 0, "Congress": 0, "Other": 0}

    state_day_score["Other"] = dict()
    for t in all_days:
        state_day_score["Other"][t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        t = df["last_updated"][i].date()
        st = df["State"][i]
        pty = df["Party"][i]
        cp = df["Compound"][i]
        if st in state_day_score:
            state_day_score[st][t][pty] += cp
        else:
            state_day_score["Other"][t][pty] += cp

    state_popularity = {}
    for i in state_day_score:
        state_popularity[i] = {"BJP": [], "Congress": [], "Other": []}

    for st in state_day_score:
        for t in state_day_score[st]:
            try:
                state_popularity[st]["BJP"].append(state_day_score[st][t]["BJP"] + state_popularity[st]["BJP"][-1])
            except IndexError:
                state_popularity[st]["BJP"].append(state_day_score[st][t]["BJP"] + 0)

    for st in state_day_score:
        for t in state_day_score[st]:
            try:
                state_popularity[st]["Congress"].append(
                    state_day_score[st][t]["Congress"] + state_popularity[st]["Congress"][-1])
            except IndexError:
                state_popularity[st]["Congress"].append(state_day_score[st][t]["Congress"] + 0)

    for st in state_day_score:
        for t in state_day_score[st]:
            try:
                state_popularity[st]["Other"].append(state_day_score[st][t]["Other"] + state_popularity[st]["Other"][-1])
            except IndexError:
                state_popularity[st]["Other"].append(state_day_score[st][t]["Other"] + 0)

    del state_popularity["Other"]
    
    return state_popularity

# Extracting unique days
all_days = sorted(set(df["last_updated"].dt.date))

# Compute state-wise daily and cumulative popularity scores
state_popularity = compute_state_popularity(df, all_days)

# Plotting
plt.rcParams["figure.figsize"] = [20, 15]
plt.figure(num='Lok Sabha')
for i in state_popularity:
    plt.plot([i for i in range(len(all_days))], state_popularity[i]["BJP"], label=i)
plt.style.use('fivethirtyeight')
plt.title("Popularity of BJP over Time")
ax = plt.gca()
plt.grid(True)
plt.legend(loc='best', bbox_to_anchor=(1, 0, 0.2, 1.02))
st.pyplot()



# Expensive computation function for state-wise daily and cumulative popularity scores
@st.cache_resource
def compute_state_popularity(df, all_days):
    state_day_score = {}
    for i in set(list(df["State"])):
        if list(df["State"]).count(i) > 500 and i != '':
            state_day_score[i] = dict()
            for t in all_days:
                state_day_score[i][t] = {"BJP": 0, "Congress": 0, "Other": 0}

    state_day_score["Other"] = dict()
    for t in all_days:
        state_day_score["Other"][t] = {"BJP": 0, "Congress": 0, "Other": 0}

    for i in range(len(df)):
        t = df["last_updated"][i].date()
        st = df["State"][i]
        pty = df["Party"][i]
        cp = df["Score"][i]
        if st in state_day_score:
            state_day_score[st][t][pty] += cp
        else:
            state_day_score["Other"][t][pty] += cp

    state_popularity = {}
    for i in state_day_score:
        state_popularity[i] = {"BJP": [], "Congress": [], "Other": []}

    for st in state_day_score:
        for t in state_day_score[st]:
            try:
                state_popularity[st]["BJP"].append(state_day_score[st][t]["BJP"] + state_popularity[st]["BJP"][-1])
            except IndexError:
                state_popularity[st]["BJP"].append(state_day_score[st][t]["BJP"] + 0)
            
    for st in state_day_score:
        for t in state_day_score[st]:
            try:
                state_popularity[st]["Congress"].append(
                    state_day_score[st][t]["Congress"] + state_popularity[st]["Congress"][-1])
            except IndexError:
                state_popularity[st]["Congress"].append(state_day_score[st][t]["Congress"] + 0)

    for st in state_day_score:
        for t in state_day_score[st]:
            try:
                state_popularity[st]["Other"].append(
                    state_day_score[st][t]["Other"] + state_popularity[st]["Other"][-1])
            except IndexError:
                state_popularity[st]["Other"].append(state_day_score[st][t]["Other"] + 0)

    del state_popularity["Other"]

    return state_popularity

# Extracting unique days
all_days = sorted(set(df["last_updated"].dt.date))

# Compute state-wise daily and cumulative popularity scores
state_popularity = compute_state_popularity(df, all_days)

# Plotting BJP popularity over time for each state
plt.rcParams["figure.figsize"] = [20, 15]
plt.figure(num='Lok Sabha - BJP Popularity')
for i in state_popularity:
    plt.plot([i for i in range(len(all_days))], state_popularity[i]["BJP"], label=i)
plt.style.use('fivethirtyeight')
plt.title("BJP Popularity over Time for Each State")
plt.xlabel("Days")
plt.ylabel("Popularity")
plt.legend(loc='best', bbox_to_anchor=(1, 0, 0.2, 1.02))
st.pyplot()

# Plotting Congress popularity over time for each state
plt.figure(num='Lok Sabha - Congress Popularity')
plt.rcParams["figure.figsize"] = [20, 15]
for i in state_popularity:
    plt.plot([i for i in range(len(all_days))], state_popularity[i]["Congress"], label=i)
plt.style.use('fivethirtyeight')
plt.title("Congress Popularity over Time for Each State")
plt.xlabel("Days")
plt.ylabel("Popularity")
plt.legend(loc='best', bbox_to_anchor=(1, 0, 0.2, 1.02))
st.pyplot()

# Plotting Popularity of Other Parties over Time
plt.figure(num='Lok Sabha - Other Parties Popularity')
plt.rcParams["figure.figsize"] = [20, 15]
for i in state_popularity:
    plt.plot([i for i in range(len(all_days))], state_popularity[i]["Other"], label=i)
plt.style.use('fivethirtyeight')
plt.title("Popularity of Other Parties over Time")
plt.xlabel("Days")
plt.ylabel("Popularity")
plt.legend(loc='best', bbox_to_anchor=(1, 0, 0.2, 1.02))
st.pyplot()



# Plotting Final Popularity before Results
plt.rcParams["figure.figsize"] = [6,6]
x = matplotlib.pyplot.bar(["BJP","Congress","Other"],[max(day_total_bjp),max(day_total_cong),max(day_total_other)],width=0.3,color=["Orange","green","grey"])
plt.legend([x[0],x[1],x[2]],["BJP","Congress","Other"])
plt.title("Final Popularity Index Of Parties")
st.pyplot(plt)


states = list(set(list(df["State"])))
states.remove("")










# Most tweeted from States
state_count = dict(df["State"].value_counts())
del state_count[""]
state_count["Other"] = 0
most_state_count = dict()
most_state_count["Other"] =0

for i in state_count:
    if state_count[i]<150:
        most_state_count["Other"]+=state_count[i]
    else:
        most_state_count[i] = state_count[i]

plt.rcParams["figure.figsize"] = [30,15]
del most_state_count["Other"]
plt.style.use('fivethirtyeight')
fig = plt.bar(list(most_state_count.keys()),list(most_state_count.values()),width = 0.8)
plt.title("Most Tweeted from States")


fp1 = r"INDIA\IND_adm1.shp"
map_df1 = gpd.read_file(fp1)

count = dict(df["State"].value_counts())
states = list(count.keys())
states.remove("")

geo=[]
for i in states:
    for j in range(0,36):
        name1 = i
        name2 = map_df1["NAME_1"][j]
        if(name1==name2):
            geo.append(map_df1["geometry"][j])
            break
        elif len(set(name1.split()).intersection(set(name2.split())))>1:
            geo.append(map_df1["geometry"][j])
            break
        elif name2=="Puducherry" and name1 == "Pondicherry":
            geo.append(map_df1["geometry"][j])
            break


bjp,cong,other,party,tweets,bjppop,congpop,opop=[],[],[],[],[],[],[],[]
partycount = dict()
for i in states:
    partycount[i] = {"BJP":0,"Congress":0,"Other":0,"BJPPop":0,"CongPop":0,"OtherPop":0,
                     'narendramodi':0,'bjp4india':0,'incindia':0,'rahulgandhi':0,
                    "htmodi":0,'htbjp':0,'htloksabha':0,'htcong':0,'htrahulgandhi':0,'htchow':0}        

for i in range(0,39874):
    p = df["Party"][i]
    s = df["State"][i]
    cp = df["Compound"][i]
    temp = df["hashtags"][i]
    user = df["user_mentions_screen_name"][i]
    if s=="":
        continue
    partycount[s][p]+=1
    if p=="BJP":
        partycount[s]["BJPPop"]+=cp
    if p=="Congress":
        partycount[s]["CongPop"]+=cp
    if p=="Other":
        partycount[s]["OtherPop"]+=cp
    for hst in temp.split(','):
        if hst =="BJP":
            partycount[s]['htbjp']+=1
        if hst =="Modi":
            partycount[s]['htmodi']+=1
        if hst =="LokSabhaElections2019":
            partycount[s]['htloksabha']+=1
        if hst =="Congress":
            partycount[s]['htcong']+=1
        if hst =="RahulGandhi":
            partycount[s]['htrahulgandhi']+=1
        if hst =="MainBhiChowkidar":
            partycount[s]['htchow']+=1
    for x in user.split(','):
        if x=="BJP4India":
            partycount[s]['bjp4india']+=1
        if x=="narendramodi":
            partycount[s]['narendramodi']+=1
        if x=="INCIndia":
            partycount[s]['incindia']+=1
        if x=="RahulGandhi":
            partycount[s]['rahulgandhi']+=1 


htbjp,htmodi,htls,htcong,htrg,htchow = [],[],[],[],[],[]
hdbjp,hdmodi,hdcong,hdrg=[],[],[],[]

for i in partycount:
    bjp.append(partycount[i]["BJP"])
    cong.append(partycount[i]["Congress"])
    other.append(partycount[i]["Other"])
    tweets.append(partycount[i]["BJP"]+partycount[i]["Congress"]+partycount[i]["Other"])
    bjppop.append(partycount[i]["BJPPop"])
    congpop.append(partycount[i]["CongPop"])
    opop.append(partycount[i]["OtherPop"])
    
    htbjp.append(partycount[i]['htbjp'])
    htmodi.append(partycount[i]['htmodi'])
    htls.append(partycount[i]['htloksabha'])
    htcong.append(partycount[i]['htcong'])
    htrg.append(partycount[i]['htrahulgandhi'])
    htchow.append(partycount[i]['htchow'])

    hdbjp.append(partycount[i]['bjp4india'])
    hdmodi.append(partycount[i]['narendramodi'])
    hdcong.append(partycount[i]['incindia'])
    hdrg.append(partycount[i]['rahulgandhi'])
    
    if partycount[i]["BJPPop"]>partycount[i]["CongPop"] and partycount[i]["BJPPop"]>partycount[i]["OtherPop"]:
        party.append("BJP")
    elif partycount[i]["BJPPop"]<partycount[i]["CongPop"] and partycount[i]["OtherPop"]<partycount[i]["CongPop"]:
        party.append("Congress")
    else:
        party.append("Other")


testdf = pd.DataFrame({"State":states,"Tweets":tweets,"Geo":geo,"Party":party,"BJP":bjp,"Congress":cong,"Other":other,
                      "BJPPop":bjppop,"Congpop":congpop,"OtherPop":opop,
                      "#BJP":htbjp,"#Modi":htmodi,"#LokSabhaElections2019":htls,"#Congress":htcong,"#RahulGandhi":htrg,"#MainBhiChowkidar":htchow,
                      "@BJP4India":hdbjp,"@narendramodi":hdmodi,"@INCIndia":hdcong,"@RahulGandhi":hdrg})
testdf.head()

gdf = gpd.GeoDataFrame(testdf, geometry="Geo")

st.subheader("States tweeting about Other Parties")

image1 = Image.open('image19/first/mostTweetFromState.png')
st.image(image1, caption='Most TweetS From State')

image2 = Image.open('image19/first/States tweeting about Congress.png')
st.image(image2, caption='States tweeting about Congress')


image3 = Image.open('image19/first/stateTweetingAboutBJP.png')
st.image(image3, caption='States tweeting about BJP')


image4 = Image.open('image19/first/States tweeting about Other Parties.png')
st.image(image4, caption='States tweeting about Other Parties')

st.subheader("Popularity Trends")

image21 = Image.open('image19/second/Popularity Trend Across States.png')
st.image(image21, caption='Popularity Trend Across States')


image22 = Image.open('image19/second/BJP Popularity.png')
st.image(image22, caption='Popularity Of BJP')


image23 = Image.open('image19/second/Congress Popularity.png')
st.image(image23, caption='Congress Popularity')


image24 = Image.open('image19/second/Other Party Popularity.png')
st.image(image24, caption='Other Party Popularity')


image25 = Image.open('image19/second/Most Tweeted from Cities BARPLOT.png')
st.image(image25, caption='Most Tweeted from Cities')


st.subheader("Most Popular Hastag")

image31 = Image.open('image19/third/a#BJP.png')
st.image(image31, caption='#BJP')

image32 = Image.open('image19/third/b#Modi.png')
st.image(image32, caption='#Modi')
image33 = Image.open('image19/third/c#Congress.png')
st.image(image33, caption='#Congress')

image34 = Image.open('image19/third/d#RahulGandhi.png')
st.image(image34, caption='#RahulGandhi')

image35 = Image.open('image19/third/e#MainBhiChowkidar.png')
st.image(image35, caption='#MainBhiChowkidar')

image36 = Image.open('image19/third/f#LokSabhaElections2024.png')
st.image(image36, caption='#LokSabhaElections2024')

image37 = Image.open('image19/third/Most Popular Hashtags BARPLOT.png')
st.image(image37, caption='Most Popular Hashtags')


st.subheader("Most Tagged Handles")


image41 = Image.open('image19/four/aMost Tagged Handles.png')
st.image(image41, caption='Most Tagged Handles')


image42 = Image.open('image19/four/bPeople tagging @Modi.png')
st.image(image42, caption='People tagging @Modi')


image43 = Image.open('image19/four/cPeople tagging @BJP4India.png')
st.image(image43, caption='People tagging @BJP4India')


image44 = Image.open('image19/four/dPeople tagging @Congress.png')
st.image(image44, caption='People tagging @Congress')


image45 = Image.open('image19/four/ePeople tagging @Congress.png')
st.image(image45, caption='People tagging @Congress')


image46 = Image.open('image19/four/fPeople tagging @Congress.png')
st.image(image46, caption='People tagging @Congress')


st.subheader("Frequency of Tweets per Day")


image51 = Image.open('image19/five/aFrequency of Tweets per Day.png')
st.image(image51, caption='Frequency of Tweets per Day')

image52 = Image.open('image19/five/bFrequency of Tagging.png')
st.image(image52, caption='Frequency of Tagging')

image53 = Image.open('image19/five/cPopularity of Handles over Time.png')
st.image(image53, caption='Popularity of Handles over Time')

image54 = Image.open('image19/five/dWinning Probability.png')
st.image(image54, caption='Winning Probability')

image55 = Image.open('image19/five/eChances of Winning.png')
st.image(image55, caption='Chances of Winning')








