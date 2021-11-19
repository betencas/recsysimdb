
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
#from fuzzywuzzy import fuzz
#from fuzzywuzzy import process

movies = pd.read_csv('c:/py/homemovies.csv', sep = ",")

X = movies[['genres_Action',
       'genres_Adventure', 'genres_Animation', 'genres_Biography',
       'genres_Comedy', 'genres_Crime', 'genres_Documentary', 'genres_Drama',
       'genres_Family', 'genres_Fantasy', 'genres_Film-Noir',
       'genres_Game-Show', 'genres_History', 'genres_Horror', 'genres_Music',
       'genres_Musical', 'genres_Mystery', 'genres_News', 'genres_Reality-TV',
       'genres_Romance', 'genres_Sci-Fi', 'genres_Short', 'genres_Sport',
       'genres_Talk-Show', 'genres_Thriller', 'genres_War', 'genres_Western',
       'director_nb', 'startYear', 'averageRating']]

distanceKNN = NearestNeighbors(n_neighbors=5).fit(X)

st.title('Lights out and grab some popcorns! Welcome to our Movie Recommendation System!')

user_input = st.text_input("Tell us a movie so we can show you some recommendations: ")

input_results = movies[movies['primaryTitle'].str.contains(user_input, na = False)]['primaryTitle']

options = st.multiselect('Please choose one film from below:', input_results)

clean_option = str(options)
cleaner_option = clean_option[2:-2]

if st.button('Submit'):
    movie = movies.loc[movies.primaryTitle == cleaner_option]
    movie_index = movie.index.astype(int)
    favorite = list(X.iloc[movie_index[0]])

    prediction = distanceKNN.kneighbors([favorite])
    results = prediction[1].tolist()
    results = results[0]
    results_n = movies.iloc[results]
    results_a = results_n[["primaryTitle","director"]]
    results_a.rename(columns={"primaryTitle": "Title:", "director" : "Directed by:"}, inplace = True)

    st.header("Thank you very much! Please check below what we have for you:")
    st.write(results_a[1:5])

    title= str(results_n.iloc[1]['title_id'])
    url = f"https://www.imdb.com/title/{title}/?ref_=nv_sr_srsg_0"
    st.write(f"Check [here]({url}) for the details for our first recommendation :)")
