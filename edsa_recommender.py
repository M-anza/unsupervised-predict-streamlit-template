
"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


# Data Loading
title_list = load_movie_titles('movies.csv')
rating_m = pd.read_csv('ratings.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Predict Overview","Exploratory Data Analysis","Search for a Movie","New Movie Release","Feature engineering","Modeling"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Unsupervised geniuses')
        st.write('## Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            
            if st.button("Recommend"):
                st.text("test test test test test test")
                try:
                    st.text("test2 test2 test2 test2 test2 test2")
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.text("test3 test3 test3 test3 test3 test3")
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                    
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Predict Overview":
        
        st.title("EA Movie Recommendation Predict 2023-2024")
        st.image('resources/imgs/movie.jpg',width = 600)
        
        st.markdown('In todays technology-driven era, the sheer volume of available content often overwhelms consumers. They are left grappling with decision fatigue, unsure of which movies to watch amidst the vast array of options. The lack of personalized recommendations further exacerbates this issue, leading to missed opportunities for both viewers and content providers.')
        st.title("Solution overview")           
        st.markdown('The objective of this project is to build a movie recommendation system that provides personalized movie recommendations to users based on their preferences and historical interactions.')       
        st.markdown("New User Onboarding: Prompt new users to provide initial preferences to build a starting user profile, balancing the registration process length with the amount of data needed for accurate recommendations.\
                     Hybrid Recommenders: Mitigate the limitations of individual models by combining them. The hybrid approach enhances recommendation quality and robustness.")
        st.markdown('Root Mean Square Error (RMSE): Used to measure the accuracy of the recommendation models. RMSE quantifies the deviation between predicted and actual ratings, providing insights into model performance.\
                    The Hybrid Recommender System achieves a certain RMSE for a subset of movies. Due to memory limitations, the analysis is performed on a subset of the dataset, ensuring computational feasibility.', unsafe_allow_html=True)        
        st.markdown('The movie recommendation system leverages a hybrid approach to overcome the limitations of individual models and provides personalized recommendations to users', unsafe_allow_html=True) 
        
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    if page_selection  =="Search for a Movie":
        
        st.title("Search for Movies")
        st.markdown('If you decide not to use the recommender systems you can use this page to filter movies based on the rating of the movie , the year in which the movie was released and the genre of the movies. After you change the filter you will be left with movies that are specific to that filter used. Then when you scroll down you will see the movie name and the link to a youtube trailer of that movie. When you click the link ,you will see a page on youtube for that specific movie and you can watch the trailer and see if you like it. This is an alternative method to you if you are not satisfied with the recommender engine . Enjoy! ', unsafe_allow_html=True)
        # Movies
        df = pd.read_csv('movies.csv')

        def explode(df, lst_cols, fill_value='', preserve_index=False):
            import numpy as np
             # make sure `lst_cols` is list-alike
            if (lst_cols is not None
                    and len(lst_cols) > 0
                    and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
                lst_cols = [lst_cols]
            # all columns except `lst_cols`
            idx_cols = df.columns.difference(lst_cols)
            # calculate lengths of lists
            lens = df[lst_cols[0]].str.len()
            # preserve original index values    
            idx = np.repeat(df.index.values, lens)
            # create "exploded" DF
            res = (pd.DataFrame({
                        col:np.repeat(df[col].values, lens)
                        for col in idx_cols},
                        index=idx)
                    .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
            # append those rows that have empty lists
            if (lens == 0).any():
                # at least one list in cells is empty
                res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                            .fillna(fill_value))
            # revert the original index order
            res = res.sort_index()   
            # reset index if requested
            if not preserve_index:        
                res = res.reset_index(drop=True)
            return res  

        movie_data = pd.merge(rating_m, df, on='movieId')
        movie_data['year'] = movie_data.title.str.extract('(\(\d\d\d\d\))',expand=False)
        #Removing the parentheses
        movie_data['year'] = movie_data.year.str.extract('(\d\d\d\d)',expand=False)

        movie_data.genres = movie_data.genres.str.split('|')
        movie_rating = st.sidebar.number_input("Pick a rating ",0.5,5.0, step=0.5)

        movie_data = explode(movie_data, ['genres'])
        movie_title = movie_data['genres'].unique()
        title = st.selectbox('Genre', movie_title)
        movie_data['year'].dropna(inplace = True)
        movie_data = movie_data.drop(['movieId','timestamp','userId'], axis = 1)
        year_of_movie_release = movie_data['year'].sort_values(ascending=False).unique()
        release_year = st.selectbox('Year', year_of_movie_release)

        movie = movie_data[(movie_data.rating == movie_rating)&(movie_data.genres == title)&(movie_data.year == release_year)]
        df = movie.drop_duplicates(subset = ["title"])
        if len(df) !=0:
            st.write(df)
        if len(df) ==0:
            st.write('We have no movies for that rating!')        
        def youtube_link(title):
    
            """This function takes in the title of a movie and returns a Search query link to youtube
    
            INPUT: ('The Lttle Mermaid')
            -----------
    
            OUTPUT: https://www.youtube.com/results?search_query=The+little+Mermaid&page=1
            ----------
            """
            title = title.replace(' ','+')
            base = "https://www.youtube.com/results?search_query="
            q = title
            page = "&page=1"
            URL = base + q + page
            return URL            
        if len(df) !=0:           
            for _, row in df.iterrows():
                st.write(row['title'])
                st.write(youtube_link(title = row['title']))

     # Building out the EDA page
    if page_selection == "Exploratory Data Analysis":
        
        st.title("Insights on how people rate movies")   
           
        if st.checkbox('Highest movies per year'):
            st.image('resources/imgs/eda1.webp',width = 800)
           
        if st.checkbox('Top 10 movies'):
            st.image('resources/imgs/eda2.webp',width = 800)
            
        if st.checkbox('User engagement over time'):   
           st.image('resources/imgs/eda3.webp',width = 800)
        
        
    if page_selection == "Feature engineering":
        st.image('resources/imgs/data.jpg',width = 600)
        st.title("Memory reduction function")
        st.markdown('To optimize memory usage')
        st.title("Converting rating column to float")
        st.markdown('We convert the ratings column to float data type')
        st.title("Dimensional reduction using PCA")
        st.markdown('Dimensionality reduction using Principal Component Analysis (PCA)')

    if page_selection == "Modeling":
        st.title("Modeling")
        st.markdown("Content_based flitering")
        st.image('resources/imgs/model1.jpg',width = 600)
        st.markdown("Content-based filtering is like having a smart friend who knows your likes and dislikes")

        st.markdown("Collaborative flitering")
        st.image('resources/imgs/model2.jpg',width = 600)
        st.markdown("Collaborative filtering is a method that suggests movies you might like based on what similar users have enjoyed")
        
        st.markdown("Hybrid")
        st.image('resources/imgs/model3.jpg',width = 600)
        st.markdown("Hybrid filtering is like a mix-and-match of two methods: collaborative filtering and content-based filtering")
        

    if page_selection == "New Movie Release":
        st.title("New Movie Release ")
        st.markdown('You will find all new movie releases here . Enjoy!')
        st.subheader("The Old Guard")
        st.video('https://www.youtube.com/watch?v=aK-X2d0lJ_s')
        st.markdown("Directed by :	Gina Prince-Bythewood")
        st.markdown("Starring : Charlize Theron, KiKi Layne, Marwan Kenzari, Luca Marinelli ,Harry Melling ")
        st.markdown("Plot : Led by a warrior named Andy (Charlize Theron), a covert group of tight-knit mercenaries with a mysterious inability to die have fought to protect the mortal world for centuries. But when the team is recruited to take on an emergency mission and their extraordinary abilities are suddenly exposed, it's up to Andy and Nile (Kiki Layne), the newest soldier to join their ranks, to help the group eliminate the threat of those who seek to replicate and monetize their power by any means necessary.")
        st.markdown("")
        st.subheader("Bad Boys For Life (2020)")
        st.video('https://www.youtube.com/watch?v=jKCj3XuPG8M')
        st.markdown('Directed by :	Adil & Bilall')
        st.markdown('Starring : Will Smith, Martin Lawrence, Paola Núñez, Vanessa Hudgens, Alexander Ludwig, Charles Melton, Jacob Scipio')
        st.markdown('Plot : Marcus and Mike have to confront new issues (career changes and midlife crises), as they join the newly created elite team AMMO of the Miami police department to take down the ruthless Armando Armas, the vicious leader of a Miami drug cartel')
        st.subheader("Bloodshot (2020)")
        st.video('https://www.youtube.com/watch?v=vOUVVDWdXbo')
        st.markdown('Directed by : David S. F. Wilson ')
        st.markdown('Starring : Vin Diesel, Eiza González ,Sam Heughan, Toby Kebbell')
        st.markdown("Plot : After he and his wife are murdered, marine Ray Garrison is resurrected by a team of scientists. Enhanced with nanotechnology, he becomes a superhuman, biotech killing machine-'Bloodshot'. As Ray first trains with fellow super-soldiers, he cannot recall anything from his former life. But when his memories flood back and he remembers the man that killed both him and his wife, he breaks out of the facility to get revenge, only to discover that there's more to the conspiracy than he thought. ")
        st.markdown("")
        st.markdown("You can watch new movies in the following sites, Enjoy!")
        
        st.markdown('<p><a href="https://www.netflix.com/za/">Netflix</a></p>', unsafe_allow_html=True)	
       
        st.markdown('<p><a href="https://www.showmax.com/eng/browse?type=movie">Showmax</a></p>', unsafe_allow_html=True)
        
        st.markdown('<p><a href="https://preview.disneyplus.com/za">Disney plus</a></p>', unsafe_allow_html=True)	
        st.markdown('<p>This application is sponsored by <a href="https://explore-datascience.net/">Explore Data Science Academy</a> </p>', unsafe_allow_html=True)
       
if __name__ == '__main__':
    main()
