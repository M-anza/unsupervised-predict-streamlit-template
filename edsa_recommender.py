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
#imdb = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/imdb_data.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Predict Overview","Exploratory Data Analysis","Search for a Movie",]

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
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
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
        #st.image('resources/imgs/search.webp',width = 600)
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
           
        if st.checkbox('Show Rating graph'):
            rating_m.groupby('rating')['userId'].count().plot(kind = 'bar', color = 'g',figsize = (8,7))
            plt.xticks(rotation=85, fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.xlabel('Ratings (scale: 0.5 - 5.0)', fontsize=16)
            plt.ylabel('No. of Ratings', fontsize=16)
            plt.title('Distribution of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot()
            st.markdown("This is a bar graph showing the rating of movie by people who have watched them.")
            st.markdown("The number of ratings is the total number of rating for each scale from 0.5 upto 5.0 rated by people who watched the movies.")
        if st.checkbox('Show Pie chart for ratings'):
            # Calculate and categorise ratings proportions
            a = len(rating_m.loc[rating_m['rating']== 0.5]) / len(rating_m)
            b = len(rating_m.loc[rating_m['rating']==1.0]) / len(rating_m)
            c = len(rating_m.loc[rating_m['rating']==1.5]) / len(rating_m)
            d = len(rating_m.loc[rating_m['rating']==2.0]) / len(rating_m)
            low_ratings= a+b+c+d
            e = len(rating_m.loc[rating_m['rating']==2.5]) / len(rating_m)
            f = len(rating_m.loc[rating_m['rating']== 3.0]) / len(rating_m)
            g = len(rating_m.loc[rating_m['rating']==3.5]) / len(rating_m)
            medium_ratings= e+f+g
            h = len(rating_m.loc[rating_m['rating']==4.0]) / len(rating_m)
            i = len(rating_m.loc[rating_m['rating']==4.5]) / len(rating_m)
            j = len(rating_m.loc[rating_m['rating']==5.0]) / len(rating_m)
            high_ratings= h+i+j 
            # To view proportions of ratings categories, it is best practice to use pie charts
            # Where the slices will be ordered and plotted clockwise:
            labels = 'Low Ratings', 'Medium Ratings', 'High Ratings'
            sizes = [low_ratings, medium_ratings,  high_ratings]
            explode = (0, 0, 0.1)  # Only "explore" the 3rd slice (i.e. 'Anti')

            # Create pie chart with the above labels and calculated class proportions as inputs
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=270)#,textprops={'rotation': 65}
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Categorised Proportions of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot()
            st.markdown("This is a pie chart showing the rating of movies by people who have watched them.")
            st.markdown("Low Ratings (scale: 0.5 - 2.0)")
            st.markdown("Medium Ratings (scale: 2.5 - 3.5)")
            st.markdown("High Ratings (scale: 4.0 - 5.0)")

        if st.checkbox('Show WordCloud of directors'):   
            imdb["title_cast"] = imdb["title_cast"].astype('str')
            imdb["director"] = imdb["director"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('|',' '))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace(' ',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('|',' '))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace(' ',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('Seefullsummary',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('nan',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('nan',''))
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('nan',''))  

            directors = ' '.join([text for text in imdb["director"]])

            # Word cloud for the overall data checking out which words do people use more often
            wordcloud = WordCloud(width=1000, height=800).generate(directors)

            #ploting the word cloud
            plt.figure(figsize=(16,12))
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot() 
            st.markdown("This is a wordcloud of the directors of movies in this Application.")
            st.markdown("This wordcloud shows the most popular directors on the movies.")
        if st.checkbox('Show WordCloud of Actors/Actresses'):
            imdb["title_cast"] = imdb["title_cast"].astype('str')
            imdb["director"] = imdb["director"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('|',' '))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace(' ',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('|',' '))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace(' ',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('Seefullsummary',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('nan',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('nan',''))
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('nan',''))   

            title_cast= ' '.join([text for text in imdb["title_cast"]])

            # Word cloud for the overall data checking out which words do people use more often
            wordcloud = WordCloud(width=1000, height=800).generate(title_cast)

            #ploting the word cloud
            plt.figure(figsize=(16,12))
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot()  
            st.markdown("This is a wordcloud for Actors/Actresses on the movies on this Application.")
            st.markdown("This wordcloud shows the most popular Actors/Actresses on the movies.")
        if st.checkbox("Show wordcloud of different genres"):    
            movies = pd.read_csv('resources/data/movies.csv')
            #here we  make census of the genres:
            genre_labels = set()
            for s in movies['genres'].str.split('|').values:
                genre_labels = genre_labels.union(set(s))  

            #counting how many times each of genres occur:
            keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
            #Finally, the result is shown as a wordcloud:
            words = dict()
            trunc_occurences = keyword_occurences[0:50]
            for s in trunc_occurences:
                words[s[0]] = s[1]
            tone = 100 # define the color of the words
            f, ax = plt.subplots(figsize=(14, 6))
            wordcloud = WordCloud(width=1000,height=800, background_color='white', 
                                max_words=1628,relative_scaling=0.7,
                                color_func = random_color_func,
                                normalize_plurals=False)
            wordcloud.generate_from_frequencies(words)
            plt.figure(figsize=(16,12))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot()
            st.markdown("This is a wordcloud for all the different genres in this Application.")
        #st.image('resources/imgs/eda.png',width = 600) 
        #st.image('resources/imgs/genre.jpg',width = 600)              
    
if __name__ == '__main__':
    main()
