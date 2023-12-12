# Final-project-recommended-system
# Introduction 
A Python recommendation system is a mechanism that generates personalized recommendations for users based on their preferences, behavior, or similarities to other users. It analyzes data such as user ratings, purchase history, or browsing patterns to provide relevant suggestions for items, products, movies, music, or any content of interest. There are various approaches for building recommendation systems, such as collaborative filtering, content-based filtering, and hybrid methods. Libraries like scikit-learn, Surprise, and TensorFlow can be used to implement recommendation systems in Python.

Our project is a movel recommendation system with data collection and processing from a fake data library,including details such as title, director, actors, genre, rating, etc., as well as users' historical viewing records and ratings. This data is then cleaned, removing duplicates and handling missing values. Next, appropriate recommendation algorithms, such as content-based, collaborative filtering, and deep learning, are selected and implemented. Python's machine learning and deep learning libraries are utilized to train and optimize the recommendation models based on users' viewing records, ratings, and movie characteristics. Additionally, user profiling and interest modeling are developed to gain insights into users' interests, preferences, and behavior patterns, utilizing machine learning libraries in Python. A user-friendly interactive interface is designed, enabling users to easily search, browse, and watch movies while receiving real-time personalized recommendations. This interface can be developed using Python's web framework or GUI library. Finally, the recommendation system is evaluated and optimized by analyzing performance metrics like accuracy, recall rate, and coverage rate. User feedback and behavior data are collected to further enhance the recommendation algorithm and model, improving the overall efficacy of the system. 

# Methodology
## Data collection
Download and read the fake movie information data file

## Data processing
1. Clean the original data
2. Filter the data
3. Data partitioning

## Data visualization
Import visualization library matplotlib to create:
1. Number of films released in each country
2. Which types of movies are most popular
3. The proportion of different types of movies
4. The film length distribution

## Function
1. Feature extraction and representation
2. Construct a feature vector space
3. Similarity calculation
4. User interest modeling
5. Recommended movie
6. Create a website by GUI
7. Iterative optimization

# Interface interaction description: 
1. RECOMMENDATIONS: the movie name and rating of this movie from movielens is shown to recommend for user, user can click on like or dislike to make a feedback and click on refresh, a new movie would show up
2. Number of films released in each country : shows a figure based on data from movielens
3. Which types of movies are most popular: shows a keywords cloud about types of movie include: include Action, Comedy, Drama, Romance, Thriller, Horror, Science Fiction, Fantasy, Animation, Documentary, base on what movielens give us
4. The proportion of different types of movies: shows a figure based on data from movielens
5. The film length distribution:shows a figure based on data from movielens

# Recommended system
Base on the data catch from movielens below
1. Rating: the higher the rating the movie more possible to be recommended
2. Feedback: base on the movies people select likes, the system recommend similar one to them based on direct name, type, film length, country, release time, movielens rating, film length, director, actors
3. Types: which types has most movies, which would be more possible to be recommended
4. Lengths: which types has most popular length, which would be more possible to be recommended

# Training
Provide users with feedback options, including "like", "dislike", "refresh", etc., and adjust the recommendation results according to user feedback; Incremental learning of the algorithm or real-time updating of the user interest model can improve the recommendation effect.

# Group information
YIXIN HU
yhu134@jh.edu

YUANKUN LI
yli604@jh.edu
