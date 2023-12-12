import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics.pairwise import pairwise_distances
import tkinter as tk
from tkinter import ttk
from tkinter import Label
from tkinter.scrolledtext import ScrolledText
from difflib import get_close_matches
import ttkbootstrap as ttk
from ttkbootstrap import Style
from fuzzywuzzy import process


def create_rating_matrix(data, num_users, num_movies):
    """
    Create a user-movie rating matrix from the given data.

    Parameters:
        data (pd.DataFrame): DataFrame containing user-movie ratings.
        num_users (int): Number of unique users.
        num_movies (int): Number of unique movies.

    Returns:
        np.ndarray: User-movie rating matrix.
    """
    ratings_matrix = np.zeros((num_users, num_movies))
    for line in data.itertuples():
        ratings_matrix[line[1] - 1, line[2] - 1] = line[3]
    return ratings_matrix


def evaluate_collaborative_filtering(train_data_matrix, test_data_matrix, similarity_metric='cosine', model_k=20):
    """
    Evaluate collaborative filtering models using user-based, item-based, and model-based approaches.

    Parameters:
        train_data_matrix (np.ndarray): User-movie rating matrix for training.
        test_data_matrix (np.ndarray): User-movie rating matrix for testing.
        similarity_metric (str): Similarity metric for user and item similarity (default: 'cosine').
        model_k (int): Number of dimensions for matrix factorization (default: 20).

    Returns:
        tuple: Tuple containing User-Based RMSE, Item-Based RMSE, and Model-Based RMSE.
    """
    # 用户-产品协同过滤
    user_similarity = pairwise_distances(train_data_matrix, metric=similarity_metric)
    item_similarity = pairwise_distances(train_data_matrix.T, metric=similarity_metric)

    user_prediction = predict_user_based(train_data_matrix, user_similarity)
    item_prediction = predict_item_based(train_data_matrix, item_similarity)

    user_based_rmse = rmse(user_prediction, test_data_matrix)
    item_based_rmse = rmse(item_prediction, test_data_matrix)

    # Model-based collaborative filtering
    u, s, vt = svds(train_data_matrix, k=model_k)
    s_diag_matrix = np.diag(s)
    model_based_prediction = np.dot(np.dot(u, s_diag_matrix), vt)
    model_based_rmse = rmse(model_based_prediction, test_data_matrix)

    print("User-Based CF RMSE:", user_based_rmse)
    print("Item-Based CF RMSE:", item_based_rmse)
    print("Model-Based CF RMSE:", model_based_rmse)
    return user_based_rmse, item_based_rmse, model_based_rmse


def predict_user_based(ratings, similarity):
    """
    Predict user-based recommendations.

    Parameters:
        ratings (np.ndarray): User-movie rating matrix.
        similarity (np.ndarray): User similarity matrix.

    Returns:
        np.ndarray: Predicted user-based ratings.
    """
    mean_user_rating = ratings.mean(axis=1)
    ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
    denominator = np.abs(similarity).sum(axis=1)
    denominator[denominator == 0] = 1e-8

    pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([denominator]).T
    return pred


def predict_item_based(ratings, similarity):
    """
    Predict item-based recommendations.

    Parameters:
        ratings (np.ndarray): User-movie rating matrix.
        similarity (np.ndarray): Item similarity matrix.

    Returns:
        np.ndarray: Predicted item-based ratings.
    """
    pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


def rmse(prediction, ground_truth):
    """
    Calculate Root Mean Squared Error (RMSE) between predicted and actual ratings.

    Parameters:
        prediction (np.ndarray): Predicted ratings.
        ground_truth (np.ndarray): Actual ratings.

    Returns:
        float: RMSE value.
    """
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))


# Read the U.ATA file
data = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

# Remove duplicate data
data = data.drop_duplicates()

# Handle missing values, if any
print(data.isnull().sum())  # Check for missing values
data = data.dropna()  # Deletes rows that contain missing values

# Handle rating range errors, for example, assuming that the rating should be between 1 and 5
data['rating'] = np.clip(data['rating'], 1, 5)

# Count the number of unique users and movies
unique_users = data['userId'].nunique()
unique_movies = data['movieId'].nunique()

print("\nNumber of unique users:", unique_users)
print("Number of unique films:", unique_movies)

# Split the data set into a training set and a test set
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

# Create user-product scoring matrices (training and test sets)
train_data_matrix = create_rating_matrix(train_data, unique_users, unique_movies)
test_data_matrix = create_rating_matrix(test_data, unique_users, unique_movies)

# Evaluate collaborative filtering models
evaluate_collaborative_filtering(train_data_matrix, test_data_matrix)

# Read data
movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1', header=None, names=['movieId', 'title'],
                     usecols=[0, 1])


def get_movie_title(movie_id):
    """
    Get the movie title based on the given movie ID.

    Parameters:
        movie_id (int): Movie ID.

    Returns:
        str: Movie title.
    """
    # Gets the movie title based on the movie ID
    return movies[movies['movieId'] == movie_id]['title'].values[0]


def get_movie_id(movie_title):
    """
    Get the movie ID based on the given movie title.

    Parameters:
        movie_title (str): Title of the movie.

    Returns:
        int: Movie ID if found, otherwise -1.
    """
    movie = movies[movies['title'] == movie_title]
    if not movie.empty:
        return movie['movieId'].values[0]
    else:
        print(f"Movie '{movie_title}' not found.")
        return -1


def recommend_movies_user_based(user_id, user_prediction, num_recommendations=5):
    """
    Recommend movies to a user using model-based collaborative filtering.

    Parameters:
        user_id (int): User ID.
        model_based_prediction (np.ndarray): Predicted model-based ratings.
        num_recommendations (int): Number of recommendations to provide (default: 5).
    """
    # Get the user's unrated movie forecast rating
    user_ratings = user_prediction[user_id - 1]
    unrated_movies = np.where(train_data_matrix[user_id - 1] == 0)[0]

    # Sort the unrated movies and get the top N recommended movies
    recommendations = sorted(list(zip(unrated_movies, user_ratings[unrated_movies])), key=lambda x: x[1], reverse=True)[
                      :num_recommendations]

    if not recommendations:
        print(f"\nUser {user_id} has rated all movies.")
    else:
        print(f"\nTop {num_recommendations} Movie Recommendations for User {user_id}:")
        for movie_id, rating in recommendations:
            print(f"{get_movie_title(movie_id + 1)} (MovieID: {movie_id + 1}, Predicted Rating: {rating:.2f})")


def recommend_movies_model_based(user_id, model_based_prediction, num_recommendations=5):
    """
    Recommend movies to a user using model-based collaborative filtering.

    Parameters:
        user_id (int): User ID.
        model_based_prediction (np.ndarray): Predicted model-based ratings.
        num_recommendations (int): Number of recommendations to provide (default: 5).
    """
    # Get the user's unrated movie forecast rating
    user_ratings = model_based_prediction[user_id - 1]
    unrated_movies = np.where(user_ratings == 0)[0]

    # Sort the unrated movies and get the top N recommended movies
    recommendations = sorted(list(zip(unrated_movies, user_ratings[unrated_movies])), key=lambda x: x[1], reverse=True)[
                      :num_recommendations]

    print(f"\nTop {num_recommendations} Movie Recommendations for User {user_id}:")
    for movie_id, rating in recommendations:
        print(f"{get_movie_title(movie_id + 1)} (MovieID: {movie_id + 1}, Predicted Rating: {rating:.2f})")


# Model testing: Recommendation using user-product collaborative filtering
user_id = 1  # Change to the user ID you want to generate recommendations for
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
user_prediction = predict_user_based(train_data_matrix, user_similarity)
recommend_movies_user_based(user_id, user_prediction)

# Model testing: Recommendation using user-product collaborative filtering
user_id = 1  # Change to the user ID you want to generate recommendations for
u, s, vt = svds(train_data_matrix, k=20)
s_diag_matrix = np.diag(s)
model_based_prediction = np.dot(np.dot(u, s_diag_matrix), vt)
recommend_movies_model_based(user_id, model_based_prediction)


def recommend_movies_for_movie(movie_title, model_based_prediction, num_recommendations=10):
    """
    Recommend movies based on the similarity to a given movie.

    Parameters:
        movie_title (str): Title of the movie.
        model_based_prediction (np.ndarray): Predicted model-based ratings.
        num_recommendations (int): Number of recommendations to provide (default: 10).

    Returns:
        list: List of movie recommendations.
    """
    movie_id = get_movie_id(movie_title)

    if movie_id != -1:
        movie_ratings = model_based_prediction[:, movie_id - 1]

        # Sort the unrated movies and get the top N recommended movies
        recommendations = sorted(list(enumerate(movie_ratings)), key=lambda x: x[1], reverse=True)[:num_recommendations]

        print(f"\nTop {num_recommendations} Movie Recommendations for Movie '{movie_title}':")
        for movie_id, rating in recommendations:
            print(f"{get_movie_title(movie_id + 1)} (MovieID: {movie_id + 1}, Predicted Rating: {rating:.2f})")

        return recommendations
    else:
        print(f"Movie '{movie_title}' not found.")
        return []


class MovieRecommendationApp:
    def __init__(self, root, unique_users, unique_movies, user_based_rmse, item_based_rmse, model_based_rmse):
        self.root = root
        self.root.title("Movie Recommendation")

        # Create ttkbootstrap style
        style = Style(theme='lumen')
        style.configure('.', font=('Helvetica', 12))

        # Frame for user input and button
        input_frame = ttk.Frame(root, padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        self.label = ttk.Label(input_frame, text="Enter the name of a movie for recommendations:")
        self.label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        # Entry widget with autocomplete
        self.movies = pd.read_csv('data/ml-100k/u.item', sep='|', encoding='latin-1', header=None,
                                  names=['movieId', 'title'],
                                  usecols=[0, 1])
        self.movie_titles = self.movies['title'].tolist()
        self.entry_var = tk.StringVar()
        self.entry = ttk.Combobox(input_frame, textvariable=self.entry_var, values=self.movie_titles)
        self.entry.set('')
        self.entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.entry.bind("<KeyRelease>", self.autocomplete)

        self.button = ttk.Button(input_frame, text="Get Recommendations", command=self.show_recommendations)
        self.button.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Frame for Result text box
        frame_result = ttk.Frame(root, padding=10)
        frame_result.grid(row=1, column=0, sticky="w")

        # Result text box (ScrolledText for scrollable text)
        self.result_text = ScrolledText(frame_result, wrap=tk.WORD, height=12, width=88)
        self.result_text.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Frame for displaying information
        frame_info = ttk.Frame(root, padding=10)
        frame_info.grid(row=2, column=0, sticky="sw")

        # Labels displaying information
        ttk.Label(frame_info, text=f"Unique Users: {unique_users}").grid(row=0, column=0, sticky="w")
        ttk.Label(frame_info, text=f"Unique Movies: {unique_movies}").grid(row=1, column=0, sticky="w")
        ttk.Label(frame_info, text=f"User-Based RMSE: {user_based_rmse:.4f}").grid(row=2, column=0, sticky="w")
        ttk.Label(frame_info, text=f"Item-Based RMSE: {item_based_rmse:.4f}").grid(row=3, column=0, sticky="w")
        ttk.Label(frame_info, text=f"Model-Based RMSE: {model_based_rmse:.4f}").grid(row=4, column=0, sticky="w")

    def show_recommendations(self):
        input_movie_title = self.entry_var.get().strip()
        recommendations = recommend_movies_for_movie(input_movie_title, model_based_prediction)

        if recommendations:
            result_text = "\n".join(
                [f"{get_movie_title(movie_id + 1)} (MovieID: {movie_id + 1}, Predicted Rating: {rating:.2f})"
                 for movie_id, rating in recommendations])
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Top Movie Recommendations for '{input_movie_title}':\n{result_text}")
        else:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"No recommendations found for movie '{input_movie_title}'")

    def autocomplete(self, event):
        # Perform fuzzy search and update combobox values
        input_text = self.entry_var.get()
        if input_text:
            matches = process.extract(input_text, self.movie_titles, limit=10)
            fuzzy_matches = [match[0] for match in matches if match[1] >= 70]  # Adjust the threshold as needed
            self.entry['values'] = fuzzy_matches
        else:
            self.entry['values'] = self.movie_titles


user_based_rmse, item_based_rmse, model_based_rmse = evaluate_collaborative_filtering(train_data_matrix,
                                                                                      test_data_matrix)

if __name__ == "__main__":
    root = tk.Tk()
    app = MovieRecommendationApp(root, unique_users, unique_movies, user_based_rmse, item_based_rmse, model_based_rmse)
    root.mainloop()