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

    # 基于模型的协同过滤
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

print("\n唯一用户数量:", unique_users)
print("唯一电影数量:", unique_movies)

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

