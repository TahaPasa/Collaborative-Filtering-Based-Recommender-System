# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:54:01 2022

@author: TahaAlpKocyigit
"""

#%%
import numpy as np
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.reader import Reader
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



#%%

# note: its not movielens dataset!
films = pd.read_csv('C:/Users/takoc/Desktop/Huawei_task/ml-latest/ratings.csv', names=['userId', 'movieId', 'Rating','timestamp'], header=None)

films.drop('timestamp', axis=1, inplace=True)
films.drop(index=films.index[0], axis=0, inplace=True)

films['Rating'] = pd.to_numeric(films['Rating'])
films['movieId'] = pd.to_numeric(films['movieId'])
films['userId'] = pd.to_numeric(films['userId'])

films = films[:1000000]  #1.000.000 would be enough right? :D


films_groupby_users_Ratings = films.groupby('userId')['Rating']
films_groupby_users_Ratings = pd.DataFrame(films_groupby_users_Ratings.count())



#%%





df = films


user_ids = df["userId"].unique().tolist()
movie_ids = df["movieId"].unique().tolist()

min_rating = min(df["Rating"])
max_rating = max(df["Rating"])


num_users = len(user_ids)
num_movies = len(movie_ids)

movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["movieId"] = df["movieId"].map(movie2movie_encoded)

user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
df["userId"] = df["userId"].map(user2user_encoded)


#%% NORMALIZING

x = df[["userId", "movieId"]].values



y = preprocessing.normalize([df["Rating"]])        
y = y.T


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)




#%%

EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()

        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="lecun_uniform",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="lecun_uniform",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)
        
        
        self.dense1 = tf.keras.layers.Dense(30, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(20, activation=tf.nn.relu)
        self.dense3 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense4 = tf.keras.layers.Dense(5, activation=tf.nn.relu)
        self.dense5 = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropla = tf.keras.layers.Dropout(0.5)
        self.flatto = tf.keras.layers.Flatten()

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])            #creating biases
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])          #creating biases        
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        
    
        ###############################################################
       
        y = self.flatto(x)
        
        y = self.dense1(y)
        y = self.dense2(y)
        
        y = self.dropla(y)
        
        y = self.dense3(y)
        y = self.dense4(y)
        
        
        output = self.dense5(y)
        return output
        
        
        


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
)

#%% 

history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=1024,
    epochs=5,
    verbose=1,
    validation_data=(x_test, y_test),
)
#%%   OBSERVE


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()




#%%
"""
np.save('./history1.npy',history.history)
 
history1=np.load('./history1.npy',allow_pickle='TRUE').item()
"""

#%%

model.save('./')


#%%


os.chdir('C:/Users/takoc/Desktop/Huawei_task/saved_model.pb')

model = keras.models.load_model('C:/Users/takoc/Desktop/Huawei_task/saved_model.pb')

#%%


movie_df = pd.read_csv('C:/Users/takoc/Desktop/Huawei_task/ml-latest/movies.csv')

# Let us get a user and see the top recommendations.
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)]["movieId"]
movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))  
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched))
ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices]



#%%
print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)
top_movies_user = (movies_watched_by_user.sort_values(by="Rating", ascending=False).head(5).movieId.values)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("Top 5 movie recommendations")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)][:5]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)
