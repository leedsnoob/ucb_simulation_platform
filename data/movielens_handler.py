import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

class MovieLensDataHandler:
    """
    Handles loading and preprocessing of the MovieLens 1M dataset to simulate a contextual bandit environment.
    """
    def __init__(self, data_path: str = 'ml-1m', reward_threshold: float = 4.0):
        """
        Initializes the data handler by loading and preprocessing the data.

        Args:
            data_path (str): The path to the directory containing MovieLens data files.
            reward_threshold (float): The rating threshold to binarize the reward (rating >= threshold is 1, else 0).
        """
        self.data_path = Path(data_path)
        self.reward_threshold = reward_threshold
        self.cache_path = self.data_path / 'preprocessed_data.pkl'
        
        self._load_and_preprocess_data()
        self.current_index = 0
        self.context_dimension = self.data.iloc[0]['context_vector'].shape[0]
        print(f"Context vector dimension: {self.context_dimension}")


    def _load_and_preprocess_data(self):
        """
        Loads and preprocesses data. If a cached version exists, it's loaded directly.
        Otherwise, it processes the raw data and creates a cache for future use.
        """
        if self.cache_path.exists():
            print("Loading preprocessed data from cache...")
            cache_data = pd.read_pickle(self.cache_path)
            self.data = cache_data['data']
            self.all_genres = cache_data['all_genres']
            print("Cached data loaded successfully.")
            return

        print("No cache found. Processing raw data...")
        # Define column names as they are not present in the .dat files
        r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        u_cols = ['user_id', 'gender', 'age', 'occupation', 'zipcode']
        m_cols = ['movie_id', 'title', 'genres']

        # Load data files
        ratings = pd.read_csv(self.data_path / 'ratings.dat', sep='::', names=r_cols, engine='python', encoding='latin-1')
        users = pd.read_csv(self.data_path / 'users.dat', sep='::', names=u_cols, engine='python', encoding='latin-1')
        movies = pd.read_csv(self.data_path / 'movies.dat', sep='::', names=m_cols, engine='python', encoding='latin-1')

        # --- User Feature Engineering ---
        # One-hot encode categorical user features
        users_features = pd.get_dummies(users, columns=['gender', 'age', 'occupation'], dtype=float)
        # We drop zipcode as it has too high cardinality for this simple model
        users_features = users_features.drop(columns=['zipcode'])
        
        # --- Movie Feature Engineering ---
        # Create multi-hot encoding for genres
        self.all_genres = sorted(list(set([g for genre_list in movies['genres'].str.split('|') for g in genre_list])))
        genre_map = {genre: i for i, genre in enumerate(self.all_genres)}
        
        genre_vectors = []
        for genre_list in movies['genres'].str.split('|'):
            vec = np.zeros(len(self.all_genres))
            for genre in genre_list:
                if genre in genre_map:
                    vec[genre_map[genre]] = 1
            genre_vectors.append(vec)
        movies['genres_vector'] = genre_vectors
        movies_features = movies.drop(columns=['title', 'genres'])

        # --- Merge DataFrames ---
        # Merge ratings with user features
        merged_data = pd.merge(ratings, users_features, on='user_id')
        # Merge with movie features
        merged_data = pd.merge(merged_data, movies_features, on='movie_id')

        # --- Final Processing ---
        # Sort data by timestamp to create a sequential event stream
        merged_data.sort_values(by='timestamp', inplace=True)
        
        # Binarize the reward
        merged_data['reward'] = (merged_data['rating'] >= self.reward_threshold).astype(int)
        
        # Create the final context vector by combining user and movie features
        user_feature_cols = [col for col in users_features.columns if col != 'user_id']
        
        def create_context_vector(row):
            user_vec = row[user_feature_cols].values.astype(float)
            movie_vec = row['genres_vector']
            return np.concatenate([user_vec, movie_vec])

        # Use tqdm.pandas() for progress bar on .apply()
        tqdm.pandas(desc="Creating Context Vectors")
        merged_data['context_vector'] = merged_data.progress_apply(create_context_vector, axis=1)

        # RENAME movie_id to chosen_arm for consistency across handlers
        merged_data.rename(columns={'movie_id': 'chosen_arm'}, inplace=True)

        # Select final columns and reset index
        self.data = merged_data[['user_id', 'chosen_arm', 'reward', 'context_vector']].reset_index(drop=True)
        
        # Save the processed data and metadata to a cache object
        cache_to_save = {
            'data': self.data,
            'all_genres': self.all_genres
        }
        pd.to_pickle(cache_to_save, self.cache_path)
        print(f"Processed data saved to cache at {self.cache_path}")
        
        print(f"Loaded and preprocessed {len(self.data)} interactions from MovieLens 1M.")


    def __iter__(self):
        """
        Allows the handler to be used as an iterator.
        """
        self.current_index = 0
        return self

    def __next__(self):
        """
        Returns the next interaction in the stream.
        """
        if self.current_index < len(self.data):
            interaction = self.data.iloc[self.current_index]
            self.current_index += 1
            
            context = {
                'user_id': interaction['user_id'],
                'vector': interaction['context_vector']
            }
            arm = interaction['chosen_arm']
            reward = interaction['reward']
            
            # In this simple replay, available_arms is just the one chosen arm
            return {'context': context, 'available_arms': [arm], 'chosen_arm': arm, 'reward': reward}
        else:
            raise StopIteration

    def reset(self):
        """
        Resets the iterator to the beginning of the data stream.
        """
        self.current_index = 0

    def __len__(self):
        """
        Returns the total number of interactions in the dataset.
        """
        return len(self.data)
