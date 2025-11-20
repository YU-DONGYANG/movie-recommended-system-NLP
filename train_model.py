import os
import pickle
import pandas as pd
import ast
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('stopwords')

# Object for porterStemmer
ps = PorterStemmer()


def get_genres(obj):
    """Extract genre names from JSON string"""
    lista = ast.literal_eval(obj)
    l1 = []
    for i in lista:
        l1.append(i['name'])
    return l1


def get_cast(obj):
    """Extract top 10 cast members from JSON string"""
    a = ast.literal_eval(obj)
    l_ = []
    len_ = len(a)
    for i in range(0, 10):
        if i < len_:
            l_.append(a[i]['name'])
    return l_


def get_crew(obj):
    """Extract director names from crew JSON string"""
    l1 = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l1.append(i['name'])
            break
    return l1


def stemming_stopwords(li):
    """Apply stemming and remove stopwords from text"""
    ans = []

    for i in li:
        ans.append(ps.stem(i))

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    filtered_sentence = []
    for w in ans:
        w = w.lower()
        if w not in stop_words:
            filtered_sentence.append(w)

    str_ = ''
    for i in filtered_sentence:
        if len(i) > 2:
            str_ = str_ + i + ' '

    # Removing Punctuations
    punc = string.punctuation
    str_.translate(str_.maketrans('', '', punc))
    return str_


def read_csv_to_df(data_dir='data'):
    """Read and preprocess movie data"""
    print("Reading CSV files...")

    # Reading both the csv files
    credit_ = pd.read_csv(os.path.join(data_dir, 'tmdb_5000_credits.csv'))
    movies = pd.read_csv(os.path.join(data_dir, 'tmdb_5000_movies.csv'))

    # Merging the dataframes
    movies = movies.merge(credit_, on='title')

    movies2 = movies.copy()
    movies2.drop(['homepage', 'tagline'], axis=1, inplace=True)
    movies2 = movies2[['movie_id', 'title', 'budget', 'overview', 'popularity', 'release_date', 'revenue', 'runtime',
                       'spoken_languages', 'status', 'vote_average', 'vote_count']]

    # Extracting important and relevant features
    movies = movies[
        ['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'production_companies', 'release_date']]
    movies.dropna(inplace=True)

    # Applying functions to convert from list to only items
    movies['genres'] = movies['genres'].apply(get_genres)
    movies['keywords'] = movies['keywords'].apply(get_genres)
    movies['top_cast'] = movies['cast'].apply(get_cast)
    movies['director'] = movies['crew'].apply(get_crew)
    movies['prduction_comp'] = movies['production_companies'].apply(get_genres)

    # Removing spaces from between the lines
    movies['overview'] = movies['overview'].apply(lambda x: x.split())
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcast'] = movies['top_cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tcrew'] = movies['director'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies['tprduction_comp'] = movies['prduction_comp'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Creating a tags where we have all the words together for analysis
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['tcast'] + movies['tcrew']

    # Creating new dataframe for the analysis part only
    new_df = movies[['movie_id', 'title', 'tags', 'genres', 'keywords', 'tcast', 'tcrew', 'tprduction_comp']]

    new_df['genres'] = new_df['genres'].apply(lambda x: " ".join(x))
    new_df['tcast'] = new_df['tcast'].apply(lambda x: " ".join(x))
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: " ".join(x))

    new_df['tcast'] = new_df['tcast'].apply(lambda x: x.lower())
    new_df['genres'] = new_df['genres'].apply(lambda x: x.lower())
    new_df['tprduction_comp'] = new_df['tprduction_comp'].apply(lambda x: x.lower())

    # Applying stemming on tags and keywords
    new_df['tags'] = new_df['tags'].apply(stemming_stopwords)
    new_df['keywords'] = new_df['keywords'].apply(stemming_stopwords)

    return movies, new_df, movies2


def vectorise(new_df, col_name):
    """Vectorize text data using CountVectorizer"""
    print(f"Vectorizing {col_name}...")
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vec_tags = cv.fit_transform(new_df[col_name]).toarray()
    sim_bt = cosine_similarity(vec_tags)
    return sim_bt


def save_dataframes(movies, new_df, movies2, output_dir='models'):
    """Save preprocessed dataframes as pickle files"""
    print("Saving dataframes...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save movies dataframe
    movies_dict = movies.to_dict()
    with open(os.path.join(output_dir, 'movies_dict.pkl'), 'wb') as pickle_file:
        pickle.dump(movies_dict, pickle_file)

    # Save movies2 dataframe
    movies2_dict = movies2.to_dict()
    with open(os.path.join(output_dir, 'movies2_dict.pkl'), 'wb') as pickle_file:
        pickle.dump(movies2_dict, pickle_file)

    # Save new_df dataframe
    df_dict = new_df.to_dict()
    with open(os.path.join(output_dir, 'new_df_dict.pkl'), 'wb') as pickle_file:
        pickle.dump(df_dict, pickle_file)


def train_similarity_models(new_df, output_dir='models'):
    """Train and save similarity models for different features"""
    print("Training similarity models...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Features to train similarity models for
    features = ['tags', 'genres', 'keywords', 'tcast', 'tprduction_comp']

    for feature in features:
        print(f"Training model for {feature}...")
        similarity_matrix = vectorise(new_df, feature)

        # Save similarity matrix
        with open(os.path.join(output_dir, f'similarity_tags_{feature}.pkl'), 'wb') as pickle_file:
            pickle.dump(similarity_matrix, pickle_file)

        print(f"Saved similarity model for {feature}")


def main():
    """Main function to run the model training pipeline"""
    print("Starting movie recommendation model training...")

    # Define directories
    data_dir = 'data'
    models_dir = 'models'

    # Step 1: Read and preprocess data
    movies, new_df, movies2 = read_csv_to_df(data_dir)

    # Step 2: Save preprocessed dataframes
    save_dataframes(movies, new_df, movies2, models_dir)

    # Step 3: Train and save similarity models
    train_similarity_models(new_df, models_dir)

    print("Model training completed successfully!")
    print(f"Models saved in: {models_dir}")

    # List generated files
    print("\nGenerated files:")
    for file in os.listdir(models_dir):
        if file.endswith('.pkl'):
            print(f"  - {file}")


if __name__ == '__main__':
    main()