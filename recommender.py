import pandas as pd  
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  
from fuzzywuzzy import process
TMDB_API_KEY = "e34aab34678515db586350f332d0c747"

def fetch_poster(movie_title):
    """Returns the full poster URL for a movie title"""
    try:
        # Search for movie by title
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        response = requests.get(url)
        data = response.json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            full_path = "https://image.tmdb.org/t/p/w500" + poster_path
            return full_path
        else:
            return ""
    except:
        return ""
def get_closest_match(movie_name, movie_list):
    match, score = process.extractOne(movie_name, movie_list)
    return match

def load_data():
    movies = pd.read_csv("data/tmdb_5000_movies.csv")
    credits = pd.read_csv("data/tmdb_5000_credits.csv")
    return movies, credits
def merge_data(movies, credits):
    df = movies.merge(credits, left_on="id", right_on="movie_id")
    df = df.rename(columns={"title_x": "title"})
    df = df.drop(columns=["title_y"])
    return df
def select_features(df):
    return df[
        ["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]
    ]
def convert(text):
    """
    Converts JSON-like string to list of names
    """
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items]
    except:
        return []
def convert_cast(text):
    try:
        items = ast.literal_eval(text)
        return [item["name"] for item in items[:3]]
    except:
        return []
def fetch_director(text):
    try:
        items = ast.literal_eval(text)
        for item in items:
            if item["job"] == "Director":
                return [item["name"]]
        return []
    except:
        return []
def preprocess_features(df):
    df["genres"] = df["genres"].apply(convert)
    df["keywords"] = df["keywords"].apply(convert)
    df["cast"] = df["cast"].apply(convert_cast)
    df["crew"] = df["crew"].apply(fetch_director)

    df["overview"] = df["overview"].fillna("").apply(lambda x: x.split())

    return df
def create_tags(df):
    df["tags"] = (
        df["overview"]
        + df["genres"]
        + df["keywords"]
        + df["cast"]
        + df["crew"]
    )

    df["tags"] = df["tags"].apply(lambda x: " ".join(x).lower())
    return df

def vectorize(df):
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(df["tags"]).toarray()
    return vectors

def compute_similarity(vectors):
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie_title, df, similarity):
    movie_index = df[df["title"] == movie_title].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for idx, score in movie_list:
        title = df.iloc[idx]["title"]
        poster = fetch_poster(title)
        recommendations.append((title, poster, f"{round(score*100, 2)} %"))
    
    return recommendations


if __name__ == "__main__":
    movies, credits = load_data()
    df = merge_data(movies, credits)
    df = select_features(df)
    df = preprocess_features(df)
    df = create_tags(df)

    vectors = vectorize(df)
    similarity = compute_similarity(vectors)

    print(recommend("Avatar", df, similarity))

