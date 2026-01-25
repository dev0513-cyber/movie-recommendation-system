import pandas as pd
import ast


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

if __name__ == "__main__":
    movies, credits = load_data()
    df = merge_data(movies, credits)
    df = select_features(df)
    df = preprocess_features(df)
    df = create_tags(df)

    print(df[["title", "tags"]].head(1))
