import pandas as pd

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
if __name__ == "__main__":
    movies, credits = load_data()
    df = merge_data(movies, credits)
    df = select_features(df)

    print(df.shape)
    print(df.columns)
    print(df.head())
