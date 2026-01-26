from flask import Flask, render_template, request  # pyright: ignore[reportMissingImports]
from recommender import (
    get_closest_match,
    load_data,
    merge_data,
    select_features,
    preprocess_features,
    create_tags,
    vectorize,
    compute_similarity,
    recommend
)

app = Flask(__name__)

# Load ML pipeline ONCE
movies, credits = load_data()
df = merge_data(movies, credits)
df = select_features(df)
df = preprocess_features(df)
df = create_tags(df)

vectors = vectorize(df)
similarity = compute_similarity(vectors)
movie_titles = df["title"].values

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []

    if request.method == "POST":
        movie_name = request.form.get("movie")
        if movie_name:
            movie_name = get_closest_match(movie_name, movie_titles)
            recommendations = recommend(movie_name, df, similarity)  # list of (title, poster)

    return render_template(
        "index.html",
        movie_list=movie_titles,
        recommendations=recommendations
    )
if __name__ == "__main__":
    app.run(debug=True)