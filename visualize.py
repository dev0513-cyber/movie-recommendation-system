import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from recommender import (
    load_data,
    merge_data,
    select_features,
    preprocess_features,
    create_tags,
    vectorize,
    compute_similarity,
    recommend
)

# ---------------------------
# Prepare pipeline
# ---------------------------
movies, credits = load_data()
df = merge_data(movies, credits)
df = select_features(df)
df = preprocess_features(df)
df = create_tags(df)

vectors = vectorize(df)
similarity = compute_similarity(vectors)

# ---------------------------
# Create images folder if not exists
# ---------------------------
if not os.path.exists("images"):
    os.makedirs("images")

# ---------------------------
# Cosine Similarity Distribution
# ---------------------------
sim_values = similarity.flatten()
plt.figure(figsize=(8,5))
sns.histplot(sim_values, bins=50, kde=True)
plt.title("Cosine Similarity Distribution Between Movies")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Movie Pairs")
plt.tight_layout()
plt.savefig("images/cosine_similarity_distribution.png")
plt.close()  # close plot to avoid overlap

# ---------------------------
# 2D Visualization Using PCA
# ---------------------------
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

plt.figure(figsize=(10,6))
plt.scatter(vectors_2d[:,0], vectors_2d[:,1], alpha=0.5)
plt.title("Movie Vectors in 2D (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("images/pca_2d_plot.png")
plt.close()

# ---------------------------
# Top-5 Recommendations Similarity
# ---------------------------
movie = "Avatar"  # change to any movie
recommended_movies = recommend(movie, df, similarity)
movie_idx = df[df["title"] == movie].index[0]
recommended_indices = [df[df["title"] == m].index[0] for m in recommended_movies]
scores = similarity[movie_idx, recommended_indices]

plt.figure(figsize=(6,4))
sns.barplot(x=recommended_movies, y=scores)
plt.title(f"Top-5 Recommendation Similarity for '{movie}'")
plt.ylabel("Cosine Similarity")
plt.xticks(rotation=45)
plt.ylim(0,1)
plt.tight_layout()
plt.savefig(f"images/top5_similarity_{movie}.png")
plt.close()

# ---------------------------
# Average similarity metric
# ---------------------------
avg_similarity = np.mean(scores)
print(f"Average similarity of top 5 recommendations for '{movie}': {avg_similarity:.3f}")
