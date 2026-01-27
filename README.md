# movie-recommendation-system
Content-based movie recommendation system using NLP and cosine similarity

## Problem Statement
The objective of this project is to build a movie recommendation system that suggests relevant movies to users based on the content similarity of movies using machine learning techniques.

---

## Project Overview
This project implements a **content-based movie recommendation system** using textual features such as genres, overview, keywords, cast, and crew. The system analyzes these features to recommend movies that are most similar to a user-selected movie.

The model is deployed as a **Flask web application** with an interactive and user-friendly interface, making it suitable as a real-world, internship-ready machine learning project.

---

## Dataset
The project uses the **TMDB Movie Dataset**:
- `movies.csv` – movie metadata
- `credits.csv` – cast and crew information
- Approximately **5,000 movies**
- Target is not a label but **movie similarity**

---

## Approach
1. Loaded and merged movie and credit datasets  
2. Selected relevant textual features for recommendation  
3. Cleaned and preprocessed text data  
4. Combined all text features into a single `tags` column  
5. Converted text into vectors using **CountVectorizer**  
6. Computed similarity using **Cosine Similarity**  
7. Recommended the **top 5 most similar movies**  

---

## Model & Evaluation
This is a **recommendation system**, so traditional accuracy metrics are not applicable.

### Evaluation Metric
- **Average similarity score of top 5 recommendations**

| Average Top-5 Similarity | ~30–45% |

This score indicates that the recommended movies share meaningful content similarity with the input movie, which is considered good for content-based filtering systems.

---

## Visualizations
The following visualizations are generated to analyze model behavior:
- Cosine similarity distribution
- PCA-based 2D visualization of movie vectors
- Similarity score comparison for top 5 recommendations

(All visualizations are stored in the `images/` folder.)

---

## Key Findings
- Content-based filtering effectively recommends similar movies  
- Genres, overview, and keywords contribute most to similarity  
- Cosine similarity provides interpretable and consistent results  
- The system performs well for both popular and less-known movies  

---

## Tech Stack
- **Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK  
- **Web Framework:** Flask  
- **Visualization:** Matplotlib  
- **API:** TMDB API  
- **Tools:** Git, GitHub  

---

## How to Run
**Clone the repository:
   
git clone https://github.com/dev0513-cyber/movie-recommendation-system.git
cd movie-recommendation-system

**Install dependencies:

pip install -r requirements.txt


**Add your TMDB API key in recommender.py

**Run the application:

python app.py


**Open in browser:

http://127.0.0.1:5000/
