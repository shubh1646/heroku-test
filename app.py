
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import linear_kernel
import sys
from sklearn.metrics.pairwise import linear_kernel

def cosine_simi():
    df = pd.read_csv('data.csv')
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    df['overview'] = df['overview'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def get_recommendations(title, cosine_sim=cosine_simi()):
    df = pd.read_csv('data.csv')

    if title not in df['title'].unique():
        return('This movie is not in our database.\nPlease check if you spelled it correct.')
    else:
	    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
	    # Get the index of the movie that matches the title
	    idx = indices[title]

	    # Get the pairwsie similarity scores of all movies with that movie
	    sim_scores = list(enumerate(cosine_sim[idx]))

	    # Sort the movies based on the similarity scores
	    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

	    # Get the scores of the 10 most similar movies
	    sim_scores = sim_scores[1:11]

	    # Get the movie indices
	    movie_indices = [i[0] for i in sim_scores]

	    # Return the top 10 most similar movies
	    #return df['title'].iloc[movie_indices]


	    # making an empty list that will containg all 10 movie recommendations
	    l = []
	    for i in range(len(sim_scores)):
	        a = sim_scores[i][0]
	        l.append(df['title'][a])
	    return l

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    title = request.args.get('title')
    recommendation = get_recommendations(title)
    #print(recommendation, file=sys.stderr)
    if type(recommendation)==type('string'):
        return render_template('recommend.html',movie=title,r=recommendation,t='s')
    else:
        return render_template('recommend.html',movie=title,r =recommendation,t='l')
    
    

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
