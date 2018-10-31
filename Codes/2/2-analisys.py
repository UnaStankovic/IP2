import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import stemmer
from sklearn.cluster import KMeans

# Stop reci na srpskom jeziku
stopwords_file = open('stopwords-sr.txt')
stopwords = stopwords_file.read().split('\n')

# Preprocesiranje teksta
def preprocess(raw_text):
	# Stemmer za srpski jezik
	raw_words = stemmer.stem_arr(raw_text)
	words = []
	
	for raw_word in raw_words:
		# Rec smatramo validnom ako sadrzi samo karaktere i '
		if re.search('^[a-zA-ZćžčšđĆŽČŠĐ\']+$', raw_word):
			# Stem-ovanje reci
			words.append(raw_word)

	return words


def main():
	# Ucitavanje CSV fajla sa novinskim clancima
	# kategorija, tekst
	df = pd.read_csv('scraped.csv')

	texts = []
	urls = []

	# Izdvajanje kolone sa tekstom i kolone sa kategorijom
	for row in df.values:
		texts.append(row[1])
		urls.append(row[0])


	# Inicijalizacija vektorizatora za kreiranje TF-IDF matrice
	tfidf_vectorizer = TfidfVectorizer(tokenizer=preprocess, max_df=0.9, ngram_range=(1,3))

	# Vektorizacija tekstova iz Bag of Words reprezentacije u matricnu TF-IDF reprezentaciju
	tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

	# Klasterovanje clanaka K-Means algoritmom
	# n = 11
	# km = KMeans(n_clusters=n, n_init=300)

	# km.fit(tfidf_matrix)

	# clusters = {}

	# for i in range(len(urls)):
	# 	cluster_num = km.labels_[i]

	# 	if cluster_num not in clusters:
	# 		clusters[cluster_num] = []
	# 	clusters[cluster_num].append(urls[i])

	# for i in clusters:
	# 	print('Cluster {}'.format(i))
	# 	print()
	# 	for url in clusters[i]:
	# 		print(url)
	# 	print()

	# Pretraga stranica na osnovu trazenih pojmova
	query_string = "vesti iz regiona"

	query_transformed = tfidf_vectorizer.transform([query_string])

	query_column = np.transpose(query_transformed)

	product = tfidf_matrix.dot(query_column)

	weights = [0 for i in range(len(urls))]

	for i in range(len(urls)):
		weights[i] = product[i, 0]

		zipped_product = zip(urls, weights)

		# Vracaju se prvih 5 najbitnijih stranica za zadate pojmove
		sorted_product = sorted(zipped_product, key=lambda x: x[1], reverse=True)[:5]

	for (url, w) in sorted_product:
		print(url)


if __name__ == "__main__":
	main()