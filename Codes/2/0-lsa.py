import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD

# Preprocesiranje teksta
def preprocess(raw_text):
	# Izdvajanje recenica
	sentences = nltk.sent_tokenize(raw_text)
	words = []
	# Inicijalizacija stemmer-a za engleski jezik
	stemmer = SnowballStemmer('english')
	for sentence in sentences:
		# Izdvajanje reci (tokena) iz recenica
		raw_words = nltk.word_tokenize(sentence)
		for raw_word in raw_words:
			# Rec smatramo validnom ako sadrzi samo karaktere i '
			if re.search('^[a-zA-Z\']+$', raw_word):
				# Stem-ovanje reci
				word = stemmer.stem(raw_word)
				words.append(word)
	return words

def main():
	# Ucitavanje CSV fajla sa novinskim clancima
	# kategorija, tekst
	df = pd.read_csv('articles.csv')
	texts = []
	y = []
	# Izdvajanje kolone sa tekstom i kolone sa kategorijom
	for row in df.values:
		texts.append(row[1])
		y.append(row[0])
	# Inicijalizacija vektorizatora za kreiranje TF-IDF matrice
	tfidf_vectorizer = TfidfVectorizer(stop_words='english', tokenizer=preprocess, max_df=0.9)
	# Vektorizacija tekstova iz Bag of Words reprezentacije u matricnu TF-IDF reprezentaciju
	tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
	# SVD (LSA)
	lsa = TruncatedSVD(n_components=200, n_iter=100)
	# Fitovanje LSA pomocu TF-IDF matrice
	lsa.fit(tfidf_matrix)
	# Izdvajanje naziva kolona TF-IDF matrice
	terms = tfidf_vectorizer.get_feature_names()
	# Ispis kolona (reci) sa najvecim tezinama po komponentama
	for (i, comp) in enumerate(lsa.components_):
		terms_in_comp = zip(terms, comp)
		sorted_comps = sorted(terms_in_comp, key=lambda x: x[1], reverse=True)[:10]
		print("Component {}".format(i))
		print()
		for term in sorted_comps:
			print(term)
		print()
	# Transformisanje TF-IDF matrice koriscenjem LSA
	transformed = lsa.transform(tfidf_matrix)

if __name__ == "__main__":
	main()