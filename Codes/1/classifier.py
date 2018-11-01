import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

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
	# Deljenje podataka na trening i test skup 70%-30%
	X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.3)
	# clf = KNeighborsClassifier()
	# clf = SGDClassifier()
	# Klasifikacija koriscenjem Naivnog Bajesa
	clf = MultinomialNB()
	# Fitovanje klasifikacionog modela
	clf.fit(X_train, y_train)
	# Ukoliko je model zadovoljavajuci,
	# finalna verzija modela se dobija fitovanjem
	# cele matrice i cele y kolone
	# clf.fit(tfidf_matrix, y)
	# Racunanje preciznosti modela na trening i test podacima
	print('Train acc: {}'.format(clf.score(X_train, y_train)))
	print('Test acc: {}'.format(clf.score(X_test, y_test)))
	# Ucitavanje tekstualnog fajla za test modela
	test_file = open('test.txt')
	test_text = test_file.read()
	# Transformacija test teksta u TF-IDF
	test_text_transformed = tfidf_vectorizer.transform([test_text])
	# Predikcija kategorije teksta
	print(clf.predict(test_text_transformed))
	# Ispis kategorija u sortiranom redosledu
	print(sorted(set(y)))
	# Ispis vrednosti verovatnoca NB klasifikatora
	# vezanih za pripadnosti test teksta nekoj od kategorija
	print(clf.predict_proba(test_text_transformed))

if __name__ == "__main__":
	main()