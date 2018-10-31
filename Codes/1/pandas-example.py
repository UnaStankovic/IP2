import pandas as pd

def main():
	# Pravljenje Pandas DataFrame objekta od matrice podataka
	# df = pd.DataFrame([
	# 	[1,2,3,'klasa1'],
	# 	[4,2.2,3,'klasa2'],
	# 	[5,3,1,'klasa3']
	# 	], columns=['prva', 'druga', 'treca', 'klasa'],
	# 	index=range(1,4))

	# Ucitavanje CSV fajla u  Pandas DataFrame
	df = pd.read_csv('iris.csv')

	print(df)

	# Ispis tipova kolona
	print(df.dtypes)

	# Ispis naziva kolona
	print(df.columns)

	# Ispis indeksa redova tabele
	print(df.index)

	# Ispis matrice sa vrednostima
	print(df.values)

	# Ispis reda matrice sa indeksom 0
	print(df.values[0])

	# Ispis vrednosti matrice sa indeksom 2 u redu sa indeksom 0
	print(df.values[0][2])

	# Izdvajanje kolona po nazivima
	# print(df.loc[:,['prva', 'treca']])

	# Izdvajanje kolona po indeksima
	print(df.iloc[:, range(1,3)])

	# Uslovno izdvajanje redova koji zadovoljavaju odgovarajuci uslov
	# print(df[df.prva > 2])


if __name__ == "__main__":
	main()