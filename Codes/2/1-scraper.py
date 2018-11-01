import re
import urllib.request
from transliterate import translit
from collections import deque

# Ciscenje sadrzaja HTML taga od ostalih HTML tagova koji
# se nalaze unutar sadrzaja kao i specijalnih HTML karaktera
# oblika &tekst;
def clean(raw_html):
	tags = re.compile('<.*?>|\&[a-z]+;')
	return tags.sub('', raw_html)

# Prevodjenje teksta iz cirilice u latinicu.
# Ako je tekst vec cirilicni, tekst ce ostati neizmenjen
def transliterate(text):
	return translit(text, 'sr', reversed=True)

# Preuzimanje sadrzaja HTML stranice
def get_webpage(URL):
	try:
		# Pripremanje zahteva za trazenim URL
		req = urllib.request.Request(URL)
		# Otvaranje konekcije ka stranici
		response = urllib.request.urlopen(req)
		# Citanje i dekodiranje sadrzaja stranice
		data = response.read().decode('utf-8')
		# Transliteracija
		data = transliterate(data)
		# Brisanje karaktera prelaska u novi red, 
		# zareza, tabulatora i apostrofa
		data = data.replace('\n', '')
		data = data.replace('\t', '')
		data = data.replace('\r', '')
		data = data.replace(',', '')
		data = data.replace('\'', '')
		# Uklanjanje suvisnih razmaka '____' -> '_'
		space_regex = re.compile(r'[ ]+')
		data = space_regex.sub(' ', data)
	except:
		data = ''
	return data

# Izdvajanja sadrzaja zadatih HTML elemenata
def extract_elements(element, raw_html):
	regex = '<' + element + '(.*?)>(.*?)</'+element+'>'
	raw_elements = re.findall(regex, raw_html)
	for i in range(len(raw_elements)):
		raw_elements[i] = (raw_elements[i][0], clean(raw_elements[i][1]))
	return raw_elements

# Izdvajanje sadrzaja zadatih atributa iz HTML elemenata
def extract_attributes(attribute, attribute_string):
	regex = attribute + '="(.*?)"'
	attributes = re.findall(regex, attribute_string)
	return attributes
	
# Izdvajanja sadrzaja tekstualnih elemenata sa stranice zadate URL-om
def extract_texts(URL):
	texts = []
	page_html = get_webpage(URL)
	texts += [el[1] for el in extract_elements('h1', page_html)]
	texts += [el[1] for el in extract_elements('h2', page_html)]
	texts += [el[1] for el in extract_elements('h3', page_html)]
	texts += [el[1] for el in extract_elements('h4', page_html)]
	texts += [el[1] for el in extract_elements('h5', page_html)]
	texts += [el[1] for el in extract_elements('h6', page_html)]
	texts += [el[1] for el in extract_elements('p', page_html)]
	return " ".join(texts)

# Izdvajanje vrednosti href atributa svih linkova na stranici zadatoj URL-om
def extract_links(URL):
	raw_links = []
	page_html = get_webpage(URL)
	raw_links = extract_elements('a', page_html)
	links = []
	for link in raw_links:
		hrefs = extract_attributes('href', link[0])
		if len(hrefs) > 0:
			url = hrefs[0]
			# Filtriranje unutrasnjih linkova (#...) i apsolutnih adresa
			if len(url) > 0 and url[0] == '/' and not re.search('#', url):
				links.append(url)
	return links

# Nalazenje liste susednih strana za zadatu stranu
# base - domen
# path - putanja
def get_neighbors(base, path):
	links = extract_links(base + path)
	return links

# BFS obilazak stranica od zadate pocetne adrese
def bfs(start_page_URL):
	queue = deque(['/'])
	base = start_page_URL
	marked = {}
	marked['/'] = True
	texts = []
	limit = 200
	current_count = 0
	while len(queue) > 0:
		curr = queue.popleft()
		print(curr)
		# Ako je tekst clanak, izdvaja se njegov sadrzaj
		if re.search('clanak', curr):
			print('clanak:')
			texts.append((base + curr, extract_texts(base + curr)))
			current_count += 1
			if current_count == limit:
				return texts
		# U suprotnom, nastavlja se obilazak od trenutne stranice
		else:
			for neighbor in get_neighbors(base, curr):
				if neighbor not in marked:
					marked[neighbor] = True
					queue.append(neighbor)
	return texts

def main():
	article_texts = bfs('http://www.politika.rs')
	# Priprema CSV formata
	header = 'url, text\n'
	content = ''
	# Formatiranje podataka za CSV zapis
	for text in article_texts:
		content += text[0] + ',' + text[1] + '\n'
	# Sastavljanje sadrzaja CSV fajla
	articlesCSV = header + content
	# Zapisivanje CSV fajla
	output = open('scraped.csv', 'w')
	output.write(articlesCSV)
	
if __name__ == "__main__":
	main()