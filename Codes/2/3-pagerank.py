import numpy as np
from numpy.linalg import matrix_power
import urllib.request
from collections import deque
import re
from transliterate import translit
from googletrans import Translator

# Formiranje ulazne matrica za PageRank algoritam
# od dobijenog grafa
def generatePRMatrix(G):
	keys = {}
	num = 0
	for key in G:
		keys[key] = num
		num += 1
		for key2 in G[key]:
			if key2 not in keys:
				keys[key2] = num
				num += 1
	A = [[0 for i in range(num)] for j in range(num)]
	for v in G:
		neighbors = G[v]
		num = len(neighbors)
		for w in neighbors:
			A[keys[w]][keys[v]] = 1/num
	return np.matrix(A)

# Izracunavanje znacajnosti stranica pomocu PageRank algoritma.
# Matrica A je dobijena od grafa stranica
def page_rank(A):
	n = A.shape[0]
	# Pocetne vrednosti svake stranice su jednake i iznose 1/n,
	# gde je n ukupan broj stranica
	v = np.matrix([1/n for i in range(n)])
	return matrix_power(A, n).dot(np.transpose(v))

def transliterate(text):
	return translit(text, 'sr', reversed=True)

def get_webpage(URL):
	req = urllib.request.Request(URL)
	response = urllib.request.urlopen(req)
	data = response.read().decode('utf-8')
	data = data.replace('\n', '')
	data = data.replace('\r', '')
	data = data.replace('\t', '')
	data = data.replace(',', '')
	data = data.replace('\'', '')
	space_regex = re.compile(r'[ ]+')
	data = space_regex.sub(' ', data)
	return transliterate(data)

def clean(rawHTML):
	tags = re.compile(r'<.*?>|\&.+;')
	return tags.sub('', rawHTML)

def extract_elements(element, rawHTML):
	regex = '<'+element+'(.*?)>(.*?)</'+element+'>'
	raw_elements = re.findall(regex, str(rawHTML))
	elements = []
	for i in range(len(raw_elements)):
		elements.append((raw_elements[i][0], clean(raw_elements[i][1])))
	return elements

def extract_attributes(attribute, element):
	regex = attribute + '="(.*?)"'
	attributes = re.findall(regex, element[0])
	return attributes

def get_neighbors(base, URL, relative_only=True):
	if URL[0] == '/':
		URL = base + URL
	pageHTML = get_webpage(URL)
	elements = extract_elements('a', pageHTML)
	links = []
	for element in elements:
		links += extract_attributes('href', element)
	if relative_only:
		links = list(filter(lambda x: x[0] == '/', links))
		return [base + link for link in links]	
	else:
		return links

def addEdge(G, v_from, v_to):
	if v_from not in G:
		G[v_from] = []
	if v_to not in G[v_from]:
		G[v_from].append(v_to)

def bfs(startPage):
	marked = {}
	marked[startPage] = True
	queue = deque(['/'])
	base = startPage
	G = {}
	limit = 300
	currNum = 0;
	while len(queue) > 0:
		if (currNum == limit):
			break;
		curr = queue.popleft()
		print(curr)
		neighbors = get_neighbors(base, startPage, True)
		for neighbor in neighbors:
			addEdge(G, curr, neighbor)
			if neighbor not in marked and not re.search('#', neighbor):
				marked[neighbor] = True
				queue.append(neighbor)
				currNum += 1
	return G

def main():
	G = bfs('http://www.politika.rs')
	A = generatePRMatrix(G)
	print(page_rank(A))

if __name__ == "__main__":
	main()