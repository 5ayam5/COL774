import numpy as np
import nltk
import json
import re
import string
from collections import defaultdict, Counter


def extract_data(file: str, split, field='reviewText'):
	with open(file) as f:
		X = []
		Y = []
		for line in f:
			X.append(defaultdict(int))
			x = json.loads(line)
			Y.append(int(x['overall']))
			for word in split(x[field]):
				X[-1][word] += 1
		return [np.array(X), np.array(Y)]


def train_model(X: np.ndarray, Y: np.ndarray, k: int):
	vocab = defaultdict(lambda: np.ones(k))
	Phi = np.zeros(k, np.float64)
	n = np.zeros(k, np.int32)
	for x, y in zip(X, Y):
		y -= 1
		Phi[y] += 1
		for word, count in x.items():
			vocab[word][y] += count
			n[y] += count
	n += len(vocab)
	for word in vocab:
		vocab[word] = np.log(vocab[word] / n)
	n = np.log(1 / (1 + n))
	return vocab, n, np.log(Phi / np.sum(Phi))


def predict_nb(vocab: defaultdict, n: np.ndarray, Phi: np.ndarray, x: defaultdict, ignore=False):
	pred = -1
	best = -float("inf")
	for y, phi in enumerate(Phi):
		prob = phi
		for word in x:
			if word in vocab:
				prob += vocab[word][y] * x[word]
			elif not ignore:
				prob += n[y]
		if prob > best:
			best = prob
			pred = y + 1
	return pred


def accuracy(pred_Y: np.ndarray, Y: np.ndarray):
	return sum(pred_Y == Y) / Y.shape[0]


def confusion_matrix(Y: np.ndarray, pred_Y: np.ndarray, k: int):
	confusion = np.zeros((k, k), np.int32)
	for y, pred_y in zip(Y, pred_Y):
		confusion[y - 1][pred_y - 1] += 1
	return confusion


def gen_train_test(train: str, test: str, split_fn, field='reviewText'):
	print("Extracting data...")
	training_data = extract_data(train, split_fn, field)
	test_data = extract_data(test, split_fn, field)
	print("Data extracted!")
	return training_data, test_data


stop_words = nltk.corpus.stopwords.words('english')
porter = nltk.stem.PorterStemmer()
stemmed = dict()


def stem(word: str):
	if word not in stemmed:
		stemmed[word] = porter.stem(word)
	return stemmed[word]


def stem_split(s: str):
	raw = nltk.tokenize.word_tokenize(
		re.sub('[{}]'.format(string.punctuation), ' ', s))
	stop_and_stem = []
	for word in raw:
		if not word in stop_words:
			stop_and_stem.append(stem(word))
	return stop_and_stem


def bigram_split(s: str):
	stemmed = stem_split(s)
	ret = []
	for i in range(len(stemmed) - 1):
		ret.append(stemmed[i] + stemmed[i + 1])
	return ret


def bigram_split_alter(s: str):
	split_text = re.sub('[{}]'.format(string.punctuation), ' ', s).split()
	ret = []
	for i in range(len(split_text) - 1):
		ret.append(split_text[i] + split_text[i + 1])
	return ret


def trigram_split(s: str):
	split_text = s.split()
	ret = []
	for i in range(len(split_text) - 2):
		ret.append(split_text[i] + split_text[i + 1] + split_text[i + 2])
	return ret


def f1_score(confusion: np.ndarray):
	f1 = np.zeros(confusion.shape[0])
	for i in range(f1.shape[0]):
		f1[i] = 2 * confusion[i][i] / (np.sum(confusion[i, :]) + np.sum(confusion[:, i]))
	return np.average(f1)


def combine(X1: np.ndarray, X2: np.ndarray, f):
	for y in range(X1.shape[0]):
		for word in X2[y]:
			X1[y][word] += f(X2[y][word])
	return X1
