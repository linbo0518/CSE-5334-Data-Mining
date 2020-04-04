import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TFIDF:

    def __init__(self):
        self._tokens = None
        self._tf = None
        self._idf = None

    def fit(self, filepath, selected_words=None):
        sentences = self._read_data(filepath)
        docs, words = self._preprocessing(sentences)
        if selected_words:
            words = np.array(selected_words)

        cfs = []
        for e in docs:
            e = e.split(" ")
            cf = [e.count(word) for word in words]
            cfs.append(cf)

        tfs = []
        tfs.extend(
            np.array(e) / (len(doc.split(" "))) for e, doc in zip(cfs, docs))
        N = len(docs)

        dfs = list(np.zeros(words.size, dtype=int))
        for i in range(words.size):
            for doc in docs:
                if doc.find(words[i]) != -1:
                    dfs[i] += 1

        idfs = [(np.log10(N * 1.0 / (1 + e))) for e in dfs]

        tfidfs = []
        for i in range(len(docs)):
            word_tfidf = np.multiply(tfs[i], idfs)
            tfidfs.append(word_tfidf)
        return np.array(tfidfs)

    def _read_data(self, filepath, use_col='Review'):
        df = pd.read_csv(filepath)
        sentences = []
        for idx in df.index:
            senten = df[use_col][idx]
            sentences.append(senten.lower())
        return sentences

    def _preprocessing(self, sentences):
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()
        tokenize_sentences = []
        for senten in sentences:
            token_senten = []
            tokens = tokenizer.tokenize(senten)
            for token in tokens:
                if token not in stop_words:
                    token = lemmatizer.lemmatize(token)
                    token_senten.append(token)
            tokenize_sentences.append(token_senten)
        docs = []
        for token_senten in tokenize_sentences:
            senten = ""
            for idx, token in enumerate(token_senten):
                if idx == len(token_senten) - 1:
                    senten = senten + token
                else:
                    senten = senten + token + ' '
            docs.append(senten)

        words = set()
        for token_list in tokenize_sentences:
            for token in token_list:
                words.add(token)
        return np.array(docs), np.array(list(words))

    def test(self, file):
        sen = self._read_data(file)
        return self._preprocessing(sen)