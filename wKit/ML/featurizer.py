# coding=utf-8

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bow_tfidf(corpus, kind='bow', ngram_range=(1,2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range) if kind=='bow' else TfidfVectorizer(ngram_range=ngram_range)
    ftr = vectorizer.fit_transform(corpus)
    v = vectorizer.vocabulary_
    v = sorted(v, key=v.get(1))
    return ftr, v