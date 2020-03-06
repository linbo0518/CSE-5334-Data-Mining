import numpy as np
from p1.kmeans import KMeans
from p2.tfidf import TFIDF

# Generate 2D Gaussian data
mu_1 = np.array([1, 0])
mu_2 = np.array([0, 1.5])
sigma_1 = np.array([[0.9, 0.4], [0.4, 0.9]])
sigma_2 = np.array([[0.9, 0.4], [0.4, 0.9]])

gaussian_1 = np.random.multivariate_normal(mu_1, sigma_1, 500)
gaussian_2 = np.random.multivariate_normal(mu_2, sigma_2, 500)

x = np.concatenate([gaussian_1, gaussian_2])

# KMeans
# k = 2
k = 2
model = KMeans(k)
centers, labels, n_iter = model.fit(x)

# k = 2 with given centers
k = 2
c = ((10, 10), (-10, -10))
model = KMeans(k, c)
centers, labels, n_iter = model.fit(x)

# k = 4 with given centers
k = 4
c = ((10, 10), (-10, -10), (10, -10), (-10, 10))
model = KMeans(k, c)
centers, labels, n_iter = model.fit(x)

# tfidf
model = TFIDF()
tfidf = model.fit('p2/Amazon_Reviews.csv')

# normalize
img = (tfidf / tfidf.max()) * 255

# 5 postive and 5 negative
postive_words = ["amazing", "favourite", "best", "good", "great"]
negative_words = ["bad", "fake", "poor", "disappointed", "negative"]
selected = postive_words + negative_words
tfidf = model.fit('p2/Amazon_Reviews.csv', selected)

# normalize
img = (tfidf / tfidf.max()) * 255

# docs vector
postive_vector = np.sum(tfidf[:, :5], axis=1, keepdims=True)
negative_vector = np.sum(tfidf[:, 5:], axis=1, keepdims=True)
docs_vector = np.concatenate((postive_vector, negative_vector), axis=1)

# k = 2
k = 2
model = KMeans(k)
centers, labels, n_iter = model.fit(docs_vector)

# k = 3
k = 3
model = KMeans(k)
centers, labels, n_iter = model.fit(docs_vector)

# k = 4
k = 4
model = KMeans(k)
centers, labels, n_iter = model.fit(docs_vector)
