from math import *
from decimal import Decimal
import numpy as np

# Calculates the euclidean distance between two vectors
def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

# Calculates the euclidean distance between two vectors using numpy
def euclidean_distance_numpy(x, y, axis = 0):
   return np.linalg.norm(x - y, axis=axis)

# Calculates the manhattan distance between two vectors
def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))

# Helper function to calculate the nth root of a number
def _nth_root(value, n_root):
    root_value = 1 / float(n_root)
    return round(Decimal(value) ** Decimal(root_value), 3)

# Calculates the minkowski distance between two vectors
def minkowski_distance(x, y, p_value = 3):
    return _nth_root(sum(pow(abs(a - b), p_value) for a, b in zip(x, y)), p_value)

# Helper function to calculate square root of a number
def _square_root(x):
    return round(sqrt(sum([a * a for a in x])), 3)

# Calculates the cosine similarity between two vectors
def cosine_similarity(x, y):
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = _square_root(x) * _square_root(y)
    return round(numerator / float(denominator), 3)

# Calculates the cosine distance between two vectors
def cosine_distance(x, y):
    distance = 1 - cosine_similarity(x, y)
    return round(distance, 3)

# Calculates the jaccard similarity between two vectors
def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


