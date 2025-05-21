import numpy as np
from collections import Counter


def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def best_split(X, y):
    m, n = X.shape
    best_gain = 0
    best_feature = None
    best_threshold = None
    current_entropy = entropy(y)
    for feature in range(n):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            p = float(len(left)) / len(y)
            gain = current_entropy - p * entropy(left) - (1 - p) * entropy(right)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


def build_tree(X, y, depth=0, max_depth=5):
    if len(set(y)) == 1 or depth >= max_depth:
        most_common = Counter(y).most_common(1)[0][0]
        return Node(value=most_common)
    feature, threshold = best_split(X, y)
    if feature is None:
        most_common = Counter(y).most_common(1)[0][0]
        return Node(value=most_common)
    left_idx = X[:, feature] <= threshold
    right_idx = X[:, feature] > threshold
    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)
    return Node(feature, threshold, left, right)


def predict_one(node, x):
    while node.value is None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value


def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])