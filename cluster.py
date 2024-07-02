from collections import OrderedDict
from typing import Dict

import numpy as np
from sklearn.cluster import AffinityPropagation, OPTICS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances

def keep_vectors(matrix: np.ndarray) -> list:
    """
    Keep only the vectors that have at least one non-zero value, and return their indices
    :param matrix: The matrix to filter
    :return: the vectors indices to retain
    """
    return [i for i, v in enumerate(matrix) if np.any(v)]


def freqs2clustering(dic_mots):
    if not dic_mots:
        return {}

    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

    set_words = {item for item in dic_mots if len(item) > 1}
    liste_words = list(set_words)

    dic_output = {}

    V = CountVectorizer(ngram_range=(2, 3), analyzer='char', min_df=3)
    X = V.fit_transform(liste_words).toarray()

    ids = keep_vectors(X)

    X = X[ids]
    words = np.array(liste_words)[ids]
    excluded = set(liste_words) - set(words)

    matrice_def = -1 * cosine_distances(X)


    ##### CLUSTER

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
    affprop.fit_predict(matrice_def)

    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        dic = new_d.get(exemplar)
        # print(exemplar, " ==> ", list(cluster))
        if dic is not None:
            dic_output[exemplar] = {
                "Freq.centroide": dic,
                "Termes": set(cluster) - {exemplar}
            }

    for word in excluded:
        dic = new_d.get(word)
        if dic is not None:
            dic_output[word] = {
                "Freq.centroide": dic,
                "Termes": set()
            }

    return dic_output
