from collections import OrderedDict
from typing import Dict

import numpy as np
from sklearn.cluster import AffinityPropagation, OPTICS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances


def compare_vectors(v1: np.ndarray, v2: np.ndarray) -> int:
    v1 = v1.toarray()[0]
    v2 = v2.toarray()[0]
    return np.sum(np.logical_and(v1, v2))


def old_freqs2clustering(dic_mots):
    if not dic_mots:
        return {}

    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

    Set_00 = set(dic_mots.keys())
    liste_words = [item for item in Set_00 if len(item) != 1]

    V = CountVectorizer(ngram_range=(2, 4), analyzer='char_wb')  # Vectorisation bigramme et trigramme de caractères
    X = V.fit_transform(liste_words)
    X_dist = cosine_distances(X)  # Distance avec cosinus

    # # Set to 1 the distance of words that have less than 2 trigram in common
    # for i in range(X.shape[0]):
    #     for j in range(X.shape[0]):
    #         if compare_vectors(X[i], X[j]) < 3:
    #             X_dist[i][j] = 1

    # matrice = []
    #
    words = np.asarray(liste_words)  # So that indexing with a list will work
    # for w in words:
    #     liste_vecteur = []
    #     for w2 in words:
    #             V = CountVectorizer(ngram_range=(2, 3), analyzer='char_wb')  # Vectorisation bigramme et trigramme de caractères
    #             X = V.fit_transform([w, w2])
    #             distance_tab1 = sklearn.metrics.pairwise.cosine_distances(X)  # Distance avec cosinus
    #             liste_vecteur.append(distance_tab1[0][1])  # stockage de la mesure de similarité
    #     matrice.append(liste_vecteur)
    # matrice_def = -1 * np.array(matrice)

    ##### CLUSTER

    # affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
    #
    # # print("="*64)
    # # print("="*64)
    # # print("="*64)
    # # print()
    # dic_output = {}
    # affprop.fit(X_dist)
    # for cluster_id in np.unique(affprop.labels_):
    #     exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
    #     cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
    #     dic = new_d.get(exemplar)
    #     # print(exemplar, " ==> ", list(cluster))
    #     if dic is not None:
    #         dic_output[exemplar] = {
    #             "Freq.centroide": dic,
    #             "Termes": list(cluster),
    #         }
    # print()
    # print("="*64)
    # print("="*64)
    # print("="*64)

    # # calculate optimal kmeans from the silhouette score
    #
    # silhouette_avg = -1
    # n_clusters = 2
    # # kmeans: KMeans = None
    # sil_by_n_clusters = []
    # # while silhouette_avg < 0.5:
    # for _ in range(distance_tab1.shape[0] - 1):
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distance_tab1)
    #     silhouette_avg = silhouette_score(distance_tab1, kmeans.labels_)
    #     sil_by_n_clusters.append(silhouette_avg)
    #     n_clusters += 1
    #
    # n_clusters = sil_by_n_clusters.index(max(sil_by_n_clusters)) + 2
    #
    # kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(distance_tab1)

    # hdb = HDBSCAN(min_cluster_size=2, metric='precomputed')
    # hdb.fit(distance_tab1)
    # dic_output = {}
    # for cluster_id in np.unique(hdb.labels_):
    #     cluster = np.unique(np.array(liste_words)[np.nonzero(hdb.labels_ == cluster_id)])
    #     for word in cluster:
    #         dic = new_d.get(word)
    #         if dic is not None:
    #             dic_output[word] = {
    #                 "Freq.centroide": dic,
    #                 "Termes": list(cluster),
    #             }

    op = OPTICS(min_samples=2, metric='precomputed')
    op.fit(X_dist)
    # dic_output = {}
    # for cluster_id in np.unique(op.labels_):
    #     cluster = np.unique(np.array(liste_words)[np.nonzero(op.labels_ == cluster_id)])
    #     for word in cluster:
    #         dic = new_d.get(word)
    #         if dic is not None:
    #             dic_output[word] = {
    #                 "Freq.centroide": dic,
    #                 "Termes": list(cluster),
    #             }
    word_by_cluster = {cluster_id: [] for cluster_id in np.unique(op.labels_)}
    for i, word in enumerate(liste_words):
        word_by_cluster[op.labels_[i]].append(word)

    dic_output = {}
    for cluster_id, words in word_by_cluster.items():
        exemplar = words[0]
        dic = new_d.get(exemplar)
        if dic is not None:
            dic_output[exemplar] = {
                "Freq.centroide": dic,
                "Termes": words,
            }

    return dic_output


def test_freqs2clustering(dic_mots: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    # print(dic_mots)
    if not dic_mots:
        return {}

    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))
    # print(new_d)

    Set_00 = set(dic_mots.keys())
    liste_words = [item for item in Set_00 if len(item) != 1]

    V = CountVectorizer(ngram_range=(2, 4), analyzer='char_wb',
                        min_df=2)  # Vectorisation bigramme et trigramme de caractères
    X = V.fit_transform(liste_words)
    X_dist = cosine_distances(X)  # Distance avec cosinus

    # Set to 1 the distance of words that have less than 2 trigram in common
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if compare_vectors(X[i], X[j]) < 3:
                X_dist[i][j] = 1

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)

    words = np.asarray(liste_words)  # So that indexing with a list will work
    dic_output = {}
    matrice_def = -1 * X_dist
    affprop.fit(matrice_def)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        dic = new_d.get(exemplar)
        # print(exemplar, " ==> ", list(cluster))
        if dic is not None:
            dic_output[exemplar] = {
                "Freq.centroide": dic,
                "Termes": list(cluster),
            }

    # put evety ones with a single word in a cluster
    for word in liste_words:
        if word not in dic_output:
            dic_output[word] = {
                "Freq.centroide": dic_mots[word],
                "Termes": [],
            }
    # print()
    # print("="*64)
    # print("="*64)
    # print("="*64)

    return dic_output


def _freqs2clustering(dic_mots: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    if not dic_mots:
        return {}

    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

    Set_00 = set(dic_mots.keys())
    liste_words = [item for item in Set_00 if len(item) != 1]
    dic_output = {}
    matrice = []

    words = np.asarray(liste_words)  # So that indexing with a list will work
    for w in words:
        liste_vecteur = []
        for w2 in words:
            V = CountVectorizer(ngram_range=(3, 5),
                                analyzer='char_wb')  # Vectorisation bigramme et trigramme de caractères
            X = V.fit_transform([w, w2]).toarray()
            distance_tab1 = cosine_distances(X)  # Distance avec cosinus
            liste_vecteur.append(distance_tab1[0][1])  # stockage de la mesure de similarité
        matrice.append(liste_vecteur)
    matrice_def = -1 * np.array(matrice)

    ##### CLUSTER

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)
    affprop.fit(matrice_def)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        dic = new_d.get(exemplar)
        # print(exemplar, " ==> ", list(cluster))
        if dic is not None:
            dic_output[exemplar] = {
                "Freq.centroide": dic,
                "Termes": list(cluster),
            }
    # print()
    # print("="*64)
    # print("="*64)
    # print("="*64)

    return dic_output


def freqs2clustering(dic_mots):
    if not dic_mots:
        return {}

    new_d = OrderedDict(sorted(dic_mots.items(), key=lambda t: t[0]))

    Set_00 = set(dic_mots.keys())
    liste_words = [item for item in Set_00 if len(item) != 1]
    dic_output = {}
    matrice = []

    words = np.asarray(liste_words)  # So that indexing with a list will work
    for w in words:
        liste_vecteur = []
        for w2 in words:
            V = CountVectorizer(ngram_range=(2, 3),
                                analyzer='char')  # Vectorisation bigramme et trigramme de caractères
            X = V.fit_transform([w, w2]).toarray()
            distance_tab1 = cosine_distances(X)  # Distance avec cosinus
            liste_vecteur.append(distance_tab1[0][1])  # stockage de la mesure de similarité
        matrice.append(liste_vecteur)
    matrice_def = -1 * np.array(matrice)

    ##### CLUSTER

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5, random_state=None)

    # print("="*64)
    # print("="*64)
    # print("="*64)
    # print()
    affprop.fit(matrice_def)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        dic = new_d.get(exemplar)
        # print(exemplar, " ==> ", list(cluster))
        if dic is not None:
            dic_output[exemplar] = {
                "Freq.centroide": dic,
                "Termes": list(cluster),
            }
    # print()
    # print("="*64)
    # print("="*64)
    # print("="*64)

    return dic_output
