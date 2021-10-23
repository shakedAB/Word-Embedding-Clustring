# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:26:15 2021

@author: Owner
"""

import gensim
from gensim.models import Word2Vec
import pandas as pd
import numpy as np 
from termcolor import colored
from sklearn.utils import resample
#conda install -c conda-forge scikit-learn-extra
from sklearn_extra import cluster
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import silhouette_score 
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster
from sklearn import preprocessing  # to normalise existing X
import math

#%matplotlib qt
# %matplotlib inline

# =============================================================================
# Global variables
# =============================================================================
tokens_labels = pd.DataFrame(columns = ["token","cluster"])
with open('listfile.data', 'rb') as filehandle:
    # read the data as binary data stream
    emails_after_process_list = pickle.load(filehandle)
#emails_after_process_list = pd.read_csv(r"C:\Users\Owner\Desktop\פרויקט גמר\emails_after_process.csv")
with open ('model_pickle','rb') as w2v:
   W2V_model =  pickle.load(w2v)

# =============================================================================
# Word Embedding model & parameters
# =============================================================================

W2V_model = Word2Vec(emails_after_process_list, min_count=4 ,size=100 ,
                     window=5 ,negative=5 , sg=0)
X = W2V_model.wv[W2V_model.wv.vocab]   #sparse matrix 

X_Norm = preprocessing.normalize(X)

vocabulary = list(W2V_model.wv.vocab)

# =============================================================================
# Word Groups
# =============================================================================

GroupA = pd.read_csv(r"C:\Users\abrahamy\Desktop\פרויקט גמר- שקד\חלוקת מילים לקבוצות\GroupA.csv")
GroupB = pd.read_csv(r"C:\Users\abrahamy\Desktop\פרויקט גמר- שקד\חלוקת מילים לקבוצות\GroupB.csv")
GroupC = pd.read_csv(r"C:\Users\abrahamy\Desktop\פרויקט גמר- שקד\חלוקת מילים לקבוצות\GroupC.csv")
GroupD = pd.read_csv(r"C:\Users\abrahamy\Desktop\פרויקט גמר- שקד\חלוקת מילים לקבוצות\GroupD.csv")
GroupA.columns,GroupB.columns,GroupC.columns,GroupD.columns = ["words"],["words"],["words"],["words"]
Groups = pd.concat([GroupA, GroupB,GroupC,GroupD])
Groups = [w for w in Groups["words"] if w in vocabulary]
All_Groups = [GroupA,GroupB,GroupC,GroupD]

def Group_words_embedding():
    Group_words_embedding= np.zeros((1,X.shape[1])) #100 dor my model \ 300 for google
    for word in Groups:
            b = W2V_model.wv[word]
            Group_words_embedding = np.vstack([Group_words_embedding, b])
    Group_words_embedding = np.delete(Group_words_embedding, 0, 0)
    return Group_words_embedding
x= Group_words_embedding()

def Group_words_embedding_normelized():
    Group_words_embedding_normelized = np.zeros((1,X.shape[1]))
    for word in Groups:
        inde = W2V_model.wv.vocab[word].index
        b = X_Norm[inde]
        Group_words_embedding_normelized = np.vstack([Group_words_embedding_normelized, b])
    Group_words_embedding_normelized = np.delete(Group_words_embedding_normelized, 0, 0)
    return Group_words_embedding_normelized

# =============================================================================
# Entropy
# =============================================================================
def Calculate_Entropy(list_of_prob):
    Entropy = 0
    for pro in list_of_prob:
        if pro == 0:
            continue
        Entropy =  Entropy - pro*math.log(pro)
    return Entropy

def Entropy_All_G(k):
    for g in All_Groups:
        w = g["words"]
        distribution_of_Group = Group_distribution(w,k,labels,centroids)
        Entropy = Calculate_Entropy(distribution_of_Group)
        print(f" in group {g} the entropy is : {Entropy} ")
        
def Entropy_All_G_Dendogram(k):
    for g in All_Groups:
        w = g["words"]
        distribution_of_Group = Group_distribution_dendogram(w,k)
        Entropy = Calculate_Entropy(distribution_of_Group)
        print(f" in group {g} the entropy is : {Entropy} ")
        
# =============================================================================
# finding the optimal K 
# =============================================================================
from sklearn import cluster
sil = []
kmax = 30
'silhouette_score'
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(10, kmax+1, 4):
  kmeans = cluster.KMeans(n_clusters = k ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
  kmeans.fit(X_Norm)
  labels = kmeans.labels_
  sil.append(silhouette_score(X, labels, metric = 'euclidean'))
  print(f"the shill score for {k} clusters is {sil[k-2]}")




'Elbow Method for kmeans '
wcss=[] # Within-Cluster-Sum of Squared
for k in range(2,30,2):
    print(f" k = {k} ")
    kmeans = cluster.KMeans(n_clusters = k ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
    kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(4,60,4),wcss)
plt.title('The Elbow Method for KMedoids ')
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = cluster.KMeans(n_clusters = 11 ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
kmeans.fit(X)
Entropy_All_G(11)
# =============================================================================
# finding the optimal K - data normelized
# =============================================================================
from sklearn import cluster
sil = []
kmax = 30
'silhouette_score'
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(10, kmax+1, 4):
  kmeans = cluster.KMeans(n_clusters = k ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
  kmeans.fit(X_Norm)
  labels = kmeans.labels_
  sil.append(silhouette_score(X_Norm, labels, metric = 'euclidean'))
  print(f"the shill score for {k} clusters is {sil[k-2]}")




'Elbow Method for kmeans '
wcss=[] # Within-Cluster-Sum of Squared
for k in range(2,30,2):
    print(f" k = {k} ")
    kmeans = cluster.KMeans(n_clusters = k ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
    kmeans.fit_predict(X_Norm)
    wcss.append(kmeans.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(4,60,4),wcss)
plt.title('The Elbow Method for KMedoids ')
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

kmeans = cluster.KMeans(n_clusters = 8 ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
kmeans.fit(X_Norm)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
Entropy_All_G(8)

# =============================================================================
#                                            kmeans
# =============================================================================
from sklearn import cluster
kmeans = cluster.KMeans(n_clusters = 6 ,init='k-means++',
                        n_init=20, max_iter=300, tol=0.0001, 
                        verbose=0, random_state=678,
                        copy_x=True, algorithm='auto')
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print (colored('Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):' , 'red'))
print (kmeans.score(X))
silhouete_score6 = metrics.silhouette_score(X, labels, metric='euclidean')
print (colored('Silhouette_score: ', 'red'))
print (silhouete_score6) 
tokens_labels = pd.DataFrame(columns = ["token","cluster"])
for idx,l in enumerate(kmeans.labels_):
    #print(l,vocabulary[idx])
    tokens_labels.loc[len(tokens_labels.index)] = [vocabulary[idx],l]

def cluster_distrebution(k,tokens_labels):
    t =[0] * k
    for i in tokens_labels["cluster"]:
        x = t[i]
        t[i]  = x + 1
    for i in range(len(t)):
        t[i]= t[i]/len(tokens_labels)
    print(t)
    return t
cluster_distrebution6 = cluster_distrebution(6,tokens_labels)
Entropy6 = Calculate_Entropy(cluster_distrebution6)
# =============================================================================
# Group Distribuation
# =============================================================================
GroupA = pd.read_csv(r"C:\Users\abrahamy\Desktop\פרויקט גמר- שקד\חלוקת מילים לקבוצות\GroupA.csv")
GroupA = GroupA["words"]
    
def Group_distribution(group,k,Labels,Centroids):
    labels_list = list(Labels)
    newgroup = []
    sum_from_cluster = [0]*k
    words_by_cluster =  [[] for _ in range(k)]
    distribution_of_cluster = [0]*k
    avg_distance_from_prim_centroid = [0]*k
    for t in group:
        t = t.lower()
        newgroup.append(t)
    
    for word in newgroup:
       if word in vocabulary:
           index = vocabulary.index(word)
           k_num = labels_list[index]
           sum_from_cluster[k_num] = sum_from_cluster[k_num] + 1
           words_by_cluster[k_num].append(word)
       else:
            print(f"{word} not in vocabulary")
    total_num_from_G = sum(sum_from_cluster)
    for i in range(len(sum_from_cluster)):
        distribution_of_cluster[i] = sum_from_cluster[i]/total_num_from_G
    print(f"sum_from_cluster : {sum_from_cluster}")
    print(f"distribution_of_cluster : {distribution_of_cluster}")
    prim_cluster = sum_from_cluster.index(max(sum_from_cluster))
    print(f"the primary cluster is : {prim_cluster}")
    prim_center = Centroids[prim_cluster]
    for i in range(len(words_by_cluster)):
        for w in words_by_cluster[i]:
            wmbedding_of_word = W2V_model.wv[w]
            d = distance.euclidean(wmbedding_of_word,prim_center)
            avg_distance_from_prim_centroid[i] = avg_distance_from_prim_centroid[i] + d
    for j in range(len(avg_distance_from_prim_centroid)):
        if sum_from_cluster[j] != 0 :
            avg_distance_from_prim_centroid[j]=  avg_distance_from_prim_centroid[j]/sum_from_cluster[j]
    print(f"the avg_distance_from_prim_centroid is : {avg_distance_from_prim_centroid}")
    return distribution_of_cluster
        
 
distribution_of_cluster = Group_distribution(GroupA,6,labels,centroids)
Entropy6 = Calculate_Entropy(distribution_of_cluster)

Entropy_All_G(6)
# =============================================================================
# Clustering goodness with Entropy
#The entropy is 0 if all records of the cluster belong to the same posture,
# and the entropy is maximal if we have a uniform postures distribution
# =============================================================================


# =============================================================================
#                                            Denedogram
# =============================================================================
#Group_words_embedding = Group_words_embedding()
Group_words_embedding_normelized = Group_words_embedding_normelized()
l = linkage(Group_words_embedding_normelized, method='complete')

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.ylabel('word')
plt.xlabel('distance')

dendrogram(
    l,
    leaf_rotation=75.,  # rotates the x axis labels
    leaf_font_size=16.,  # font size for the x axis labels
    orientation='top',
    leaf_label_func=lambda v: str(W2V_model.wv.index2word[v]),
    
)
plt.show()
# in linkage the index of cluster start with 1, ot like k means that start with 0
clusters = fcluster(l,11, criterion='maxclust')
Group_labels = pd.DataFrame(columns = ["word","cluster"])
Group_labels["word"] = Groups
Group_labels["cluster"] = clusters        

def cluster_distrebution_dendogram(k,tokens_labels):
    t =[0] * k
    for i in tokens_labels["cluster"]:
        x = t[i]
        t[i]  = x + 1
    for i in range(len(t)):
        t[i]= t[i]/len(tokens_labels)
    print(t)
    return t
cluster_distrebution6 = cluster_distrebution_dendogram(12,Group_labels)

# idw = Group_labels[Group_labels['word']=='busy'].index.values
# c = Group_labels["cluster"][idw].astype(int)
# c = c.astype('int')

def Group_distribution_dendogram(group,k):  
    sum_from_cluster = [0]*(k+1)
    distribution_of_cluster = [0]*(k+1)
    for word in group: 
        if word in Groups:
            index = Groups.index(word)
            k_num = clusters[index]
            sum_from_cluster[k_num] = sum_from_cluster[k_num] + 1
    print(f" sum of words by cluster is : {sum_from_cluster}")
    total_num_from_G = sum(sum_from_cluster)
    for i in range(len(sum_from_cluster)):
        distribution_of_cluster[i] = sum_from_cluster[i]/total_num_from_G
    print(f" distribution of words by cluster is : {distribution_of_cluster}")
    return sum_from_cluster

Group_distribution_dendogram_A = Group_distribution_dendogram(GroupA,11)

Entropy_All_G_Dendogram(11)

# =============================================================================
#                           K-MEDIOD with cosine-similarity
# =============================================================================
''' X(sparse metrix) - not normalized '''
# =============================================================================
# finding the optimal K
# =============================================================================

sil = []
kmax = 20
'silhouette_score'
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmedian = cluster.KMedoids(n_clusters = k, metric='cosine',
                           init='heuristic',
                           max_iter=300, random_state=123).fit(X)
  labels = kmedian.labels_
  sil.append(silhouette_score(X, labels, metric = 'cosine'))
  print(f"the shill score for {k} clusters is {sil[k-2]}")

'Elbow Method for KMedoids '
wcss=[]
for i in range(2,25,1):
    print(f" i = {i} ")
    kmedian = cluster.KMedoids(n_clusters=i, metric='cosine',
                           init='heuristic',
                           max_iter=300, random_state=123)
    kmedian.fit_predict(X)
    wcss.append(kmedian.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(2,25,1),wcss)
plt.title('The Elbow Method for KMedoids ')
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

'training K-Mediods'
kmedian = cluster.KMedoids(n_clusters=10, metric='cosine',
                           init='heuristic',
                           max_iter=300, random_state=123)
kmedian.fit(X) # get sparse metrix
labels_kmedian = kmedian.labels_
centroids_kmedian  = kmedian.cluster_centers_

tokens_labels_mediods = pd.DataFrame(columns = ["token","cluster"])
for idx,l in enumerate(kmedian.labels_):
    #print(l,vocabulary[idx])
    tokens_labels_mediods.loc[len(tokens_labels_mediods.index)] = [vocabulary[idx],l]

cluster_distrebution_kmedian10= cluster_distrebution(10,tokens_labels_mediods)

Ax_x = [x for x in range(10)]
plt.bar(Ax_x,cluster_distrebution_kmedian10)
plt.show()

avg_dist_from_prim_centroid = Group_distribution(GroupA,10,labels_kmedian,centroids_kmedian)


''' X(sparse metrix) -  normalized '''
# =============================================================================
# finding the optimal K
# =============================================================================

sil = []
kmax = 20
'silhouette_score'
# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmedian = cluster.KMedoids(n_clusters = k, metric='cosine',
                           init='heuristic',
                           max_iter=300, random_state=123).fit(X_Norm)
  labels = kmedian.labels_
  sil.append(silhouette_score(X_Norm, labels, metric = 'cosine'))
  print(f"the shill score for {k} clusters is {sil[k-2]}")

'Elbow Method for KMedoids '
wcss=[]
for i in range(2,25,1):
    print(f" i = {i} ")
    kmedian = cluster.KMedoids(n_clusters=i, metric='cosine',
                           init='heuristic',
                           max_iter=300, random_state=123)
    kmedian.fit_predict(X_Norm)
    wcss.append(kmedian.inertia_)
    
import matplotlib.pyplot as plt
plt.plot(range(2,25,1),wcss)
plt.title('The Elbow Method for KMedoids ')
plt.xlabel("No of Clusters")
plt.ylabel("WCSS")
plt.show()

'training K-Mediods'
kmedian = cluster.KMedoids(n_clusters=10, metric='cosine',
                           init='heuristic',
                           max_iter=300, random_state=123)
kmedian.fit(X_Norm) # get sparse metrix
labels_kmedian = kmedian.labels_
centroids_kmedian  = kmedian.cluster_centers_

tokens_labels_mediods = pd.DataFrame(columns = ["token","cluster"])
for idx,l in enumerate(kmedian.labels_):
    #print(l,vocabulary[idx])
    tokens_labels_mediods.loc[len(tokens_labels_mediods.index)] = [vocabulary[idx],l]

cluster_distrebution_kmedian10= cluster_distrebution(10,tokens_labels_mediods)

Ax_x = [x for x in range(10)]
plt.bar(Ax_x,cluster_distrebution_kmedian10)
plt.show()

avg_dist_from_prim_centroid = Group_distribution(GroupA,10,labels_kmedian,centroids_kmedian)


