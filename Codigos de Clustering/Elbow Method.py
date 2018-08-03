import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df_2014 = pd.read_csv("C:/Users/jorge/OneDrive/Escritorio/ProyectoJacobo/EnvipeLimpiax2/TPer_vic2_Clean2014_V3.csv", encoding = "ISO-8859-1")
df_2015 = pd.read_csv("C:/Users/jorge/OneDrive/Escritorio/ProyectoJacobo/EnvipeLimpiax2/TPer_vic2_Clean2015_V3.csv", encoding = "ISO-8859-1")
df_2016 = pd.read_csv("C:/Users/jorge/OneDrive/Escritorio/ProyectoJacobo/EnvipeLimpiax2/TPer_vic2_Clean2016_V3.csv", encoding = "ISO-8859-1")
df_2017 = pd.read_csv("C:/Users/jorge/OneDrive/Escritorio/ProyectoJacobo/EnvipeLimpiax2/TPer_vic2_Clean2017_V3.csv", encoding = "ISO-8859-1")

def calculaNClusters(df):
    distortions = []
    for i in range(1, 11):
         km = KMeans(n_clusters=i,
                     init='k-means++',
                     max_iter=300,
                     random_state=0)
         km.fit(df)
         distortions.append(km.inertia_)

    plt.plot(range(1,11), distortions, marker='*',c="black"  )
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

calculaNClusters(df_2017)
calculaNClusters(df_2016)
calculaNClusters(df_2015)
calculaNClusters(df_2014)
