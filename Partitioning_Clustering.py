import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


quarantine_data = pd.read_csv('quarantine_data.csv')
hci_data = pd.read_csv('hci_data.csv')


quarantine_features = quarantine_data[['number_of_quarantined', 'number_of_cases', 'number_of_recovered', 'number_of_deaths']]
hci_features = hci_data[['HCI_index', 'education_index', 'health_index']]

scaler = StandardScaler()
quarantine_features_scaled = scaler.fit_transform(quarantine_features)
hci_features_scaled = scaler.fit_transform(hci_features)


def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    score = silhouette_score(data, clusters)
    return clusters, score


quarantine_kmeans_clusters, quarantine_kmeans_score = kmeans_clustering(quarantine_features_scaled, 3)
hci_kmeans_clusters, hci_kmeans_score = kmeans_clustering(hci_features_scaled, 3)

print(f'Quarantine K-Means Silhouette Score: {quarantine_kmeans_score}')
print(f'HCI K-Means Silhouette Score: {hci_kmeans_score}')
