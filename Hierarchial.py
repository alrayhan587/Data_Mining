from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def hierarchical_clustering(data, n_clusters, method='ward'):
    Z = linkage(data, method=method)
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    score = silhouette_score(data, clusters)
    return clusters, score, Z


quarantine_hier_clusters, quarantine_hier_score, quarantine_Z = hierarchical_clustering(quarantine_features_scaled, 3)
hci_hier_clusters, hci_hier_score, hci_Z = hierarchical_clustering(hci_features_scaled, 3)

print(f'Quarantine Hierarchical Silhouette Score: {quarantine_hier_score}')
print(f'HCI Hierarchical Silhouette Score: {hci_hier_score}')


plt.figure(figsize=(10, 7))
plt.title("Quarantine Dendrogram")
dendrogram(quarantine_Z)
plt.show()

plt.figure(figsize=(10, 7))
plt.title("HCI Dendrogram")
dendrogram(hci_Z)
plt.show()
