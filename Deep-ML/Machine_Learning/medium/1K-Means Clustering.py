import numpy as np
def k_means_clustering(points: list[tuple[float, float]], k: int, initial_centroids: list[tuple[float, float]], max_iterations: int) -> list[tuple[float, float]]:
    # Your code here
    points = np.array(points)
    centroids = np.array(initial_centroids)

    for _ in range(max_iterations):
        distances = np.array([np.sqrt((centroid - points) ** 2).sum(axis=1) for centroid in centroids])
        # print(distances)
        assignments = np.argmin(distances, axis=0)
        # print(assignments)
        new_centroids = np.array([points[assignments == i].mean(axis = 0) if len(assignments == i) != 0 \
                        else centroids[i] for i in range(k) ])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    
    return np.round(centroids,4)




# import numpy as np
# def euclidean_distance(a, b):
#     return np.sqrt(((a - b) ** 2).sum(axis=1))
    
# def k_means_clustering(points, k, initial_centroids, max_iterations):
#     points = np.array(points)
#     centroids = np.array(initial_centroids)
    
#     for iteration in range(max_iterations):
#         # Assign points to the nearest centroid
#         distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
#         assignments = np.argmin(distances, axis=0)

#         new_centroids = np.array([points[assignments == i].mean(axis=0) if len(points[assignments == i]) > 0 \
#                                   else centroids[i] for i in range(k)])
        
#         # Check for convergence
#         if np.all(centroids == new_centroids):
#             break
#         centroids = new_centroids
#         centroids = np.round(centroids,4)
#     return [tuple(centroid) for centroid in centroids]


