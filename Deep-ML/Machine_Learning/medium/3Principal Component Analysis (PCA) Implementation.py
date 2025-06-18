import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
  mean = np.mean(data, axis=0)
  sd = np.std(data, axis=0)
  standard_data = (data - mean)/sd
  co_matrix = np.cov(standard_data,rowvar=False,ddof=1)
  eigvalue,eigvector = np.linalg.eig(co_matrix)
  idx = np.argsort(eigvalue)[::-1]
  #eigvector = eigvector[:, idx]
  return(np.round(eigvector[:, idx[:k]],4))
  

print(pca(
  np.array([[2,1,3],[4,2,5],[6,3,7]]),
  2
  ))
