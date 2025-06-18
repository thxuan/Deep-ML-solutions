import numpy as np 
def cosine_similarity(v1, v2):
  v1 = np.array(v1)
  v2 = np.array(v2)
  r1 = v1.dot(v2)
  r2 = np.linalg.norm(v1) * np.linalg.norm(v2)
  return( round( r1/r2 , 3) )
  
  
print( 
  cosine_similarity(
  [1,2,3],
  [2,4,6])
  )