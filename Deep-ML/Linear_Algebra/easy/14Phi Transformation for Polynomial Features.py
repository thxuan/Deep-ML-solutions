import numpy as np 
def phi_transform(data: list[float], degree: int) -> list[list[float]]:
  if degree <= 0:
    return []
  data = np.array(data)
  result = []
  new_data = []
  for x in data:
    for i in range(degree+1):
      new_data.append( np.power(x,i) )
    result.append(new_data)
    new_data = []
  return(np.array(result).tolist())
  
  
print(phi_transform(
  [1.0,2.0],
  2
  ))