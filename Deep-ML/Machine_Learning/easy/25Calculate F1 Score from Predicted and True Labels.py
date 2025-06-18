import numpy as np
def calculate_f1_score(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  truep = sum( (y_true == 1) & (y_pred == 1) )
  falp = sum( (y_true == 0) & (y_pred == 1) )
  faln = sum( (y_true == 1) & (y_pred == 0) )
  if (truep + falp ) == 0 or (truep + faln) == 0:
    return 0.0
  else:
    p = truep/(truep + falp)
    r = truep/(truep + faln)
    f1 = 2*p*r/(p+r)
    return(round(f1,3))
  
  
print(calculate_f1_score(
  [1,0,1,1,0],
  [1,0,0,1,1]
  ))