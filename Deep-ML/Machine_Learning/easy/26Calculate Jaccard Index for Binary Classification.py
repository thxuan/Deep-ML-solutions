import numpy as np 
def jaccard_index(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  #inte = np.array( [y_true[i] == 1 and y_pred[i] == 1 for i in range(len(y_true))] )
  #uni  = np.array( [y_true[i] == 1 or y_pred[i] == 1 for i in range(len(y_true))] )
  inte = (y_true == 1) & (y_pred == 1)
  uni= (y_true == 1) | (y_pred == 1)
  result = sum(inte) / sum(uni)
  return( round(result,3) )


print(jaccard_index(
  y_true = [1,0,1,1,0,1],
  y_pred = [1,0,1,0,0,1]
  ))