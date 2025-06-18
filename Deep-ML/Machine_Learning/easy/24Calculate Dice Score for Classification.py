import numpy as np 
def dice_score(y_true, y_pred):
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)
  inte = sum( (y_true ==1 ) & (y_pred ==1 ) )
  a = sum(y_true ==1 ) + sum(y_pred ==1 )
  if a == 0:
    return 0.0
  else:
   return round(inte*2/a, 3)

y_true = [1,1,0,1,0,1]
y_pred = [1,1,0,0,0,1]
print( dice_score(y_true, y_pred ) )