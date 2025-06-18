from collections import Counter
def confusion_matrix(data):
  data = [tuple(item) for item in data]
  c = Counter(data)
  tp = c[1,1]
  fn = c[1,0]
  fp = c[0,1]
  tn = c[0,0]
  return([
    [tp,fn],
    [fp,tn]
    ])
  

print(confusion_matrix([
  [1,1], [1,0], [0,1], [0,0], [0,1]
  ]))