import numpy as np
def find_best_threshold(features,y,w):
    threshold = 0
    final_error = 9999
    final_polarity=1
    polarity = 1
    for feature in features:
        predictions = polarity * np.where(features <  feature, -1, 1)
        misclassified = predictions != y
        error = np.sum(w*[misclassified])
        if error >0.5:
            error = 1-error
            polarity *= -1
        if np.round(error,4) < final_error:
            final_error = error
            threshold = feature
            final_polarity = polarity       

    return threshold,final_error,final_polarity


def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []
    n_feature = 0
    alpha = 0
    for clf in range(n_clf):  
        threshold,error,polarity = find_best_threshold(X[:,n_feature],y,w)
        alpha = np.log( (1-error)/(error+1e-10) ) * 0.5
        hx = polarity* np.where( X[:,n_feature] < threshold,-1,1 )
        hx = np.array(hx)
        w = w*np.exp(-alpha*hx*y)/np.sum( w*np.exp(-alpha*hx*y) )
        result=[{'polarity':polarity,'threshold':threshold.tolist(),'feature_index':n_feature,'alpha':alpha.tolist() }]
        clfs += result

    return clfs




# X = np.array([[8, 7], [3, 4], [5, 9], [4, 0], [1, 0], [0, 7], [3, 8], [4, 2], [6, 8], [0, 2]])
# y = np.array([1, -1, 1, -1, 1, -1, -1, -1, 1, 1]) 
# n_clf = 2 
# clfs = adaboost_fit(X, y, n_clf) 
# print(clfs)


print(adaboost_fit(
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]]),
    y = np.array([1, 1, -1, -1]),
    n_clf = 3
    ))

