import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.

    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    if not corpus:
        return None
    if not query:
        return None
    N = len(corpus)
    for i in range(N):
        if len(corpus[i]) == 0:
            return None

    tf_idf = []
    d = np.array( [ len(corpus[i]) for i in range(N) ] )
    t = np.array( [ [ sum([np.where(corpus[i][j] == query[term],1,0)\
                     for j in range(len(corpus[i]))]) for i in range(N) ] \
                     for term in range(len(query))] )

    for term in range(len(query)):
        count = 0
        for i in range(N):
            if t[term][i] != 0:
                count+=1
        idf = np.log((N + 1)/(count + 1)) + 1 
        term_frequnency = t[term]/d 
        tf_idf.append(term_frequnency * idf) 

    return np.round(np.transpose(tf_idf),5)


# corpus = [ ["the", "cat", "sat", "on", "the", "mat"], 
#            ["the", "dog", "chased", "the", "cat"], 
#            ["the", "bird", "flew", "over", "the", "mat"]
#          ]
# query = ["cat", "mat"] 
# print(compute_tf_idf(corpus, query))

corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "chased", "the", "cat"],
    ["the", "bird", "flew", "over", "the", "mat"]
]
query = ["cat"]
print(compute_tf_idf(corpus, query))
