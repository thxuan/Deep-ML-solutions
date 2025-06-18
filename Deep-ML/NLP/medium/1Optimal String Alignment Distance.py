def OSA(source: str, target: str) -> int:
    m = len(source)+1
    n = len(target)+1
    D = [ [0]*(n) for _ in range(m)]
    for i in range(n-1):
        D[0][i+1] = i+1
    for i in range(m-1):
        D[i+1][0] = i+1
    transpose=999
    cost=0
    
    for i in range(1,m):
        for j in range(1,n):
            transpose=999
            if source[i-1] == target[j-1]:
                cost=0
            else:
                cost=1       
            delete = D[i-1][j]+cost
            insert = D[i][j-1]+cost
            substitute = D[i-1][j-1]+cost     
            if i>=2 and j>=2:
                if (source[i-2] == target[j-1]) and (source[i-1] == target[j-2]):
                    transpose = D[i-2][j-2]+cost
            D[i][j] = min(delete,insert,substitute,transpose)

    return(D[m-1][n-1])

# source = "abc"
# target = "fcb"
# source = "butterfly"
# target = "dragonfly"
source = "caper" 
target = "acer" 
distance = OSA(source, target)
print(distance)