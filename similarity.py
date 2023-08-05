

def similarity (vector1, vector2, metric = 'cosine'):
    if metric == 'cosine':
        norm1 = norm(vector1)
        norm2 = norm(vector2)
        dot = dot_product(vector1,vector2)
        return dot/(norm1 * norm2)
    
    if metric == 'euclidean':
        return eucl (vector1, vector2)

def norm (vector):
    sum_sqr = 0
    for i in vector:
        sum_sqr += i**2

    return sum_sqr ** 0.5

def dot_product (vector1, vector2):
    res = 0
    for i in range(len(vector1)):
        res += vector1[i] * vector2[i]

    return res

def eucl (vector1, vector2):
    sum_sqr = 0
    for i in range(len(vector1)):
        sum_sqr += (vector1[i] - vector2[i]) ** 2

    return sum_sqr ** 0.5