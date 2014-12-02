import numpy as np

def needleman_wunsh(A,B,S,d):
    #initialization
    n = len(A)+1
    m = len(B)+1
    F=np.zeros((n,m))
    for i in range(n):
        F[i,0] = d*i
    for j in range(m):
        F[0,j] = d*j
    #dynamic programming
    for i in range(1,n):
        for j in range(1,m):
            match = F[i-1,j-1] + S[i-1,j-1]
            delete = F[i-1, j] + d
            insert = F[i, j-1] + d
            F[i,j] = max([match, insert, delete])
    return F


def is_approx(first, second, tolerance = 0.0001):
    if ( first + tolerance ) > second and ( first - tolerance ) < second:
        return True
    else :
        return False


def trace_back(A,B,S,d,F):
    AlignmentA = ""
    AlignmentB = ""
    i = len(A)
    j = len(B)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and is_approx(F[i,j] , F[i-1,j-1] + S[i-1, j-1]) :
            AlignmentA = A[i-1] + AlignmentA
            AlignmentB = B[j-1] + AlignmentB
            i = i - 1
            j = j - 1
        elif i > 0 and is_approx(F[i,j] , F[i-1,j] + d):
            AlignmentA = A[i-1] + AlignmentA
            AlignmentB = "-" + AlignmentB
            i = i - 1
        elif j > 0 and is_approx(F[i,j] , F[i,j-1] + d):
            AlignmentA = "-" + AlignmentA
            AlignmentB = B[j-1] + AlignmentB
            j = j - 1
    return AlignmentA,AlignmentB