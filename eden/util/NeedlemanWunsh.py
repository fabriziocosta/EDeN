import numpy as np


def needleman_wunsh(sequence_a, sequence_b, score_matrix, gap_penalty):
    # initialization
    n = len(sequence_a) + 1
    m = len(sequence_b) + 1
    needleman_wunsh_matrix = np.zeros((n, m))
    for i in range(n):
        needleman_wunsh_matrix[i, 0] = gap_penalty * i
    for j in range(m):
        needleman_wunsh_matrix[0, j] = gap_penalty * j
    # dynamic programming
    for i in range(1, n):
        for j in range(1, m):
            match = needleman_wunsh_matrix[i - 1, j - 1] + score_matrix[i - 1, j - 1]
            delete = needleman_wunsh_matrix[i - 1, j] + gap_penalty
            insert = needleman_wunsh_matrix[i, j - 1] + gap_penalty
            needleman_wunsh_matrix[i, j] = max([match, insert, delete])
    return needleman_wunsh_matrix


def is_approx(first, second, tolerance=0.0001):
    """
    Test if two values are approximately the same.

    Keyword arguments:
    tolerance -- cutoff for allowed difference (default 0.0001)

    >>> is_approx(0.0000, 0.0001)
    False

    >>> is_approx(0.0000, 0.00009)
    True
    """
    if (first + tolerance) > second and (first - tolerance) < second:
        return True
    else:
        return False


def trace_back(sequence_a, sequence_b, score_matrix, gap_penalty, needleman_wunsh_matrix):
    alignment_a = ""
    alignment_b = ""
    i = len(sequence_a)
    j = len(sequence_b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and is_approx(needleman_wunsh_matrix[i, j], needleman_wunsh_matrix[i - 1, j - 1] + score_matrix[i - 1, j - 1]):
            alignment_a = sequence_a[i - 1] + alignment_a
            alignment_b = sequence_b[j - 1] + alignment_b
            i = i - 1
            j = j - 1
        elif i > 0 and is_approx(needleman_wunsh_matrix[i, j], needleman_wunsh_matrix[i - 1, j] + gap_penalty):
            alignment_a = sequence_a[i - 1] + alignment_a
            alignment_b = "-" + alignment_b
            i = i - 1
        elif j > 0 and is_approx(needleman_wunsh_matrix[i, j], needleman_wunsh_matrix[i, j - 1] + gap_penalty):
            alignment_a = "-" + alignment_a
            alignment_b = sequence_b[j - 1] + alignment_b
            j = j - 1
    return alignment_a, alignment_b
