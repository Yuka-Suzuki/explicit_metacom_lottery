import numpy as np

def Diversity(P, n_com):
    """
    species richness calculation
    P: S x N abundance matrix (S: the number of species, N: the number of patches)
    n_com: the number of patches
    """
    rP = P / P.sum(axis=0, dtype=float)
    # the above division gives nan for all elms in row i when sp.i went extinct from all communities. 
    # Replace this row with np.zeros(n_com) in that case.
    for i in range(len(rP)):
        if np.isnan(rP[i]).all() and P[i].sum() == 0:
            rP[i] = np.zeros(n_com)
    h_g = len(np.nonzero(rP.sum(axis=1))[0])
    w = 1/n_com
    h_a = w * len(np.nonzero(rP)[0])
    alpha = h_a
    gamma = h_g
    beta  = gamma / alpha
    return alpha, beta, gamma
