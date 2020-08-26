import numpy as np

def Diversity(P, n_com):
    """
    Gini-Simpson Diversity index calculation based on
    Jost et al (2010) Partitioning diversity for conservation analyses
    
    P: S x N abundance matrix (S: number of species, N: number of local patches)
    n_com: number of patches
    """
    rP = P / P.sum(axis=0, dtype=float) # normalization
    # the above division gives nan for all elements in row i when sp.i went extinct from all communities. 
    # Replace this row with np.zeros(n_com) in this case.
    for i in range(len(rP)):
        if np.isnan(rP[i]).all() and P[i].sum() == 0:
            rP[i] = np.zeros(n_com)
    w = 1./n_com                        # weights are equal for all communities
    h_g = 1 - np.power(w*rP.sum(axis=1),2).sum()
    h_a = w * ( 1 - np.power(rP,2).sum(axis=0) ).sum()
    gamma = 1. / (1-h_g)
    alpha = 1. / (1-h_a)
    beta  = gamma / alpha
    return alpha, beta, gamma
