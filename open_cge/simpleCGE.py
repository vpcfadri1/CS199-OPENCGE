# import packages
import numpy as np
import pandas
from pandas import Series, DataFrame
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms


def cge_system(pvec, args):
    """
    This function solves the system of equations that represents the
    CGE model.

    Args:
        pvec (Numpy array): Vector of prices
        args (tuple): Tuple of arguments for equations

    Returns:
        p_error (Numpy array): Errors from CGE equations
    """
    (p, d, ind, h, Z, Q, pd, Ff, er) = args

    py = pvec[0 : len(ind)]
    pf = pvec[len(ind) : len(ind) + len(h)]
    py = Series(py, index=list(ind))
    pf = Series(pf, index=list(h))
    
    # intermediate values
    Y = firms.eqY(p.ay, Z)
    F = hh.eqF(p.beta, py, Y, pf)


    # errors
    pf_error = agg.eqpf(F, d.Ff0) #6.24
    py_error = firms.eqpy(p.b, F, p.beta, Y) # 6.1 

    pf_error = DataFrame(pf_error)
    pf_error = pf_error.T
    pf_error = DataFrame(pf_error, columns=list(h))
    pf_error = pf_error.iloc[0]

    py_error = py_error.values
    pf_error = pf_error.values
    p_error = np.append(py_error, pf_error)

    return p_error

# Removed Equations
# Kk = agg.eqKk(pf, Ff, R, p.lam, pq)  (no capital stock)
# Kf = agg.eqKf(Kk, Kd) (no capital stock)
# Fsh = firms.eqFsh(R, Kf, er) (no repatriated profits)
# Trf = gov.eqTrf(p.tautr, pf, Ff) no transfer rate
# pk_error = agg.eqpk(F, Kk, d.Kk0, d.Ff0) (no capital stock)