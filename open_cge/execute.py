import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms, calibrate
from open_cge import simpleCGE as cge

# load social accounting matrix
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "PH_SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)
# declare sets of variables
u = (
    "AGR", "MIN", "MAN", "ESW", "CON", "WRT", "TRS", "AFS", "INF", "FIN", 
    "REA", "PBS", "PAD", "EDU", "HHS", "OTH", "CAP", "LAB", "IDT", "TRF", 
    "HOH", "GOV", "INV", "EXT"
)

ind = (
    "AGR", "MIN", "MAN", "ESW", "CON", "WRT", "TRS", "AFS", "INF", "FIN", 
    "REA", "PBS", "PAD", "EDU", "HHS", "OTH"
)

h = ("CAP", "LAB")


def check_square():
    """
    this function tests whether the SAM is a square matrix.
    """
    
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    sam_small.to_numpy(dtype=None, copy=True)

    if not sam_small.shape[0] == sam_small.shape[1]:
        raise ValueError(
            f"SAM is not square. It has {sam_small.shape[0]} rows and {sam_small.shape[0]} columns"
        )


def row_total():
    """
    This function tests whether the row sums
    of the SAM equal the expected value.
    """
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    row_sum = sam_small.sum(axis=0)
    row_sum = pd.Series(row_sum)
    return row_sum


def col_total():
    """
    This function tests whether column sums
    of the SAM equal the expected values.
    """
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    col_sum = sam_small.sum(axis=1)
    col_sum = pd.Series(col_sum)
    return col_sum


def row_col_equal():
    """
    This function tests whether row sums
    and column sums of the SAM are equal.
    """
    sam_small = sam
    sam_small = sam_small.drop("TOTAL")
    sam_small = sam_small.drop(columns=["TOTAL"])
    row_sum = sam_small.sum(axis=0)
    col_sum = sam_small.sum(axis=1)
    np.testing.assert_allclose(row_sum, col_sum)


def runner():
    """
    This function solves the CGE model
    """

    # solve cge_system
    dist = 10
    tpi_iter = 0
    tpi_max_iter = 1000
    tpi_tol = 1e-10
    xi = 0.1

    # pvec = pvec_init
    pvec = np.ones(len(ind) + len(h))

    # Load data and parameters classes
    d = calibrate.model_data(sam, h, ind) # correct
    p = calibrate.parameters(d, ind, sam) # correct
    
    # R = d.R0
    er = 1

    Zbar = d.Z0 
    Ffbar = d.Ff0
    # Kdbar = d.Kd0; N
    Qbar = d.Q0
    pdbar = pvec[0 : len(ind)]

    pm = firms.eqpm(er, d.pWm)

    while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
        tpi_iter += 1
        cge_args = [p, d, ind, h, Zbar, Qbar, pdbar, Ffbar, er]

        print("initial guess = ", pvec)
        results = opt.root(cge.cge_system, pvec, args=cge_args, method="lm", tol=1e-5)
        pprime = results.x
        pyprime = pprime[0 : len(ind)]
        pfprime = pprime[len(ind) : len(ind) + len(h)]
        pyprime = Series(pyprime, index=list(ind))
        pfprime = Series(pfprime, index=list(h))

        pvec = pprime
        #Nobuhiro Equations (not including 6.1, 6.17, 6.18, 6.20 6.21, 6.22, 6.24)
        F = hh.eqF(p.beta, pyprime, d.Y0, pfprime) #6.2
        Td = gov.eqTd(p.taud, pfprime, Ffbar) #6.6
        Xg = gov.eqXg(p.mu, d.XXg0) #6.9
        Sp = agg.eqSp(p.ssp, pfprime, Ffbar) #6.11
        
        pe = firms.eqpe(er, d.pWe)  #6.14
        pm = firms.eqpm(er, d.pWm)  #6.15
        pq = firms.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)  #6.23
        D = firms.eqD(p.gamma, p.deltad, p.eta, Qbar, pq, pdbar) #6.19
        pz = firms.eqpz(p.ay, p.ax, pyprime, pq) #6.5
        I = hh.eqI(pfprime, Ffbar, Sp, Td)
        Xp = hh.eqXp(p.alpha, I, pq) #6.13
        E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar) #6.21
        D = firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar) #6.22
        M = firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pq, pm, p.taum) #6.18
        Z = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D) #6.3
        epsilon = agg.eqbop(d.pWe, d.pWm, E, M, d.Sf0) #6.16 error function
        Tm = gov.eqTm(p.taum, pm, M) #6.8
        Tz = gov.eqTz(p.tauz, pz, Z) #6.7
        Sg = gov.eqSg(p.ssg, Td, Tz, Tm) #6.12
        Xv = firms.eqXv(p.lam, Sp, d.Sf0, er, Sg, pq)#6.10
        Y = firms.eqY(p.ay, Zbar)#6.4

        Qprime = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)#6.17
        pdprime = firms.eqpd(p.gamma, p.deltam, p.eta, Qprime, pq, D)#Unknown
        Zprime = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)#6.20

       
      
        Ffprime = d.Ff0
        dist = (((Zbar - Zprime) ** 2) ** (1 / 2)).sum()

        print("Distance at iteration ", tpi_iter, " is ", dist)
        pdbar = xi * pdprime + (1 - xi) * pdbar
        Zbar = xi * Zprime + (1 - xi) * Zbar 
        Qbar = xi * Qprime + (1 - xi) * Qbar
        Ffbar = xi * Ffprime + (1 - xi) * Ffbar

        Q = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)

    print("Model solved, Q = ", Q.to_markdown())
    print(f"6.1 err = {pyprime}")
    print(f"6.2 err = {F - hh.eqF(p.beta, pyprime, d.Y0, pfprime)}")
    print(f"6.3 err = {Z - firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)}")
    print(f"6.4 err = {Y - firms.eqY(p.ay, Zbar)}")
    print(f"6.5 err = {pz - firms.eqpz(p.ay, p.ax, pyprime, pq)}")
    print(f"6.6 err = {Td - gov.eqTd(p.taud, pfprime, Ffbar)}")
    print(f"6.7 err = {Tz - gov.eqTz(p.tauz, pz, Z)}")
    print(f"6.8 err = {Tm - gov.eqTm(p.taum, pm, M)}")
    print(f"6.9 err = {Xg - gov.eqXg(p.mu, d.XXg0)}")
    print(f"6.10 err = {Xv - firms.eqXv(p.lam, Sp, d.Sf0, er, Sg, pq)}")
    print(f"6.11 err = {Sp - agg.eqSp(p.ssp, pfprime, Ffbar)}")
    print(f"6.12 err = {Sg - gov.eqSg(p.ssg, Td, Tz, Tm)}")
    print(f"6.13 err = {Xp - hh.eqXp(p.alpha, I, pq)}")
    print(f"6.14 err = {pe - firms.eqpe(er, d.pWe)}")
    print(f"6.15 err = {pm - firms.eqpm(er, d.pWm)}")
    print(f"6.16 err = {agg.eqbop(d.pWe, d.pWm, E, M, d.Sf0)}")
    print(f"6.17 err = {Qprime - firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)}")
    print(f"6.18 err = {M - firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pq, pm, p.taum)}")
    print(f"6.19 err = {D - firms.eqD(p.gamma, p.deltad, p.eta, Qbar, pq, pdbar)}")
    print(f"6.20 err = {Zprime - firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)}")
    print(f"6.21 err = {E - firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)}")
    print(f"6.22 err = {D - firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar)}")
    print(f"6.23 err = {pq - firms.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)}")
    print(f"6.24 err = {pfprime}")


    return Q
# Removed Equations
# Kk = agg.eqKk(pfprime, Ffbar, R, p.lam, pq)
# Trf = gov.eqTrf(p.tautr, pfprime, Ffbar)
# Kf = agg.eqKf(Kk, Kdbar)
# Fsh = firms.eqFsh(R, Kf, er)
# Ffprime["CAP"] = R * Kk * (p.lam * pq).sum() / pfprime.iloc[1]
# Kdbar = xi * Kdprime + (1 - xi) * Kdbar
# Kdprime = agg.eqKd(d.g, Sp, p.lam, pq)
