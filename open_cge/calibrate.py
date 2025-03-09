# import packages
import numpy as np
from pandas import Series, DataFrame
from pprint import pprint

class model_data(object):
    """
    This function reads the SAM file and initializes variables using
    these data.

    Args:
        sam (DataFrame): DataFrame containing social and economic data
        h (list): List of factors of production
        ind (list): List of industries

    Returns:
        model_data (data class): Data used in the CGE model
    """
    print("local version")
    def __init__(self, sam, h, ind):
        # foreign saving Y
        self.Sf0 = DataFrame(sam, index=["INV"], columns=["EXT"])
        # private saving Y
        self.Sp0 = DataFrame(sam, index=["INV"], columns=["HOH"])
        # government saving/budget balance Y
        self.Sg0 = DataFrame(sam, index=["INV"], columns=["GOV"])
        # repatriation of profits N
        # self.Fsh0 = DataFrame(sam, index=["EXT"], columns=["HOH"])
        # capital stock N
        # self.Kk0 = 10510
        # foreign-owned capital stock N
        # self.Kf0 = 6414.35
        # domestically-owned capital stock N
        # self.Kd0 = self.Kk0 - self.Kf0

        # direct tax Y
        self.Td0 = DataFrame(sam, index=["GOV"], columns=["HOH"])
        # transfers (not found) N
        # self.Trf0 = DataFrame(sam, index=["HOH"], columns=["GOV"])
        # production tax
        self.Tz0 = DataFrame(sam, index=["IDT"], columns=list(ind))
        # import tariff correct
        self.Tm0 = DataFrame(sam, index=["TRF"], columns=list(ind))

        # the h-th factor input by the j-th firm correct Y
        self.F0 = DataFrame(sam, index=list(h), columns=list(ind))
        # factor endowment of the h-th factor correct Y
        self.Ff0 = self.F0.sum(axis=1)
        # composite factor (value added) Y
        self.Y0 = self.F0.sum(axis=0)
        # intermediate input Y
        self.X0 = DataFrame(sam, index=list(ind), columns=list(ind))
        # total intermediate input by the j-th sector Y
        self.Xx0 = self.X0.sum(axis=0)  # intermediate value, sum of X0
        # output of the i-th good Y
        self.Z0 = self.Y0 + self.Xx0
        pprint(self.Z0)
        # household consumption of the i-th good Y
        self.Xp0 = DataFrame(sam, index=list(ind), columns=["HOH"])
        # government consumption Y
        self.Xg0 = DataFrame(sam, index=list(ind), columns=["GOV"])
        # investment demand Y
        self.Xv0 = DataFrame(sam, index=list(ind), columns=["INV"])
        # exports Y
        self.E0 = DataFrame(sam, index=list(ind), columns=["EXT"])
        self.E0 = self.E0["EXT"]
        # imports Y
        self.M0 = DataFrame(sam, index=["EXT"], columns=list(ind))
        self.M0 = self.M0.loc["EXT"]

        # domestic supply/Armington composite good Y
        self.Q0 = (
            self.Xp0["HOH"] + self.Xg0["GOV"] + self.Xv0["INV"] + self.X0.sum(axis=1)
        )
        # production tax rate Y
        tauz = self.Tz0 / self.Z0
        # domestic tax rate Y
        self.D0 = (1 + tauz.loc["IDT"]) * self.Z0 - self.E0

        # Compute aggregates Intermediate Values

        # aggregate output
        self.Yy0 = self.Y0.sum()
        # aggregate demand
        self.XXp0 = self.Xp0.sum()
        # aggregate investment
        self.XXv0 = self.Xv0.sum()
        # aggregate government spending
        self.XXg0 = self.Xg0.sum()
        # aggregate imports
        self.Mm0 = self.M0.sum()
        # aggregate exports
        self.Ee0 = self.E0.sum()
        # aggregate gross domestic product
        # self.Gdp0 = self.XXp0 + self.XXv0 + self.XXg0 + self.Ee0 - self.Mm0
        # growth rate of capital stock
        # self.g = self.XXv0 / self.Kk0
        # interest rate
        # self.R0 = self.Ff0["CAP"] / self.Kk0

        # export price index Y
        self.pWe = np.ones(len(ind))
        self.pWe = Series(self.pWe, index=list(ind))
        # import price index Y
        self.pWm = np.ones(len(ind))
        self.pWm = Series(self.pWm, index=list(ind))


class parameters(object):
    """
    This function sets the values of parameters used in the model.

    Args:
        d (data class): Class of data for use in CGE model
        ind (list): List of industries

    Returns:
        parameters (parameters class): Class of parameters for use in
            CGE model.
    """

    def __init__(self, d, ind, sam):
        # foreign saving Y
        self.Sf0 = DataFrame(sam, index=["INV"], columns=["EXT"])
        # private saving Y
        self.Sp0 = DataFrame(sam, index=["INV"], columns=["HOH"])
        # government saving/budget balance Y
        self.Sg0 = DataFrame(sam, index=["INV"], columns=["GOV"])

        # elasticity of substitution; Y constant in GAMS sample
        self.sigma = Series(2, index=list(ind))
        # substitution elasticity parameter Y
        self.eta = (self.sigma - 1) / self.sigma

        # elasticity of transformation; Y constant in GAMS sample
        self.psi = Series(2, index=list(ind))
        # transformation elasticity parameter Y
        self.phi = (self.psi + 1) / self.psi

        # share parameter in utility function Y
        self.alpha = d.Xp0 / d.XXp0
        self.alpha = self.alpha["HOH"]  # ONLY HOH?
        # share parameter in production function Y
        self.beta = d.F0 / d.Y0
        temp = d.F0**self.beta
        # scale parameter in production function Y
        self.b = d.Y0 / temp.prod(axis=0)

        # intermediate input requirement coefficient Y
        self.ax = d.X0 / d.Z0
        # composite factor input requirement coefficient Y
        self.ay = d.Y0 / d.Z0
        self.mu = d.Xg0 / d.XXg0
        # government consumption share  N, not over total investments, but savings
        self.mu = self.mu["GOV"]  # only GOV?
        Sp0 = self.Sp0.values[0, 0]
        Sg0 = self.Sg0.values[0, 0]
        Sf0 = self.Sf0.values[0, 0]
        total_savings = Sp0 + Sg0 + Sf0
        self.lam = d.Xv0 / total_savings
        # investment demand share
        self.lam = self.lam["INV"]

        # production tax rate Y
        self.tauz = d.Tz0 / d.Z0
        self.tauz = self.tauz.loc["IDT"]
        # import tariff rate Y
        self.taum = d.Tm0 / d.M0
        self.taum = self.taum.loc["TRF"] 

        # share parameter in Armington function Y
        self.deltam = (
            (1 + self.taum)
            * d.M0 ** (1 - self.eta)
            / ((1 + self.taum) * d.M0 ** (1 - self.eta) + d.D0 ** (1 - self.eta))
        )
        self.deltad = d.D0 ** (1 - self.eta) / (
            (1 + self.taum) * d.M0 ** (1 - self.eta) + d.D0 ** (1 - self.eta)
        )

        # scale parameter in Armington function Y
        self.gamma = d.Q0 / (
            self.deltam * d.M0**self.eta + self.deltad * d.D0**self.eta
        ) ** (1 / self.eta)

        # share parameter in transformation function Y
        self.xie = d.E0 ** (1 - self.phi) / (
            d.E0 ** (1 - self.phi) + d.D0 ** (1 - self.phi)
        )
        self.xid = d.D0 ** (1 - self.phi) / (
            d.E0 ** (1 - self.phi) + d.D0 ** (1 - self.phi)
        )

        # scale parameter in transformation function Y
        self.theta = d.Z0 / (
            self.xie * d.E0**self.phi + self.xid * d.D0**self.phi
        ) ** (1 / self.phi)

        # average propensity to save N, no transfers or repatriation values
        self.ssp = (d.Sp0.values / (d.Ff0.sum()))[0]
        # direct tax rate Y
        self.taud = (d.Td0.values / d.Ff0.sum())[0]
        # transfer rate (Not Used)
        # self.tautr = (d.Trf0.values / d.Ff0["LAB"])[0]
        # government revenue, intermediate value
        self.ginc = d.Td0 + d.Tz0.sum() + d.Tm0.sum()
        self.ssg = d.Sg0 / self.ginc
        # household income N
        # self.hinc = d.Ff0.sum()
