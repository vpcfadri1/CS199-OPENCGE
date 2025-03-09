import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os
import sys
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms, calibrate
from open_cge import simpleCGE as cge
from open_cge import execute as exec
from pprint import pprint


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# load SAM
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "NOBUHIRO_SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)

#
u = (
    "BRD",
    "MLK",
    "CAP",
    "LAB",
    "IDT",
    "TRF",
    "HOH",
    "GOV",
    "INV",
    "EXT",
)
ind = ("BRD", "MLK")
h = ("CAP", "LAB")

if __name__ == "__main__":
    exec.check_square()
    exec.row_col_equal()
    d = calibrate.model_data(sam, h, ind)
    p = calibrate.parameters(d, ind, sam)
    pprint(vars(d))


