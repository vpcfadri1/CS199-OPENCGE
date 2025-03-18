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


# load social accounting matrix
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, "PH_SAM.xlsx")
sam = pd.read_excel(sam_path, index_col=0, header=0)



if __name__ == "__main__":
    exec.check_square()
    exec.row_col_equal()
    exec.runner()

