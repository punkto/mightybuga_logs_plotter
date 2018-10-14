#! /user/bin/env python
import basic_plot
import sys


class Defaults:

    INPUT_FILE = ""  # csv
    OUT_FILE = ""  # png
    MIN_ROW = 5000  # 5000
    MAX_ROW = 17000  # 17000
    SHOW_COLUMNS = "1,6"  # blue, orange, green,red, violetx
    MAX_VALUES = "1,1"

    
if __name__ == "__main__":
    basic_plot.launch(sys.argv[1:], Defaults, __doc__)

    
