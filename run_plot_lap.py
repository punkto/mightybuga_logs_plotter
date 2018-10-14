#! /user/bin/env python
import plot_lap
import sys


class Defaults:

    INPUT_FILE = ""  # csv
    OUT_FILE = ""  # png
    MIN_ROW = 5000  # 5000
    MAX_ROW = 17000  # 17000
    SHOW_COLUMNS = "1,2,3,4,5"  # blue, orange, green,red, violetx
    MAX_VALUES = "350,1000,1000,100,100"
    DIFF_COLUMNS = "2-3,4-5"
    WINDOW_SAMPLE = 10
    WINDOW_PLOTS = "1"
    HYST_PLOTS = "0.04-0.10,0.1-0.2,0.02-0.04"
    
    
if __name__ == "__main__":
    plot_lap.launch(sys.argv[1:], Defaults, __doc__)

    
