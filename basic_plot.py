import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
from itertools import izip, count
import pandas as pd


def process_args(args, defaults, description):
    """ Handle input commands
    args - list of command line arguments
    default - default command line values
    description - a string to display at the top of the help message

    """
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input_file', dest="input_file",
                        type=str, default=defaults.INPUT_FILE,
                        help=("Input file " +
                              "defaults: %(default)s"))
    
    parser.add_argument('--out_file', dest="out_file",
                        type=str, default=defaults.OUT_FILE,
                        help=("Output file " +
                              "defaults: %(default)s"))

    parser.add_argument('--min_row', dest="min_row",
                        type=int, default=defaults.MIN_ROW,
                        help=("Min row " +
                              "defaults: %(default)s"))

    parser.add_argument('--max_row', dest="max_row",
                        type=int, default=defaults.MAX_ROW,
                        help=("Max row " +
                              "defaults: %(default)s"))

    parser.add_argument('--show_columns', dest="show_columns",
                        type=str, default=defaults.SHOW_COLUMNS,
                        help=("Columns to show in plot " +
                              "defaults: %(default)"))
    
    parser.add_argument('--max_values', dest="max_values",
                        type=str, default=defaults.MAX_VALUES,
                        help=("Max values of each column " +
                              "defaults: %(default)"))
    
    parameters = parser.parse_args(args)
    return parameters


def launch(args, defaults, description):
    """ Basic launch functionality """

    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.INFO)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger('basic')

    parameters = process_args(args, defaults, description)

    execute(parameters)

    
def execute(args):

    df = pd.read_csv(args.input_file, header=None,
                     delimiter=',')

    # Number of rows
    print ("Number of rows ", df.count())
    
    print ("Columns ", df.columns)

    orig_columns = len(df.columns)

    print ("Number of original columns ", orig_columns)
    
    columns_list = [int(col) for col in args.show_columns.split(",")]

    max_values_list = [float(col) for col in args.max_values.split(",")]

    df_copy = df.copy()
    
    for column, max_value in zip(columns_list, max_values_list):
        """
        max_value = np.max(df_copy[column][args.min_row:args.max_row])
        min_value = np.min(df_copy[column][args.min_row:args.max_row])

        df_copy[column] = ((df_copy[column][args.min_row:args.max_row] - min_value)/
        (max_value - min_value))
        """
        
        df_copy[column] = df_copy[column]/max_value

    for column in df_copy.columns:

        min_column = np.min(df_copy[column][args.min_row:args.max_row])
        max_column = np.max(df_copy[column][args.min_row:args.max_row])

        norm_column = ((df_copy[column][8400:9000] - min_column)/
                       (max_column - min_column))
        
        mean = np.mean(norm_column)
        std = np.std(norm_column)

        print ("Column: ", column, " Mean: ", mean, " Std: ", std)

    # plot the figures
    plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.plot(df_copy[columns_list][args.min_row:args.max_row])
    plt.show()
