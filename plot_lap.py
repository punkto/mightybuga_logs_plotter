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
    
    parser.add_argument('--diff_columns', dest="diff_columns",
                        type=str, default=defaults.DIFF_COLUMNS,
                        help=("Build extra columns with this difference"))

    parser.add_argument('--window_plots', dest="window_plots",
                        type=str, default=defaults.WINDOW_PLOTS,
                        help=("Columns to build a plot (diff graphs will be added)"))

    parser.add_argument('--window_sample', dest="window_sample",
                       type=str, default=defaults.WINDOW_SAMPLE,
                       help=("Obtain the moving average window of window plots" +
                             "defaults: %(default)"))

    parser.add_argument('--hyst_plots', dest="hyst_plots",
                        type=str, default=defaults.HYST_PLOTS,
                        help=("Build and hysteresis with two states"))

    
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

    # build diff columns
    diff_columns_list = args.diff_columns.split(",")
    
    for i in range(len(diff_columns_list)):
        diff_columns_list[i] = [int(new) for new
                                in diff_columns_list[i].split("-")]

    diff_names_list = []
    for diff_idx in diff_columns_list:
        
        name = "diff_{}_{}".format(diff_idx[0],
                                   diff_idx[1])
        
        df_copy[name]= pd.Series(df_copy[diff_idx[0]] -
                                 df_copy[diff_idx[1]],
                                 index=df_copy.index).values

        diff_names_list.append(name)

    print ("Number of columns (with extra diff) ", len(df_copy.columns), name)

    window_plots = [int(idx) for idx in args.window_plots.split(",")]
    window_plots = window_plots + diff_names_list

    print ("Window plots ", window_plots)

    # average in window
    window_names_list = []
    for wind_idx in window_plots:
        name = "win_{}".format(wind_idx)
        moving_avg = np.convolve(
            df_copy[wind_idx],
            np.ones((args.window_sample, ))/args.window_sample, mode='same')

        df_copy[name] = pd.Series(moving_avg,
                                  index=df_copy.index).values

        window_names_list.append(name)

        print ("LEN ", np.size(moving_avg))

    sorted_name_list = []
    for wind_idx in window_names_list:
        sort_name = "sort_{}".format(wind_idx)
        sorted_vals = np.sort(np.abs(df_copy[wind_idx]))
        df_copy[sort_name] = pd.Series(sorted_vals,
                                       index=df_copy.index).values

        print ("SORTED VALS ", sorted_vals[:20])
        
        sorted_name_list.append(sort_name)

    # build hysteresis
    hyst_values_list = args.hyst_plots.split(",")

    for i in range(len(hyst_values_list)):
        hyst_values_list[i] = [float(new) for new
                               in hyst_values_list[i].split("-")]

    hyst_name_list = []
    for idx, wind_idx, hyst_value in izip(count(),
                                          window_names_list,
                                          hyst_values_list) :

        current_state = 0
        hyst_name = "hyst_{}".format(wind_idx)

        value_list = []
        
        for x in np.abs(df_copy[wind_idx]).tolist():

            if current_state == 0:
                if x < hyst_value[0]:
                    current_state = 1

            elif current_state == 1:
                if x > hyst_value[1]:
                    current_state = 0

            value_list.append(current_state + idx * 0.1)

        df_copy[hyst_name] = pd.Series(np.array(value_list),
                                       index=df_copy.index).values
        
        hyst_name_list.append(hyst_name)
                    
    # some stats
    for column in df_copy.columns:

        min_column = np.min(df_copy[column][args.min_row:args.max_row])
        max_column = np.max(df_copy[column][args.min_row:args.max_row])

        norm_column = ((df_copy[column][8400:9000] - min_column)/
                       (max_column - min_column))
        
        mean = np.mean(norm_column)
        std = np.std(norm_column)

        print ("Column: ", column, " Mean: ", mean, " Std: ", std)

    # Check correlations
        
    # obtain the correlation between two curves
    corr_pos_enc = np.argmax(signal.correlate(
        df_copy[1][args.min_row:args.max_row],
        df_copy["diff_4_5"][args.min_row:args.max_row]))
    corr_enc_pos = np.argmax(signal.correlate(
        df_copy["diff_4_5"][args.min_row:args.max_row],
        df_copy[1][args.min_row:args.max_row]))

    print ("Corr pos_enc", corr_pos_enc)
    print ("Corr enc_pos", corr_enc_pos)

    A = fftpack.fft(
        df_copy["win_1"][args.min_row:args.max_row])
    B = fftpack.fft(
        df_copy["win_diff_4_5"][args.min_row:args.max_row])
    Ar = -A.conjugate()
    Br = -B.conjugate()
    print ("FFT POS_ENC ", np.argmax(np.abs(fftpack.ifft(Ar*B))))
    print ("FFT ENC_POS ", np.argmax(np.abs(fftpack.ifft(A*Br))))

    # plot the figures
    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.plot(df_copy[columns_list][args.min_row:args.max_row])

    plt.subplot(3, 1, 2)
    plt.plot(df_copy[window_plots][args.min_row:args.max_row])

    # plot the average window
    plt.subplot(3, 1, 3)
    plt.plot(df_copy[window_names_list][args.min_row:args.max_row])

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.plot(df_copy[sorted_name_list][args.min_row:args.max_row])

    # plot the average window (again)
    plt.subplot(3, 1, 2)
    plt.plot(df_copy[window_names_list][args.min_row:args.max_row])

    
    plt.subplot(3, 1, 3)
    plt.plot(df_copy[hyst_name_list + [1]][args.min_row:args.max_row])
    plt.show()

    

    





