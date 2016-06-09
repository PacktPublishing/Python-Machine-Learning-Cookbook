import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def convert_data_to_timeseries(input_file, column, verbose=False):
    # Load the input file
    data = np.loadtxt(input_file, delimiter=',')

    # Extract the start and end dates
    start_date = str(int(data[0,0])) + '-' + str(int(data[0,1]))
    end_date = str(int(data[-1,0] + 1)) + '-' + str(int(data[-1,1] % 12 + 1))

    if verbose:
        print "\nStart date =", start_date
        print "End date =", end_date

    # Create a date sequence with monthly intervals
    dates = pd.date_range(start_date, end_date, freq='M')

    # Convert the data into time series data
    data_timeseries = pd.Series(data[:,column], index=dates)

    if verbose:
        print "\nTime series data:\n", data_timeseries[:10]

    return data_timeseries

if __name__=='__main__':
    # Input file containing data
    input_file = 'data_timeseries.txt'

    # Load input data
    column_num = 2
    data_timeseries = convert_data_to_timeseries(input_file, column_num)

    # Plot the time series data
    data_timeseries.plot()
    plt.title('Input data')

    plt.show()
