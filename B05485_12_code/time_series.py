import numpy
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec
import matplotlib.cbook as cbook
from matplotlib.ticker import Formatter

# Define a class for formatting
class DataFormatter(Formatter):
    def __init__(self, dates, date_format='%Y-%m-%d'):
        self.dates = dates
        self.date_format = date_format

    # Extact the value at time t at position 'position'
    def __call__(self, t, position=0):
        index = int(round(t))
        if index >= len(self.dates) or index < 0:
            return ''

        return self.dates[index].strftime(self.date_format)

if __name__=='__main__':
    # CSV file containing the stock quotes 
    input_file = cbook.get_sample_data('aapl.csv', asfileobj=False)

    # Load csv file into numpy record array
    data = csv2rec(input_file)
    
    # Take a subset for plotting
    data = data[-70:]

    # Create the date formatter object
    formatter = DataFormatter(data.date)

    # X axis
    x_vals = numpy.arange(len(data))

    # Y axis values are the closing stock quotes
    y_vals = data.close 

    # Plot data
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(formatter)
    ax.plot(x_vals, y_vals, 'o-')
    fig.autofmt_xdate()
    plt.show()
