
from read_data_from_gc import *
import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse
import h_transport_materials as htm
from labellines import labelLines

def plot_data(filename, t_start, title, vlines = False):
    '''
    A function to plot data received from the GC. 

    filename - a string of the path of the filename that is being analyzed
    t_start - the start of the data acquisition
    title - the title of th eplot
    vlines - a set of tuples (time, name) that represent the time of the event and the name of the event

    use semicolon after calling the function or else it plots twice
    
    '''
    data = read_data(filename)
    dates = [parse(date) for date in data["Injection_Acquired_Date"]]

    concentrations = data["Amount_ppm"].tolist()
    concentrations *= htm.ureg.ppm

    # Converting list of datetime.datetime objects to list of times in seconds since time_start
    time_start = parse(t_start)
    times_in_seconds = np.array(convert_dates_to_time(dates, time_start))
    times_in_hours = times_in_seconds / 3600
    fig = plt.figure(figsize = (8,6))
    plt.plot(times_in_hours, concentrations, marker="o")
    plt.fill_between(times_in_hours, concentrations, alpha=0.3)
    plt.ylabel(f"H2 concentration ({concentrations.units: ~P})")
    plt.xlabel("Time (h)")
    plt.grid(alpha=0.3)
    plt.xlim(left=0)

    if vlines:
        for line in vlines:
            event_time = parse(line[0])
            name = "$t_{"+line[1]+"}$"
            plt.axvline((event_time - time_start).total_seconds() / 3600, color="red", linestyle="--")
            plt.annotate(name, ((event_time - time_start).total_seconds() / 3600 * 1.02, -25.85), color="red", fontsize=14)
    plt.title(title)
    return times_in_hours
