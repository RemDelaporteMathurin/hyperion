import numpy as np
import matplotlib.pyplot as plt
from dateutil.parser import parse


def find_line_number(file_path, search_string):
    """
    Find the line number where the search_string appears.

    :param file_path: Path to the file to be searched.
    :param search_string: The string to search for in the file.
    :return: The line number where the search_string is found. Returns None if not found.
    """
    with open(file_path, "r") as file:
        for line_number, line in enumerate(file, 1):
            if search_string in line:
                return line_number
    return None


def get_nb_lines(file_path):
    """
    Get the number of lines in a file.

    :param file_path: Path to the file to be searched.
    :return: The number of lines in the file.
    """
    with open(file_path, "r") as file:
        return len(file.readlines())


def read_data(filename):
    """
    Read data from a CSV file.

    :param filename: The name of the file to be read.
    :return: A numpy structured array with the data.
    """
    nb_lines_header = find_line_number(filename, "Sample statistics:") + 1
    print(f"Located 'Sample statistics:' at line {nb_lines_header}")

    nb_lines_footer = get_nb_lines(filename) - find_line_number(
        filename, "Acquisition method:"
    )
    data = np.genfromtxt(
        filename,
        delimiter=",",
        skip_header=nb_lines_header,
        skip_footer=nb_lines_footer,
        names=True,
        dtype=None,
        comments=None,
        encoding=None,
    )

    data = data[np.where(data["Injection_Acquired_Date"] != "")]
    return data


def convert_dates_to_time(dates, time_start):
    """
    Convert a list of datetime.datetime objects to a list of times in seconds since time_start.

    :param dates: List of datetime.datetime objects.
    :param time_start: The datetime.datetime object to use as the reference time.
    :return: List of times in seconds since time_start.
    """
    return [(dt - time_start).total_seconds() for dt in dates]


filename = "mid_dry_run_report.csv"

data = read_data(filename)
dates = [parse(date) for date in data["Injection_Acquired_Date"]]

print(data.dtype.names)
print(data["Injection_Acquired_Date"])
print(data["Amount_ppm"])

time_start = dates[0]
# Converting list of datetime.datetime objects to list of times in seconds since time_start
times_in_seconds = np.array(convert_dates_to_time(dates, time_start))
times_in_hours = times_in_seconds / 3600
plt.plot(times_in_hours, data["Amount_ppm"], marker="o")
plt.ylabel("Concentration (ppm)")
plt.ylim(bottom=0)
plt.show()
