import csv
import numpy as np


class InputData():
    """This class takes a file and extracts the data point for the system

    Default wind direction is 270 (west)hg
    """

    def __init__(self, filename, directory_name):

        self.directory = directory_name

        self.file = open(filename)
        self.reader = csv.reader(self.file)

        self.initial_positions = np.array(get_data_float(self.reader)).T

        temp_data = get_data_4_float(self.reader)
        self.x_axis, self.y_axis, self.z_axis, self.jump = temp_data[0][0], \
                                                           temp_data[0][1], \
                                                           temp_data[0][2], \
                                                           temp_data[0][3]

        temp_data = get_data_3_float(self.reader)
        self.ti, self.hub_height, self.t_diameter = temp_data[0][0], \
                                                    temp_data[0][1], \
                                                    temp_data[0][2]

        temp_data = get_data(self.reader)

        if temp_data[0][0] == 'True':
            self.w_check = True
            self.ws = None
            temp_data = np.array(get_data_3_float(self.reader)).T
            self.p_wd, self.a, self.k = temp_data[0], temp_data[1], temp_data[2]

        else:
            self.w_check = False
            self.ws = float(temp_data[0][1])
            self.wd = 270
            temp_data = get_data_3(self.reader)
            self.p_wd, self.a, self.k = temp_data[0][0], temp_data[0][1], \
                                        temp_data[0][2]

        self.file.close()


def get_data(reader):
    data = []
    for line in reader:
        if not line:
            break
        elif not line[0].startswith('#'):
            data.append([line[0], line[1]])

    return data


def get_data_float(reader):
    data = []
    for line in reader:
        if not line:
            break
        elif not line[0].startswith('#'):
            data.append([float(line[0]), float(line[1])])

    return data


def get_data_3(reader):
    data = []
    for line in reader:
        if not line:
            break
        elif not line[0].startswith('#'):
            data.append([line[0], line[1], line[2]])

    return data


def get_data_3_float(reader):
    data = []
    for line in reader:
        if not line:
            break
        if not line[0].startswith('#'):
            data.append([float(line[0]), float(line[1]), float(line[2])])

    return data


def get_data_4_float(reader):
    data = []
    for line in reader:
        if not line:
            break
        if not line[0].startswith('#'):
            data.append([float(line[0]), float(line[1]), float(line[2]),
                         float(line[3])])

    return data
