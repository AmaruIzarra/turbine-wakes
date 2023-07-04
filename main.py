import Simulation
import Input_Data
from py_wake.examples.data.hornsrev1 import V80


# main file to run simulation

if __name__ == '__main__':

    data = Input_Data.InputData('input_data.csv', 'wind_speeds')
    windTurbines = V80()

    simulation = Simulation.Simulation(data, windTurbines)

    simulation.run(data)


















