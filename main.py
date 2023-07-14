import Simulation
import Input_Data
import argparse
import json
from py_wake.examples.data.hornsrev1 import V80
import new_Larsen
import wake_model_new
import time


# main file to run simulation

if __name__ == '__main__':

    # DEFAULT Values, to be overriden with a config file
    # wind speed
    #u_inf = 12
    # wind direction (vector points in direction of wind)
    wind_theta = (
        0.0  # Wind flows toward East (i.e. a west wind, b/c it comes from the west)
    )
    # turbine hub height
    z_hub = 60
    # rotor diameter
    d_rot = 40
    # thickness of the rotor relative to its diameter
    rel_thickness = 0.05
    # number of turbines
    n_turbines = 3
    # roughness length
    z0 = 0.3
    T=0.12
    cw=0.9
    # farm size
    # TODO: Define as multiple of d_rot
    farm_size = 2000
    # number of images
    n_images = 125
    # resolution for plotting (number of points per direction)
    # TODO Needs to be multiple of d_rot for better results
    grid_resolution = 400  # means one point each 5 m based on size of 2000 m
    # grid_resolution = 800 # means one point each 2.5 m based on size of 2000 m
    # image resolution
    dpi = 100
    u_inf = 13.0
    # resulting image size in INCHES (works out to 4 x 4)
    img_size = grid_resolution / dpi
    # random seed
    # seed = 13     # preferred during dev/debugging so results are repeatable
    seed = 13  # preferred b/c they should default to being random
    # MIN/MAX limits for randomized cases
    n_turbines_range = [3, 15]
    u_inf_range = [5.0, 15.0]  # based on cut-in and rated speeds found in Google (typical cut=in is 2.5, but good sites have 6.0 avg speed
    wind_theta_range = [0, 0]
    # colormap for the images
    cmap = "gray"
    # limits of colormap: WIDER than u_inf_range
    # so that speeds below min(u_inf_range) can still
    # be represented (otherwise HIGH errors at low speeds)
    # there shouldn't be any speeds above u_inf_max,
    # at least on 2D simulations
    #cmap_range = [0.0, 15.0]

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Generate images of turbine layouts and corresponding wake flow fields",
        add_help=True,
    )

    # Add the arguments
    parser.add_argument('--deficit_model', help='The deficit model the is to be used'
                                          'pass [0] for new larsen model, and '
                                          '[1] for new wake model', type=int,
                        default=0)

    parser.add_argument(
        "--n_images",
        metavar="Ni",
        type=int,
        help="Number of images to generate",
        default=n_images,
    )
    parser.add_argument(
        "--n_turbines",
        metavar="Nt",
        type=int,
        help="Number of turbines in the wind farm layout",
        default=n_turbines,
    )

    parser.add_argument(
        "--u_inf",
        metavar="u_inf",
        type=int,
        help="Number of turbines in the wind farm layout",
        default=u_inf,
    )
    parser.add_argument(
        "--base_name",
        metavar="NAME",
        type=str,
        help="Common part of the image filenames. \nThe image number will be added as suffix to each image name\nDefaults to a timestamp",
        default=time.strftime("%Y%m%d-%H%M%S"),
    )

    parser.add_argument(
        "--wind_theta",
        metavar="Theta",
        type=float,
        help='Wind direction in [deg]. \nA NE wind, meaning the wind "comes" from the north-east, would have a direction of 180 + 45 = 225 degrees',
        default=0.0,
    )

    parser.add_argument(
        "--d_rot",
        metavar="Dt",
        type=float,
        help="Rotor diameter in [m]",
        default=d_rot,
    )

    parser.add_argument(
        "--t_rot",
        metavar="t",
        type=float,
        help="Relative rotor thickness. Defaults to d_rot/20",
        default=rel_thickness,
    )

    parser.add_argument(
        "--z_hub",
        metavar="Ht",
        type=float,
        help="Hub height in [m]",
        default=z_hub,
    )
    parser.add_argument(
        "--T",
        metavar="TI",
        type=float,
        help="Turbulance Intensity",
        default=T,
    )

    parser.add_argument(
        "--cw",
        metavar="Ct",
        type=float,
        help="darg Coe",
        default=cw,
    )

    parser.add_argument(
        "--z0",
        metavar="Z0",
        type=float,
        help="Roughness length of the terrain in [m]",
        default=z0,
    )

    parser.add_argument(
        "--farm_size",
        metavar="LENGTH",
        type=float,
        help="Side length in [m] of the wind farm terrain. The wind farm is assumed to be a square with the provided size",
        default=farm_size,
    )

    parser.add_argument(
        "--grid_resolution",
        metavar="NPOINTS",
        type=int,
        help="Resolution of meshgrid used to create the images. Note that this is different from the image resolution",
        default=grid_resolution,
    )

    parser.add_argument(
        "--dpi",
        metavar="DPI",
        type=int,
        help="Resolution (DPI) of created layout and flow field images",
        default=dpi,
    )

    parser.add_argument(
        "--cmap",
        metavar="CMAP",
        type=str,
        help="Colormap used to generate images. Defaults to 'jet'",
        default=cmap,
    )

    parser.add_argument(
        "--u_inf_range",
        metavar="X",
        type=float,
        nargs=2,
        help="Min and max values of wind speed used to generate cases. Defaults to 5.0 to 15 m/s, consistent with avg wind speeds in wind farm sites and typical cut-off of wind turbines",
        default=u_inf_range,
    )
    parser.add_argument(
        "--cmap_range",
        metavar="CMAP",
        type=float,
        nargs=2,
        help="Min and max values of wind speed used to define the colormap. Defaults to 0.0 to 16 m/s.",
        default=u_inf_range,
    )

    parser.add_argument(
        "--random_seed",
        metavar="SEED",
        type=int,
        help="Seed for the random number generator",
        default=seed,
    )

    parser.add_argument(
        "--save_config",
        action="store_true",
        help="Flag indicating whether to save all parameter values to a JSON file. Requires --config_file. Cannot be used with --load-config",
        default=False,
    )

    parser.add_argument(
        "--load_config",
        action="store_true",
        help="Flag indicating whether to load all parameter values from a JSON file. Requires --config_file. Cannot be used with --save-config",
        default=False,
    )

    parser.add_argument(
        "--config_file",
        metavar="FILE",
        type=str,
        help="Filename to either load or save configuration in JSON format. Defaults to a timestamp with a .cfg extension",
        default=None,
    )

    parser.add_argument(
        "--layout_file",
        metavar="FILE",
        type=str,
        help="Filename with turbine coordinates. Must be a pickle file containing a list of numpy arrays, each array of size [nt x 2])",
        #default='C:/Users/saeed/Code/DecodeCNN/Error/Larsen/larsen1/5/test_layouts.pkl',
        default='D:/Research/CFD_data/CFD_data/CFD_128_layouts-20230329T174354Z/test_layouts.pkl',
        #default=None,

    )

    parser.add_argument(
        "--randomize_all",
        action="store_true",
        help="Flag indicating whether to randomize the number of turbines, wind speed and wind direction. Hub height is not randomized because this is a 2D model. \n Note that this overrides any conflicting command-line options",
        default=False,
    )

    parser.add_argument(
        "--randomize_nt",
        action="store_true",
        help="Flag indicating whether to randomize the number of turbines",
        default=False,
    )

    parser.add_argument(
        "--randomize_u",
        action="store_true",
        help="Flag indicating whether to randomize the wind speed",
        default=True,
    )

    parser.add_argument(
        "--randomize_theta",
        action="store_true",
        help="Flag indicating whether to randomize the wind direction",
        default=False,
    )

    config = vars(parser.parse_args())

    f = open('config.json')

    if config['deficit_model'] == 0:
        # run the new larsen model
        new_Larsen.main(config)
        print('hi')

    else:
        # run the wake model new
        wake_model_new.main(config)

    # data1 = Input_Data.InputData('input_data.csv', 'wind_speeds')
    # windTurbines = V80()

    # simulation = Simulation.Simulation(data1, windTurbines)

    # simulation.run(data1)


















