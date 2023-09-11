import json
import pickle
import random
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from tqdm import tqdm
from math import floor
from turbine_builder import Turbines
from py_wake_new_wake import MyJensen
from py_wake.site._site import UniformSite


'''
Controls whether a turbine is presented by scaled lines
or by a single pixel
'''
DRAW_TURBINES_AS_LINES = False


def power_curve(u):
    """
    Default, theoretical power curve
    """

    return 0.3 * np.power(u, 3)


def calculate_power_from_image(
        img_flow,
        img_layout,
        cmap_min,
        cmap_max,
        uinf,
        power_curve_function,
        d_rot,
        farm_size,
        grid_resolution,
        rel_thickness,
        n_turbines=None,
        turbine_pixel_coordinates=None,
        invert_image = False,
        npy_file=False,
        normal=False
):

    # if we are given filenames, read in the images

    if type(img_flow) is str:

        img_name = img_flow
        img_flow = Image.open(img_flow)
        img_flow = img_flow.convert("L")
        img_flow = np.array(img_flow)


    else:
        img_flow=img_flow

    if type(img_layout) is str:
        lay_name = img_layout
        img_layout = Image.open(img_layout)




    img_layout = img_layout.convert("1")
    # We now convert them to numpy arrays

    img_layout = np.asarray(img_layout)






    layout_mask = (img_layout == False)


    # velocities are right after the turbine, while the first
    # set of turbines should see u = 12
    layout_mask_shifted = np.zeros(layout_mask.shape,dtype=bool)
    layout_mask_shifted[:,:-1] = layout_mask[:,1:]
    layout_mask_shifted[:,-1] = layout_mask[:,-1]
    layout_mask_old = layout_mask
    layout_mask = layout_mask_shifted


    if not npy_file:

        if not invert_image:

            img_flow = cmap_min + (img_flow / 255) * (cmap_max - cmap_min)
        else:

            img_flow = cmap_min + ((255 - img_flow) / 255) * (cmap_max - cmap_min)

    # apply power curve to these speeds, using layout as mask
    if normal:

        img_flow=uinf*img_flow


    uturb = img_flow[layout_mask]

    img_power = power_curve_function(uturb)

    pix_per_m = grid_resolution / farm_size

    pix_per_turbine = max(floor(d_rot * pix_per_m), 1)
    #The thickness
    thick = max(floor(rel_thickness * d_rot * pix_per_m), 1)
    if DRAW_TURBINES_AS_LINES:
        turbine_pixel_size = pix_per_turbine * thick
    else:
        turbine_pixel_size = 1

    # totalize power AEP [kW-hr] considering number of turbines and pixels
    # this is equivalent to using the average power at each turbine site
    # with a ROUGH estimation of the number of turbines
    total_power = 8766 * img_power.sum() / turbine_pixel_size


    # if turbine coordinates are given then this has priority
    # so we overwrite the power estimate from above
    if turbine_pixel_coordinates is not None:
        mask = np.zeros(img_flow.shape)
        ixt = turbine_pixel_coordinates

        # We got turbine coordinates based on coordinate origin
        # at lower-left corner
        if DRAW_TURBINES_AS_LINES:
            for i in range(np.int_(np.floor(pix_per_turbine / 2))):
                for j in range(thick):
                    # note that x and y are reversed in image coordinates
                    # this is the same as doing it regularly,
                    # i.e. [i,j] instead of [j,i] and then transposing the array
                    mask[grid_resolution - ixt[:, 1] - i, ixt[:, 0]] = 1
                    mask[grid_resolution - ixt[:, 1] + i, ixt[:, 0]] = 1
                    mask[grid_resolution - ixt[:, 1] - i, ixt[:, 0] + j] = 1
                    mask[grid_resolution - ixt[:, 1] + i, ixt[:, 0] + j] = 1

            mask = np.bool_(mask)
            power = power_curve_function(img_flow[mask])

            total_power = 8766 * power.sum() / turbine_pixel_size
        else:

            XT=[]
            for i in range(ixt.shape[0]):
                x_t=[ixt[i,1],ixt[i,0]]
                XT.append(x_t)
            for j in XT:
                mask[j[0]][j[1]]=1

            mask = np.bool_(mask)

            power = power_curve_function(img_flow[mask])
            u_image=img_flow[mask].mean()

            u_eff=img_flow[mask]
            total_power = 8766 * power.sum()
            #print('u_eff=',u_eff)


    # re-calculate power if I have nt but not the pixel coordinates

    if n_turbines is not None and turbine_pixel_coordinates is None:
        total_power = 8766 * img_power.mean() * n_turbines



    return total_power,u_image,u_eff


def calculate_flow_field(
        x_turbines,
        farm_size,
        u_inf,
        wind_theta,
        z_hub,
        d_rot,
        z0,
        xplot,
        yplot,
        grid_resolution,
):
    """
    x_turbines: real, Cartesian coordinates of the turbines
    """

    xplot = xplot
    yplot = yplot


    name = "myTurbine"
    my_turbines = Turbines(name, d_rot, z_hub, u_inf).get_wind_turbines()
    my_site = UniformSite(ti=1/np.log(z_hub/z0), ws=u_inf)

    # getting the x, y - coordinates of the turbines
    xt = x_turbines
    if xt.ndim == 1:
        nt = xt.shape[0] / 2
        xt = np.reshape(xt, (nt, 2))
    else:
        nt = xt.shape[0]

    deficit_model = MyJensen(my_site, my_turbines[0])
    u_eff = deficit_model.calculate_new_wake_model(
        my_site, my_turbines[0], xt[0], xt[1], np.linspace(
            0,farm_size, grid_resolution), np.linspace(
            0,farm_size, grid_resolution), d_rot, u_inf)

    u_norm = u_eff[0] / u_inf

    return xplot, yplot, u_eff, u_norm


def create_turbine_polygon(xt, r_rot, wake_angle, wake_length, wind_theta):
    # master polygon: coordinate center is the turbine location
    # wake axis is the x axis
    # top of turbine rotor
    p0 = np.array([0, r_rot])
    # bottom of turbine rotor
    p3 = np.array([0, -r_rot])

    # unit vectors in the direction of wake boundaries
    rtop = np.array([np.cos(wake_angle), np.sin(wake_angle)])
    rbot = np.array([np.cos(wake_angle), -np.sin(wake_angle)])
    # wake length in the direction of the wake cone
    projected_wake_length = wake_length / np.cos(wake_angle)

    #
    p1 = p0 + rtop * projected_wake_length
    p2 = p3 + rbot * projected_wake_length

    # create, rotate, translate polygon based on wind direction
    t = Polygon([p0, p1, p2, p3])
    t = rotate(t, angle=wind_theta, origin=(0.0, 0.0), use_radians=True)
    t = translate(t, xt[0], xt[1])

    return t






##
def plot_flow_field(
        u,
        outputname,
        dpi,
        vmin,
        vmax,
        img_size,
        cmap,
):
    """
    u: Mesh-grid array based on real, cartesian coordinates.
    """
    fig = plt.figure()
    fig.set_size_inches(img_size * dpi, img_size * dpi)
    # NOTE the origin = lower parameter, so that this matches the Cartesian coordinates
    # Note also the transposition of the array, both are needed to match image coordinates
    # with Cartesian coordinates
    plt.imshow(
        u.T,
        interpolation="gaussian",
        aspect="equal",
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    plt.axis("off")
    plt.clim = [vmin, vmax]
    plt.colorbar()

    if outputname is not None:

        # this saves image data instead of rendering first before saving the image
        u=np.flip(u, axis=0)
        plt.imsave(
            outputname,
            arr=u,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
        )
        plt.close()






def plot_turbine_layout(
        xt,
        r_rot,
        wind_theta,
        rel_thickness,
        farm_size,
        outputname,
        dpi,
        img_size,
        grid_resolution,
        cmap,
        vmin=0.0,
        vmax=1.0,
):
    """
    xt: Real, Cartesian coordinates of the turbine layout
    """
    lay = np.zeros((grid_resolution, grid_resolution))
    # turbines are placed at lower left corner of each pixel
    xpix = np.int_(np.floor(xt / farm_size * grid_resolution))

    # turbine size in pixels (half size b/c we draw one half above/below)
    t_size = r_rot * grid_resolution / farm_size
    t_px_size = max(int(np.floor(t_size)), 1)
    # thickness in pixels

    t_px_thick = max(int(np.floor(rel_thickness * t_size)), 1)


    if DRAW_TURBINES_AS_LINES:

        for i in range(t_px_size):


            for j in range(t_px_thick):
                # note that x and y are reversed in image coordinates
                lay[xpix[:, 1] - i, xpix[:, 0]] = 1
                lay[xpix[:, 1] + i, xpix[:, 0]] = 1
                lay[xpix[:, 1] - i, xpix[:, 0] + j] = 1
                lay[xpix[:, 1] + i, xpix[:, 0] + j] = 1

    else:

        lay[xpix[:, 1], xpix[:, 0]] = 1



    fig = plt.figure()
    fig.set_size_inches(img_size * dpi, img_size * dpi)
    # NOTE the origin = lower parameter, so that this matches the Cartesian coordinates
    plt.imshow(
        lay,
        interpolation="gaussian",
        aspect="equal",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )

    plt.axis("off")
    plt.clim = [vmin, vmax]

    if outputname is not None:

        # this saves image data instead of rendering first before saving the image
        lay=np.flip(lay, axis=0)
        plt.imsave (
            outputname,
            arr=lay,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            dpi=dpi,
        )
        plt.close()
        return


def save_numy(u,outputname):
    np.save(outputname,u)

def calculate_power_from_wakes(
        ui,
        wake_name,
        layout_name,
        power_curve_function,
        d_rot,
        farm_size,
        grid_resolution,
        rel_thickness,
        turbine_pixel_coordinates
):

    P,_,_ = calculate_power_from_image(
        wake_name,
        layout_name,
        cmap_min=0.0,
        cmap_max=1.0,
        power_curve_function=power_curve,
        d_rot=d_rot,
        farm_size=farm_size,
        grid_resolution=grid_resolution,
        rel_thickness=rel_thickness,
        turbine_pixel_coordinates=turbine_pixel_coordinates,
        invert_image = True
    )
    P = P * ui**3

    return P


def generate_cases(
        xt_cases,
        u_inf_cases,
        wind_theta_cases,
        base_name,
        farm_size,
        z_hub,
        d_rot,
        z0,
        rel_thickness,
        dpi,
        cmap,
        vmin,
        vmax,
        img_size,
        grid_resolution,
):
    """
    Generate layout, speed  and flow field images for a set of cases.
    """

    # create a master index to store run information
    col_names = [
        "n_turbines",
        "wind_speed [m/s]",
        "wind_theta [rad]",
        "wake_image_file",
        "wind_resource_file",
        "layout_image_file",
        "turbine_coordinates [m]",
        "turbine_coordinates [px]",
        "jensen_power [kW-hr]",
        "image_power_given_xt [kW-hr]",
        "grid_resolution [px]",
        "power_error_given_xt [%]",
        "u_true",
        "u_image",
        "u_error",
        "res_error_npy",
        "u_image_npy",
        "u_error_npy",
        "res_error_nom",
        "u_image_nom",
        "u_error_nom"

    ]
    df = pd.DataFrame(columns=col_names)

    # main loop to generate images
    idx = np.arange(len(xt_cases)).tolist()
    for it, xt, ui, theta in tqdm(
            zip(idx, xt_cases, u_inf_cases, wind_theta_cases), total=len(idx)
    ):
        # generate image file name

        image_name = "{}_{:d}".format(base_name, it)

        flow_name = image_name + "_flow.png"
        flow_name1 = image_name + "_flow.npy"
        speed_name = image_name + "_wind_resource.png"
        wake_name = image_name + "_wake_effects.png"
        layout_name = image_name + "_layout.png"
        real_coords_name = image_name + "_layout.npy"
        pixel_coords_name = image_name + "_layout_pixels.npy"
        power_name = image_name + "_total_power.npy"

        # create grid for plotting and image creation
        # +1 and slicing are needed to match pixel coordinates with real
        # coordinates during image encoding/decoding.
        #INFO: Turbines are placed at lower left corner of pixels
        x_range = np.linspace(0.0, farm_size, grid_resolution+1)[:-1]
        y_range = np.linspace(0.0, farm_size, grid_resolution+1)[:-1]
        xplot, yplot = np.meshgrid(x_range, y_range)


        # calculate flow field
        x, y, u, u_norm = calculate_flow_field(
            x_turbines=xt,
            u_inf=ui,
            wind_theta=theta,
            farm_size=farm_size,
            z_hub=z_hub,
            d_rot=d_rot,
            z0=z0,
            xplot=xplot,
            yplot=yplot,
            grid_resolution=grid_resolution,
        )


        # generate images and save them
        # flow field with wakes

        save_numy(outputname=flow_name1,u=u)

        plot_flow_field(
            u,
            dpi=dpi,
            # so areas w/o wakes are white (intensity = 255)
            cmap=cmap,
            outputname=flow_name,
            vmin=vmin,
            vmax=vmax,
            img_size=img_size,
        )

        # wind resource (e.g., constant wind speed)
        plot_flow_field(
            ui * np.ones(u.shape),
            dpi=dpi,
            cmap=cmap,
            outputname=speed_name,
            vmin=vmin,
            vmax=vmax,
            img_size=img_size,
            )

        # wake effects only (u(x,y)/uinf)
        plot_flow_field(
            u_norm,
            dpi=dpi,
            outputname=wake_name,
            vmin=0.0,
            vmax=1.0,
            img_size=img_size,
            cmap=cmap
        )
        #turbine layout
        plot_turbine_layout(
            xt=xt,
            r_rot=d_rot / 2,
            wind_theta=theta,
            rel_thickness=rel_thickness,
            farm_size=farm_size,
            dpi=dpi,
            outputname=layout_name,
            img_size=img_size,
            grid_resolution=grid_resolution,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        # save turbine coordinates for each case
        np.save(real_coords_name, xt)
        # save pixel coordinates for each case
        # this helps to extract power from flow field images
        xt_pixels = np.int_(np.floor(xt / farm_size * grid_resolution))
        np.save(pixel_coords_name, xt_pixels)

        # calculate true power from Jensen's wake model

        xturb, yturb, uturb, unorm = calculate_flow_field(
            xt,
            farm_size=farm_size,
            u_inf=ui,
            wind_theta=wind_theta,
            z_hub=z_hub,
            d_rot=d_rot,
            z0=z0,
            xplot=xt[:, 0],
            yplot=xt[:, 1],
            grid_resolution=grid_resolution,
        )


        u_true=uturb.mean()
        #print("uturb=",uturb)
        Ptrue = power_curve(uturb)


        # AEP: Annual energy production, in kW-hr
        Ptrue = 8766.0 * Ptrue.sum()

        # save power values
        np.save(power_name, Ptrue)

        # # save power value estimated from the image
        # # # use turbine coordinates for better accuracy
        # ## set invert_image=True

        Pimg_xt,u_image,u_eff = calculate_power_from_image(
            flow_name,
            layout_name,
            cmap_min=vmin,
            cmap_max=vmax,
            uinf=ui,
            power_curve_function=power_curve,
            d_rot=d_rot,
            farm_size=farm_size,
            grid_resolution=grid_resolution,
            rel_thickness=rel_thickness,
            turbine_pixel_coordinates=xt_pixels,
            invert_image = False,
            npy_file=False,
            normal=False
        )
        #print('u_eff=',u_eff)


        Pimg_npy,u_image_npy,u_eff_npy = calculate_power_from_image(
            u,
            layout_name,
            cmap_min=vmin,
            cmap_max=vmax,
            uinf=ui,
            power_curve_function=power_curve,
            d_rot=d_rot,
            farm_size=farm_size,
            grid_resolution=grid_resolution,
            rel_thickness=rel_thickness,
            turbine_pixel_coordinates=xt_pixels,
            invert_image = False,
            npy_file=True,
            normal=False
        )

        Pimg_nom,u_image_nom,u_eff_nom = calculate_power_from_image(
            wake_name,
            layout_name,
            cmap_min=0,
            cmap_max=1,
            uinf=ui,
            power_curve_function=power_curve,
            d_rot=d_rot,
            farm_size=farm_size,
            grid_resolution=grid_resolution,
            rel_thickness=rel_thickness,
            turbine_pixel_coordinates=xt_pixels,
            invert_image = False,
            npy_file=False,
            normal=True
        )

        # # image intensities
        res_error_npy = np.abs(Ptrue - Pimg_npy) / Ptrue * 100
        u_error_npy=np.abs(u_true - u_image_npy) / u_true * 100

        res_error_xt = np.abs(Ptrue - Pimg_xt) / Ptrue * 100
        u_error=np.abs(u_true - u_image) / u_true * 100

        res_error_nom = np.abs(Ptrue - Pimg_nom) / Ptrue * 100
        u_error_nom=np.abs(u_true - u_image_nom) / u_true * 100
        # print('res_error_npy=',res_error_npy)
        # print('u_error_npy=',u_error_npy)
        #print('res_error_xt=',res_error_xt)
        #print('u_error=',u_error)
        #print('res_error_nom=',res_error_nom)
        #print('u_error_nom=',u_error_nom)



        # # create index of cases and files
        df= pd.concat([df, pd.DataFrame.from_records([
            dict(
                zip(
                    col_names,
                    [
                        xt.shape[0],
                        ui,
                        theta,
                        flow_name,
                        speed_name,
                        layout_name,
                        real_coords_name,
                        pixel_coords_name,
                        Ptrue,
                        Pimg_xt,
                        grid_resolution,
                        res_error_xt,
                        u_true,
                        u_image,
                        u_error,
                        res_error_npy,
                        u_image_npy,
                        u_error_npy,
                        res_error_nom,
                        u_image_nom,
                        u_error_nom

                    ],
                )
            )
        ])])

    # save ALL layouts as pickle file for re-loading via --layout_file CLI option
    pickle.dump(xt_cases, open(base_name + "_layouts.pkl", "wb"))

    # save case information as CSV and JSON for later use

    df.reset_index(drop=True, inplace=True)

    df.to_csv(base_name + "_cases.csv")


    df.to_json(base_name + "_cases.json")
    # report success, incl list of filenames
    print("\nLayout and flow field files have been created:\n")


    return



def main(params):
    """
    Calls generate_cases() with the command-line parameters provided.

    This "indirection" is needed to ensure we can generate cases using the CLI or the Python API (importing this module will setup default values for the parameters, just as the CLI does).

    `params` is a python dict, created from parsing CLI arguments
    """

    if params["config_file"] is None:
        config_file = params["base_name"] + ".cfg"
    else:
        config_file = params["config_file"]

    # First: If a config file is given, let's use that
    if params["load_config"]:
        with open(config_file, "r") as f:
            params_from_file = json.load(f)
    else:
        params_from_file = dict()
    # now update the default config with the given config
    # anything not given will keep it's default value
    params.update(params_from_file)


    # we modify params so that the saved config makes sense
    if params["save_config"]:
        p = params.copy()
        p["save_config"] = False
        p["config_file"] = config_file
        p["load_config"] = True
        with open(config_file, "w") as f:
            json.dump(p, f)

    # set random seed for both numpy and the system
    np.random.seed(params["random_seed"])
    random.seed(params["random_seed"])

    # If a file with layouts is given, let's use that
    # We do this FIRST because this file can override n_images
    if params["layout_file"] is not None:
        farm_size = params["farm_size"]
        # if a file of turbine coordinates is provided, read it in
        with open(params["layout_file"], "rb") as layout_file:
            xt_cases = pickle.load(layout_file)

        # check for inconsistency in the number of cases requested
        # if we have more layouts than n_images, let's truncate randomly
        # otherwise we fill-in what's missing by repeating, randomly
        if len(xt_cases) != params["n_images"]:
            xt_cases = random.choices(xt_cases, params["n_images"])
    else:
        xt_cases = None

    # Override inconsistent randomization settings
    randomize_nt = params["randomize_nt"]
    randomize_u = params["randomize_u"]
    randomize_theta = params["randomize_theta"]
    #
    if params["randomize_all"]:
        randomize_nt = True
        randomize_u = True
        randomize_theta = True

    if randomize_nt:
        # number of turbines
        nt_cases = np.random.randint(
            n_turbines_range[0], n_turbines_range[1], (params["n_images"])
        )
    else:
        nt_cases = [params["n_turbines"]] * params["n_images"]

    if randomize_u:
        # wind speeds
        u_inf_cases = (
                np.random.random((params["n_images"])) * (u_inf_range[1] - u_inf_range[0])
                + u_inf_range[0]
        )
    else:
        u_inf_cases = [params["u_inf"]] * params["n_images"]

    if randomize_theta:
        # wind directions
        wind_theta_cases = (
                np.random.random((params["n_images"]))
                * (wind_theta_range[1] - wind_theta_range[0])
                + wind_theta_range[0]
        )
    else:
        # expand the constant values to a list
        wind_theta_cases = [params["wind_theta"]] * params["n_images"]


    d_rot = params["d_rot"]
    grid_resolution = params["grid_resolution"]
    # image resolution
    dpi = params["dpi"]

    # img_size = (grid_resolution / dpi, grid_resolution / dpi)
    img_size = grid_resolution / dpi
    # No files with layouts were provided, so we create them randomly
    # Generate random cases through a centered Latin hypercube

    if xt_cases is None:
        xt_cases = list()
        farm_size = params["farm_size"]

        # Generate random layouts, but do it sequentially
        for icase, nti in enumerate(nt_cases):
            # create image grid in terms of pixel coordinates
            layout = np.ones((grid_resolution,grid_resolution))
            nrows, ncols = layout.shape

            # distance constraints
            m_per_pix = farm_size / grid_resolution
            # Minimum distance (dij) of 2D between turbines, expressed
            # as number of pixels
            dij = int(2 * d_rot / m_per_pix)
            # margin around the farm
            margin = 2*dij

            # create margin in the borders of the domain
            layout[:margin, :] = 0
            layout[-margin:, :] = 0
            layout[:, :margin] = 0
            layout[:, -margin:] = 0

            # pool of indices to available sites
            i, j = np.where(layout)

            # add turbines one at a time, respecting dij constraint
            xt_case = list()

            for k in range(0, nti):
                # pick a cell randomly from the available cell grid
                idx = np.random.randint(0,len(i))
                # extract its image/pixel coordinates
                xtipix = i[idx]

                ytipix = j[idx]

                # store turbine coordinates, in real values (meters)

                xt_case.append([xtipix * m_per_pix, ytipix * m_per_pix])

                # identify neighboring cells to remove (+/- dij)
                irem = xtipix + np.arange(-dij, dij+1)

                jrem = ytipix + np.arange(-dij, dij+1)

                # ensure indices are not out of bounds
                irem = irem[irem < nrows]
                jrem = jrem[jrem < ncols]
                irem = irem[irem >= 0]
                jrem = jrem[jrem >= 0]
                # mark these cells as unavailable

                for k in irem:
                    layout[k,jrem] = 0
                # redefine list of available cells
                i,j=np.where(layout)
                # If there are no more available cells, end this loop
                # and go to next case
                if len(i) == 0:
                    nt_cases[icase] = k+1
                    print("WARNING: Only {} turbines placed in case {}".format(k+1,icase))
                    break

            # convert to numpy array, add to list of cases
            xt_case = np.array(xt_case)
            xt_cases.append(xt_case)

    if np.any(np.array(params["cmap_range"]) < 0):
        vmin = None
        vmax = None
    else:
        vmin = params["cmap_range"][0]
        vmax = params["cmap_range"][1]

    generate_cases(
        xt_cases,
        u_inf_cases,
        wind_theta_cases,
        farm_size=params["farm_size"],
        z_hub=params['z_hub'],
        d_rot=d_rot,
        z0=params["z0"],
        rel_thickness=params["rel_thickness"],
        base_name=params["base_name"],
        dpi=params["dpi"],
        cmap=params["cmap"],
        vmin=params['cmap_range'][0],
        vmax=params['cmap_range'][1],
        img_size=img_size,
        grid_resolution= params["grid_resolution"],
    )
    return



if __name__ == "__main__":
    import argparse

    # DEFAULT Values, to be overriden with a config file
    # wind speed
    u_inf = 15
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
    n_turbines = 20
    # roughness length
    z0 = 0.3
    # farm size
    # TODO: Define as multiple of d_rot
    farm_size = 2000
    # number of images
    n_images = 128
    # resolution for plotting (number of points per direction)

    grid_resolution = 1000  # means one point each 5 m based on size of 2000 m
    # grid_resolution = 800 # means one point each 2.5 m based on size of 2000 m
    # image resolution
    dpi = 100
    # resulting image size in INCHES (works out to 4 x 4)
    img_size = grid_resolution / dpi
    # random seed
    # seed = 13     # preferred during dev/debugging so results are repeatable
    seed = 13  # preferred b/c they should default to being random
    # MIN/MAX limits for randomized cases
    n_turbines_range = [10, 30]
    u_inf_range = [5, 15.0]  # based on cut-in and rated speeds found in Google (typical cut=in is 2.5, but good sites have 6.0 avg speed
    wind_theta_range = [0, 0]
    # colormap for the images
    cmap = "gray"
    # limits of colormap: WIDER than u_inf_range
    # so that speeds below min(u_inf_range) can still
    # be represented (otherwise HIGH errors at low speeds)
    # there shouldn't be any speeds above u_inf_max,
    # at least on 2D simulations
    cmap_range = [0.0, 1.0]

    # Create the parser
    parser = argparse.ArgumentParser(
        description="Generate images of turbine layouts and corresponding wake flow fields",
        add_help=True,
    )

    # Add the arguments
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
        "--base_name",
        metavar="NAME",
        type=str,
        help="Common part of the image filenames. \nThe image number will be added as suffix to each image name\nDefaults to a timestamp",
        default='test',
    )

    parser.add_argument(
        "--u_inf",
        metavar="U",
        type=float,
        help="Wind speed in [m/s]",
        default=u_inf,
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
        help="Min and max values of wind speed used to define the colormap.",
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
        #default='D:/Research/CFD_data/CFD_data/CFD_128_layouts-20230329T174354Z/test_layouts.pkl',
        default=None
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
        default=False,
    )

    parser.add_argument(
        "--randomize_theta",
        action="store_true",
        help="Flag indicating whether to randomize the wind direction",
        default=False,
    )

    config = parser.parse_args()

    # using vars() converts the NameSpace object (config) into a dict
    main(vars(config))
