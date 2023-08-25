import io
import json
import pickle
import random
import time

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import PIL.Image
import scipy.stats.qmc as qmc
from PIL import Image, ImageOps, ImageStat
from scipy.stats.qmc import LatinHypercube
from shapely.affinity import rotate, translate
from shapely.geometry import Point, Polygon
from tqdm import tqdm
from IPython import embed
from math import floor
import math

'''
Controls whether a turbine is presented by scaled lines
or by a single pixel
'''
DRAW_TURBINES_AS_LINES = True


def power_curve(u):
    """
    Default, theoretical power curve
    """
    #return 1/3 * np.power(u, 3)
    return 0.3 * np.power(u, 3)


def calculate_power_from_image(
        img_flow,
        img_layout,
        cmap_min,
        cmap_max,
        power_curve_function,
        d_rot,
        farm_size,
        grid_resolution,
        rel_thickness,
        n_turbines=None,
        turbine_pixel_coordinates=None,
        invert_image=False
):

    # if we are given filenames, read in the images
    if type(img_flow) is str:
        img_name = img_flow
        img_flow = Image.open(img_flow)
    if type(img_layout) is str:
        lay_name = img_layout
        img_layout = Image.open(img_layout)

    img_flow = img_flow.convert("L")

    img_layout = img_layout.convert("1")

    # We now convert them to numpy arrays
    img_flow = np.array(img_flow)
    img_layout = np.asarray(img_layout)

    layout_mask = (img_layout == False)


    # set of turbines should see u = 12
    layout_mask_shifted = np.zeros(layout_mask.shape,dtype=bool)
    layout_mask_shifted[:,:-1] = layout_mask[:,1:]
    layout_mask_shifted[:,-1] = layout_mask[:,-1]
    layout_mask_old = layout_mask
    layout_mask = layout_mask_shifted
    #embed()
    # s
    m = np.max(img_flow)
    if not invert_image:
        img_flow = cmap_min + (img_flow / m) * (cmap_max - cmap_min)
    else:
        img_flow = cmap_min + ((m - img_flow) / m) * (cmap_max - cmap_min)

    # apply power curve to these speeds, using layout as mask
    # img_power = power_curve_function(img_flow[img_layout == 1])  # * img_layout
    uturb = img_flow[layout_mask]
    #plt.imshow(img_flow,cmap='jet')

    img_power = power_curve_function(uturb)
    # embed()
    # use image size and farm size to determine the area that each pixel covers
    # and how many pixels represent a turbine
    # need this for power calculation
    # assumes a SQUARE IMAGE
    # TODO: Assumes square image
    # pix_per_m = img_size[0] * dpi / farm_size
    pix_per_m = grid_resolution / farm_size

    pix_per_turbine = max(floor(d_rot * pix_per_m), 1)
    # now the thickness
    thick = max(floor(rel_thickness * d_rot * pix_per_m), 1)
    if DRAW_TURBINES_AS_LINES:
        turbine_pixel_size = pix_per_turbine * thick
    else:
        turbine_pixel_size = 1

    # totalize power AEP [kW-hr] considering number of turbines and pixels
    # this is equivalent to using the average power at each turbine site
    # with a ROUGH estimation of the number of turbines
    total_power = 8766 * img_power.sum() / turbine_pixel_size
    # print(total_power)

    # if turbine coordinates are given then this has priority
    # so we overwrite the power estimate from above
    if turbine_pixel_coordinates is not None:
        mask = np.zeros(img_flow.shape)
        ixt = turbine_pixel_coordinates
        # note that imshow() has reversed vertical axis w.r.t plot()
        # and we got turbine coordinates based on coordinate origin
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
                    # mask[grid_resolution-ixt[:,1],ixt[:,0]] = 1  #
            mask = np.bool_(mask)
            power = power_curve_function(img_flow[mask])
            # total_power = 8766 * power.sum()
            # print(total_power)
            total_power = 8766 * power.sum() / turbine_pixel_size
        else:
            # DRAW_TURBINES_AS_ONE_PIXEL
            # Note the "-1" which is needed so that pixels are aligned with the mask read from the layout file
            # NOT SURE WHY??? Maybe imread() issues?
            mask[grid_resolution-ixt[:,1]-1,ixt[:,0]] = 1  #
            mask = np.bool_(mask)
            power = power_curve_function(img_flow[mask])
            total_power = 8766 * power.sum()
            # print(total_power)

    # re-calculate power if I have nt but not the pixel coordinates
    # still has priority over detecting nt in the image
    if n_turbines is not None and turbine_pixel_coordinates is None:
        total_power = 8766 * img_power.mean() * n_turbines
        # print(total_power)

    # embed()
    return total_power


def calculate_flow_field(
        x_turbines,
        farm_size,
        u_inf,
        wind_theta,
        z_hub,
        d_rot,
        T,
        cw,
        z0,
        xplot,
        yplot,
        grid_resolution,
):
    """
    x_turbines: real, Cartesian coordinates of the turbines
    """

    # if given a flattened vector of turbine coordinates, convert to a matrix with x,y as columns
    F=math.pi*(d_rot/2)**2
    pi=math.pi
    Deff=d_rot*math.sqrt((1+math.sqrt(1-cw))/(2*math.sqrt(1-cw)))
    a=1.08*d_rot
    b=1.08*d_rot+21.7*(T-0.05)*d_rot
    Rnb=np.max(np.array([a,b]))
    R=0.5*(Rnb+np.min(np.array([z_hub,Rnb])))
    x0=(9.5*d_rot)/((2*R/Deff)**3-1)
    xt = x_turbines

    if xt.ndim == 1:
        nt = xt.shape[0] / 2
        xt = np.reshape(xt, (nt, 2))
    else:
        nt = xt.shape[0]

    # broadcast rotor diameter and hub height if the given value is a scalar
    if z_hub is not np.array:
        z_hub = [z_hub] * nt
    if d_rot is not np.array:
        d_rot = [d_rot] * nt

    # build matrix of plot points
    # +1 and slicing are needed so turbines are placed at lower left corners
    # of each pixel
    if xplot is None:
        x_range = np.linspace(0.0, farm_size, grid_resolution+1)[:-1]
        y_range = np.linspace(0.0, farm_size, grid_resolution+1)[:-1]
        xplot, yplot = np.meshgrid(x_range, y_range)

    # create matrix of plot points
    xyplot = np.transpose(np.vstack((xplot.ravel(), yplot.ravel())))
    #print('xyplot=',xyplot.shape)

    # auxiliary wake-indicator matrix
    #print('nt=',nt)
    wake_indicator = np.zeros((nt, xyplot.shape[0]))
    #print('wake_indicator=',np.unique(wake_indicator,return_counts=True))
    # initialize wake matrix
    u_wake = u_inf * np.ones((nt, xyplot.shape[0]))
    v_wake=np.zeros((nt, xyplot.shape[0]))
    print(v_wake.shape)
    #print('u_wake=',np.unique(u_wake,return_counts=True))
    #print('u_wake=',u_wake)

    # unit vector in wind direction
    # u_hat = np.array([np.cos(wind_theta), np.sin(wind_theta)])

    # length of wake
    farm_area = farm_size * np.ones((1, 2))
    wake_length = np.linalg.norm(farm_area)

    # loop over turbines
    for i in np.arange(0, nt):
        # turbine-specific parameters
        zi = z_hub[i]
        r_rot = d_rot[i] / 2.0

        ri = xt[i]
        x0 = x0

        c1 = (Deff/2)**(5/2)*(105/(2*pi))**(-1/2)*(cw*F*x0)**(-5/6)

        # loop over plot points
        for j in np.arange(0, xyplot.shape[0]):
            # position vector of this point
            rj = xyplot[j]
            #print('rj=',rj)
            # rj_hat = rj / np.linalg.norm(rj)

            # relative position between turbine and point
            rij = rj - ri
            #print('rij=',rij)
            # rij_hat = rij / np.linalg.norm(rij)
            dij = np.linalg.norm(rij)

            if ri[0]<rj[0]:

                r0=(35/(2*math.pi))**(1/5)*(3*c1**2)**(1/5)*(cw*F*(rj[0]-ri[0]+x0))**(1/3)
                # print('r0=',r0)
                # area above the centre line
                if(ri[1]<=rj[1]<ri[1]+r0):
                    #print(rj)
                    # velocity in x-direction
                    u_wake[i, j]=u_inf+(-1*u_inf/9*(cw*F*(rj[0]-ri[0]+x0)**(-2))**(1/3) \
                                        *((rj[1]-ri[1])**(3/2)*(3*c1**2*cw*F*(rj[0]-ri[0]+x0))**(-1/2)-(35/(2*pi))**(3/10) \
                                          *(3*c1**2)**(-1/5))**2)
                    r=rj[1]-ri[1]
                    # velocity in y-direction
                    v_wake[i, j]=-1*((u_inf/3)*(cw*F)**(1/3)*(rj[0]-ri[0]+x0)**(-5/3)*r*((r**(3/2)*(3*c1**2*cw*F*(rj[0]-ri[0]+x0))**(-1/2) \
                                                                                          -(35/(2*pi))**(3/10)*(3*c1**2)**(-1/5))**2))
                # area below centre line
                if(ri[1]>rj[1]>ri[1]-r0):
                    # print(rj[0])
                    r=ri[1]-rj[1]
                    #u_wake[i, j]=u0-(-u0/9*(cw*F*(rj[0]-ri[0]+)**(-2))**(1/3)*((ri[1]-rj[1])**(3/2)*(3*c1**2*cw*F*(rj[0]-ri[0]+x0))**(-1/2)-(35/(2*pi))**(3/10)*(3*c1**2)**(-1/5))**2)
                    # velocity in x-direction
                    u_wake[i, j]=u_inf+(-1*u_inf/9*(cw*F*(rj[0]-ri[0]+x0)**(-2))**(1/3) \
                                        *((ri[1]-rj[1])**(3/2)*(3*c1**2*cw*F*(rj[0]-ri[0]+x0))**(-1/2)-(35/(2*pi))**(3/10) \
                                          *(3*c1**2)**(-1/5))**2)
                    # velocity in y-direction
                    v_wake[i, j]=((u_inf/3)*(cw*F)**(1/3)*(rj[0]-ri[0]+x0)**(-5/3)*r*((r**(3/2)*(3*c1**2*cw*F*(rj[0]-ri[0]+x0))**(-1/2) \
                                                                                       -(35/(2*pi))**(3/10)*(3*c1**2)**(-1/5))**2))
                    #print(u_wake[i, j])

                    wake_indicator[i, j] = 1



        #print('i=',i)
    #print(ri)    #print('wake_indicator=',np.unique(wake_indicator,return_counts=True))
    # aggregate wake effects on each point
    # embed()
    #v_eff =( np.sqrt(np.sum(np.square( v_wake ), axis=0)))
    v_eff =np.sum( v_wake , axis=0)
    u_eff = u_inf * (1 - np.sqrt(np.sum(np.square(1 - u_wake / u_inf), axis=0)))

    #v_eff= (np.sqrt(np.sum(np.square(1 - u_wake / u_inf), axis=0)))
    #print('u_eff=',u_eff)


    u_eff = np.reshape(u_eff, xplot.shape)
    v_eff = np.reshape(v_eff, xplot.shape)
    #print(np.unique(u_eff))
    #plt.imshow(u_eff, cmap='jet')
    #plt.show()
    #plt.imshow(v_eff, cmap='jet')
    #plt.show()
    u_norm = u_eff / u_inf

    return xplot, yplot, u_eff, v_eff, u_norm


def plot_flow_field(
        u,
        outputname,
        dpi,
        vmin,
        vmax,
        img_size,
        cmap="gray",
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
    # ax.set_axis_off()
    plt.axis("off")
    plt.clim = [vmin, vmax]
    plt.colorbar()

    if outputname is not None:

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
                # Also, if I use origin='lower' to create the image
                # I do not need to reverse the y axis here
                lay[xpix[:, 1] - i, xpix[:, 0]] = 1
                lay[xpix[:, 1] + i, xpix[:, 0]] = 1
                lay[xpix[:, 1] - i, xpix[:, 0] + j] = 1
                lay[xpix[:, 1] + i, xpix[:, 0] + j] = 1

    else:
        lay[xpix[:, 1], xpix[:, 0]] = 1
        # lay[grid_resolution - xpix[:, 1], xpix[:, 0]] = 1

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
    # ax.set_axis_off()
    plt.axis("off")
    plt.clim = [vmin, vmax]
    #plt.colorbar()
    if outputname is not None:

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


def plot_turbine_layout_2(
        xt,
        r_rot,
        wind_theta,
        rel_thickness,
        farm_size,
        outputname,
        dpi,
        img_size,
        grid_resolution,
):

    fig = plt.figure(
        figsize=(img_size, img_size), dpi=dpi, frameon=False, facecolor="white"
    )
    fig.set_size_inches(img_size, img_size)
    ax = fig.add_subplot(111)
    plt.xlim(0.0, farm_size)
    plt.ylim(0.0, farm_size)

    # thickness of the turbine in the flow direction
    thickness = r_rot * 2 * rel_thickness

    for i in range(xt.shape[0]):
        # coordinates of the corner of the turbine footprint
        xi = xt[i] - np.array([thickness / 2, r_rot])
        # create rectangle oriented with x-y axes
        ri = patches.Rectangle(
            (xi[0], xi[1]), thickness, r_rot * 2, color="black", alpha=1.0
        )
        # rotate rectangle
        # affine transform to rotate turbines to face the wind
        rotation = (
                mpl.transforms.Affine2D().rotate_around(xt[i, 0], xt[i, 1], wind_theta)
                + ax.transData
        )

        # rotate rectangle
        ri.set_transform(rotation)
        # add it to the plot
        ax.add_patch(ri)

    plt.axis("square")
    plt.axis("off")
    if outputname is not None:
        # Workaround to get numpy matrix of actual image data drawed into the canvas
        # in the previous code section. Needed becase I am not plotting a numpy array,
        # I am drawing, and then extracting an array of the result for saving as image.
        # https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        io_buf = io.BytesIO()
        # fig.savefig(io_buf, format="raw", dpi=dpi)
        fig.savefig(io_buf, format="raw", dpi=dpi, pad_inches=0.0, bbox_inches=0.0)
        io_buf.seek(0)
        img_arr = np.reshape(
            np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
            newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1),
        )
        io_buf.close()

        plt.imsave(outputname, arr=np.squeeze(img_arr), format="png", dpi=dpi)
        # plt.savefig(outputname, dpi=dpi, bbox_inches = 'tight', pad_inches = 0.0)

        plt.close()

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

    P = calculate_power_from_image(
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


# WORK IN PROGRESS
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
        "v_image_file",
        "wind_resource_file",
        "layout_image_file",
        "turbine_coordinates [m]",
        "turbine_coordinates [px]",
        "jensen_power [kW-hr]",
        "image_power_given_xt [kW-hr]",
        "grid_resolution [px]",
        "power_error_given_xt [%]"
    ]
    df = pd.DataFrame(columns=col_names)

    # main loop to generate images
    idx = np.arange(len(xt_cases)).tolist()
    for it, xt, ui, theta in tqdm(
            zip(idx, xt_cases, u_inf_cases, wind_theta_cases), total=len(idx)
    ):
        # generate image file name
        # No zero-padding the name to make it easier to read elsewhere
        image_name = "{}_{:d}".format(base_name, it)
        # image_name = os.path.join(os.getcwd(),image_name)
        flow_name = image_name + "_flow.png"
        vflow_name = image_name + "_v_flow.png"
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
        x, y, u,v, u_norm = calculate_flow_field(
            x_turbines=xt,
            u_inf=ui,
            wind_theta=theta,
            farm_size=farm_size,
            z_hub=z_hub,
            d_rot=d_rot,
            cw=cw,
            T=T,
            z0=z0,
            xplot=xplot,
            yplot=yplot,
            grid_resolution=grid_resolution,
        )


        u_npy_name=image_name+'u_flow.npy'
        v_npy_name=image_name+'v_flow.npy'
        np.save(u_npy_name,u)
        np.save(v_npy_name,v)



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

        xt_pixels = np.int_(np.floor(xt / farm_size * grid_resolution))
        np.save(pixel_coords_name, xt_pixels)

        # calculate true power from Jensen's wake model
        xturb, yturb, uturb,vturb, unorm = calculate_flow_field(
            xt,
            farm_size=farm_size,
            u_inf=ui,
            wind_theta=wind_theta,
            z_hub=z_hub,
            d_rot=d_rot,
            cw=cw,
            T=T,
            z0=z0,
            xplot=xt[:, 0],
            yplot=xt[:, 1],
            grid_resolution=grid_resolution,)

        df= pd.concat([df, pd.DataFrame.from_records([
            dict(
                zip(
                    col_names,
                    [
                        xt.shape[0],
                        ui,
                        theta,
                        flow_name,
                        vflow_name,
                        speed_name,
                        layout_name,
                        real_coords_name,
                        pixel_coords_name,

                        #res_error_no_xt,
                        #res_error_unorm
                    ],
                )
            )
        ])])

    # save ALL layouts as pickle file for re-loading via --layout_file CLI option
    pickle.dump(xt_cases, open(base_name + "_layouts.pkl", "wb"))

    # save case information as CSV and JSON for later use
    #df.index.name = "case_number"
    df.reset_index(drop=True, inplace=True)

    df.to_csv(base_name + "_cases.csv")

    # print('1111111')
    df.to_json(base_name + "_cases.json")
    # report success, incl list of filenames
    print("\nLayout and flow field files have been created:\n")
    print(df)

    return


# WORK IN PROGRESS
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

    # If we are requested to save the configuration
    # we do so now
    # we modify params so that the saved config makes sense
    if params["save_config"]:
        p = params.copy()
        p["save_config"] = False
        p["config_file"] = config_file
        p["load_config"] = True
        with open(config_file, "w") as f:
            json.dump(p, f)

    # set random seed for both numpy and the system
    #np.random.seed(params["random_seed"])
    #random.seed(params["random_seed"])

    # If a file with layouts is given, let's use that
    # We do this FIRST because this file can override n_images
    if params["layout_file"] is not None:
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

    # Make sure all parameters used here are taken from the params dict, not the default values at the top of this script
    d_rot = params["d_rot"]
    grid_resolution = params["grid_resolution"]
    # image resolution
    dpi = params["dpi"]
    # resulting image size in INCHES (works out to 4 x 4)
    # img_size = (grid_resolution / dpi, grid_resolution / dpi)
    img_size = grid_resolution / dpi
    # No files with layouts were provided, so we create them randomly
    # Generate random cases through a centered Latin hypercube
    # TODO: Place turbines only at CENTER of PIXELS, not anywhere
    farm_size = params["farm_size"]
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
                #INFO: turbines are in corners of the grid, NOT the center
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

    # calculate_flow_field(farm_size = farm_area, z_hub = z_hub, d_rot = d_rot, z0 = z0):
    generate_cases(
        xt_cases,
        u_inf_cases,
        wind_theta_cases,
        farm_size=farm_size,
        z_hub=z_hub,
        d_rot=d_rot,
        z0=z0,
        rel_thickness=rel_thickness,
        base_name=params["base_name"],
        dpi=params["dpi"],
        cmap=params["cmap"],
        vmin=params['cmap_range'][0],
        vmax=params['cmap_range'][1],
        img_size=img_size,
        grid_resolution=grid_resolution,
    )
    return


# if __name__ == "__main__":
    import argparse

    # # DEFAULT Values, to be overriden with a config file
    # # wind speed
    # #u_inf = 12
    # # wind direction (vector points in direction of wind)
    # wind_theta = (
    #     0.0  # Wind flows toward East (i.e. a west wind, b/c it comes from the west)
    # )
    # # turbine hub height
    # z_hub = 60
    # # rotor diameter
    # d_rot = 40
    # # thickness of the rotor relative to its diameter
    # rel_thickness = 0.05
    # # number of turbines
    # n_turbines = 3
    # # roughness length
    # z0 = 0.3
    # T=0.12
    # cw=0.9
    # # farm size
    # # TODO: Define as multiple of d_rot
    # farm_size = 2000
    # # number of images
    # n_images = 125
    # # resolution for plotting (number of points per direction)
    # # TODO Needs to be multiple of d_rot for better results
    # grid_resolution = 400  # means one point each 5 m based on size of 2000 m
    # # grid_resolution = 800 # means one point each 2.5 m based on size of 2000 m
    # # image resolution
    # dpi = 100
    # u_inf = 13.0
    # # resulting image size in INCHES (works out to 4 x 4)
    # img_size = grid_resolution / dpi
    # # random seed
    # # seed = 13     # preferred during dev/debugging so results are repeatable
    # seed = 13  # preferred b/c they should default to being random
    # # MIN/MAX limits for randomized cases
    # n_turbines_range = [3, 15]
    # u_inf_range = [5.0, 15.0]  # based on cut-in and rated speeds found in Google (typical cut=in is 2.5, but good sites have 6.0 avg speed
    # wind_theta_range = [0, 0]
    # # colormap for the images
    # cmap = "gray"
    # # limits of colormap: WIDER than u_inf_range
    # # so that speeds below min(u_inf_range) can still
    # # be represented (otherwise HIGH errors at low speeds)
    # # there shouldn't be any speeds above u_inf_max,
    # # at least on 2D simulations
    # #cmap_range = [0.0, 15.0]
    #
    # # Create the parser
    # parser = argparse.ArgumentParser(
    #     description="Generate images of turbine layouts and corresponding wake flow fields",
    #     add_help=True,
    # )
    #
    # # Add the arguments
    # parser.add_argument(
    #     "--n_images",
    #     metavar="Ni",
    #     type=int,
    #     help="Number of images to generate",
    #     default=n_images,
    # )
    # parser.add_argument(
    #     "--n_turbines",
    #     metavar="Nt",
    #     type=int,
    #     help="Number of turbines in the wind farm layout",
    #     default=n_turbines,
    # )
    #
    # parser.add_argument(
    #     "--u_inf",
    #     metavar="u_inf",
    #     type=int,
    #     help="Number of turbines in the wind farm layout",
    #     default=u_inf,
    # )
    # parser.add_argument(
    #     "--base_name",
    #     metavar="NAME",
    #     type=str,
    #     help="Common part of the image filenames. \nThe image number will be added as suffix to each image name\nDefaults to a timestamp",
    #     default=time.strftime("%Y%m%d-%H%M%S"),
    # )
    #
    # # parser.add_argument(
    # # "--u_inf",
    # # metavar="U",
    # # type=float,
    # # help="Wind speed in [m/s]",
    # # default=u_inf,
    # # )
    #
    # parser.add_argument(
    #     "--wind_theta",
    #     metavar="Theta",
    #     type=float,
    #     help='Wind direction in [deg]. \nA NE wind, meaning the wind "comes" from the north-east, would have a direction of 180 + 45 = 225 degrees',
    #     default=0.0,
    # )
    #
    # parser.add_argument(
    #     "--d_rot",
    #     metavar="Dt",
    #     type=float,
    #     help="Rotor diameter in [m]",
    #     default=d_rot,
    # )
    #
    # parser.add_argument(
    #     "--t_rot",
    #     metavar="t",
    #     type=float,
    #     help="Relative rotor thickness. Defaults to d_rot/20",
    #     default=rel_thickness,
    # )
    #
    # parser.add_argument(
    #     "--z_hub",
    #     metavar="Ht",
    #     type=float,
    #     help="Hub height in [m]",
    #     default=z_hub,
    # )
    # parser.add_argument(
    #     "--T",
    #     metavar="TI",
    #     type=float,
    #     help="Turbulance Intensity",
    #     default=T,
    # )
    #
    # parser.add_argument(
    #     "--cw",
    #     metavar="Ct",
    #     type=float,
    #     help="darg Coe",
    #     default=cw,
    # )
    #
    # parser.add_argument(
    #     "--z0",
    #     metavar="Z0",
    #     type=float,
    #     help="Roughness length of the terrain in [m]",
    #     default=z0,
    # )
    #
    # parser.add_argument(
    #     "--farm_size",
    #     metavar="LENGTH",
    #     type=float,
    #     help="Side length in [m] of the wind farm terrain. The wind farm is assumed to be a square with the provided size",
    #     default=farm_size,
    # )
    #
    # parser.add_argument(
    #     "--grid_resolution",
    #     metavar="NPOINTS",
    #     type=int,
    #     help="Resolution of meshgrid used to create the images. Note that this is different from the image resolution",
    #     default=grid_resolution,
    # )
    #
    # parser.add_argument(
    #     "--dpi",
    #     metavar="DPI",
    #     type=int,
    #     help="Resolution (DPI) of created layout and flow field images",
    #     default=dpi,
    # )
    #
    # parser.add_argument(
    #     "--cmap",
    #     metavar="CMAP",
    #     type=str,
    #     help="Colormap used to generate images. Defaults to 'jet'",
    #     default=cmap,
    # )
    #
    # parser.add_argument(
    #     "--u_inf_range",
    #     metavar="X",
    #     type=float,
    #     nargs=2,
    #     help="Min and max values of wind speed used to generate cases. Defaults to 5.0 to 15 m/s, consistent with avg wind speeds in wind farm sites and typical cut-off of wind turbines",
    #     default=u_inf_range,
    # )
    # parser.add_argument(
    #     "--cmap_range",
    #     metavar="CMAP",
    #     type=float,
    #     nargs=2,
    #     help="Min and max values of wind speed used to define the colormap. Defaults to 0.0 to 16 m/s.",
    #     default=u_inf_range,
    # )
    #
    # parser.add_argument(
    #     "--random_seed",
    #     metavar="SEED",
    #     type=int,
    #     help="Seed for the random number generator",
    #     default=seed,
    # )
    #
    # parser.add_argument(
    #     "--save_config",
    #     action="store_true",
    #     help="Flag indicating whether to save all parameter values to a JSON file. Requires --config_file. Cannot be used with --load-config",
    #     default=False,
    # )
    #
    # parser.add_argument(
    #     "--load_config",
    #     action="store_true",
    #     help="Flag indicating whether to load all parameter values from a JSON file. Requires --config_file. Cannot be used with --save-config",
    #     default=False,
    # )
    #
    # parser.add_argument(
    #     "--config_file",
    #     metavar="FILE",
    #     type=str,
    #     help="Filename to either load or save configuration in JSON format. Defaults to a timestamp with a .cfg extension",
    #     default=None,
    # )
    #
    # parser.add_argument(
    #     "--layout_file",
    #     metavar="FILE",
    #     type=str,
    #     help="Filename with turbine coordinates. Must be a pickle file containing a list of numpy arrays, each array of size [nt x 2])",
    #     #default='C:/Users/saeed/Code/DecodeCNN/Error/Larsen/larsen1/5/test_layouts.pkl',
    #     default='D:/Research/CFD_data/CFD_data/CFD_128_layouts-20230329T174354Z/test_layouts.pkl',
    #     #default=None,
    #
    # )
    #
    # parser.add_argument(
    #     "--randomize_all",
    #     action="store_true",
    #     help="Flag indicating whether to randomize the number of turbines, wind speed and wind direction. Hub height is not randomized because this is a 2D model. \n Note that this overrides any conflicting command-line options",
    #     default=False,
    # )
    #
    # parser.add_argument(
    #     "--randomize_nt",
    #     action="store_true",
    #     help="Flag indicating whether to randomize the number of turbines",
    #     default=False,
    # )
    #
    # parser.add_argument(
    #     "--randomize_u",
    #     action="store_true",
    #     help="Flag indicating whether to randomize the wind speed",
    #     default=True,
    # )
    #
    # parser.add_argument(
    #     "--randomize_theta",
    #     action="store_true",
    #     help="Flag indicating whether to randomize the wind direction",
    #     default=False,
    # )
    #
    # config = parser.parse_args()
    #
    # # using vars() converts the NameSpace object (config) into a dict
    # main(vars(config))
