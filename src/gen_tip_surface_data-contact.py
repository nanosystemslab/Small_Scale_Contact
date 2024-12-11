import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
import logging
import numpy as np
import sys
from scipy.interpolate import griddata
from scipy.ndimage import label, find_objects

__version__ = "0.0.1"


def setup_logging(verbosity):
    log_fmt = ("%(levelname)s - %(module)s - "
               "%(funcName)s @%(lineno)d: %(message)s")
    logging.basicConfig(filename=None,
                        format=log_fmt,
                        level=logging.getLevelName(verbosity))
    return


def parse_command_line():
    parser = argparse.ArgumentParser(description="Analyse sensor data")
    parser.add_argument("-V", "--version", "--VERSION", action="version",
                        version="%(prog)s {}".format(__version__))
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        dest="verbosity", help="verbose output")
    parser.add_argument("-d", "--dir", dest="dirs",
                        nargs='+',
                        default=None, required=False,
                        help="directories with data files")
    parser.add_argument("-i", "--in", dest="input",
                        nargs='+',
                        default=None, required=False,
                        help="path to input")
    parser.add_argument("--xlim", dest="xlim", nargs=2, type=float,
                        default=None, required=False,
                        help="Limits for x-axis (min max)")
    parser.add_argument("--ylim", dest="ylim", nargs=2, type=float,
                        default=None, required=False,
                        help="Limits for y-axis (min max)")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])
    return ret


def points_within_circle(data, circle_center_x,
                         circle_center_y, radius):
    distance_squared = ((data['x'] - circle_center_x) ** 2 +
                        (data['y'] - circle_center_y) ** 2)
    return distance_squared <= radius ** 2


def generate_circle_points(center_x, center_y, radius,
                           z_height, num_points=100):
    theta = np.linspace(0, 2 * np.pi, num_points)
    x = center_x + radius * np.cos(theta)
    y = center_y + radius * np.sin(theta)
    z = np.full_like(x, z_height)
    return x, y, z


def main():
    args = parse_command_line()
    setup_logging(args["verbosity"])

    # Load the .asc file
    file_path = args['input'][0]
    print(f"file {file_path}")
    file_name = file_path.split("/")[-1]

    data = pd.read_csv(file_path, sep=r'\s+', header=None,
                       names=['x', 'y', 'z'])

    file_bounds = {
        "NI-Tip-1.asc":  (0.138, 0.160, 0.105, 0.127),
        "NI-Tip-2.asc":  (0.153, 0.175, 0.085, 0.107),
        "NI-Tip-3.asc":  (0.153, 0.175, 0.085, 0.107),
        "NI-Tip-4.asc":  (0.153, 0.175, 0.085, 0.107),
        "NI-Tip-5.asc":  (0.1270, 0.1495, 0.093, 0.1162)
    }

    # Extract boundaries using the dictionary
    if file_name in file_bounds:
        x_min, x_max, y_min, y_max = file_bounds[file_name]
    else:
        raise ValueError(f"Unknown file name: {file_name}")

    # Truncate data based on x and y limits
    data = data[(data['x'] >= x_min) & (data['x'] <= x_max) &
                (data['y'] >= y_min) & (data['y'] <= y_max)] * 1e3
    data['x'] = data['x'] - data['x'].min()
    data['y'] = data['y'] - data['y'].min()

    x = data['x'] * 1e3
    y = data['y'] * 1e3
    z = data['z'] * 1e3
    n = int(np.sqrt(len(z)))

    # Create a grid of x and y points
    x_grid = np.linspace(x.min(), x.max(), n)
    y_grid = np.linspace(y.min(), y.max(), n)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = griddata((x, y), z, (X, Y), method='cubic')

    # Variables for the circle center
    circle_center_x = x.max() / 2
    circle_center_y = y.max() / 2
    radius = 21.456 * 1e3 / 2

    # Define the circle's properties
    circle = Circle((circle_center_x, circle_center_y),
                    radius, color='red', fill=False)
    # Calculate the distance of each point in the grid from the circle center
    distance_from_center = np.sqrt((X - circle_center_x) ** 2 +
                                   (Y - circle_center_y) ** 2)

    # Create a mask for points outside the circle
    outside_circle_mask = distance_from_center > radius

    # Set values outside the circle to NaN
    Z[outside_circle_mask] = np.nan
    Z = Z - np.nanmean(Z)
    mean_Z = np.nanmean(Z)
    std_Z = np.nanstd(Z)

    # Removing scan values over 200
    Z[Z > 200] = 0

    # Plot the result (optional)
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(label='Z values')
    ax = plt.gca()
    ax.add_patch(circle)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Interpolated Z values over X-Y grid')

    # Create the surface plot
    # Mask NaN values to avoid plotting them
    masked_Z = np.ma.array(Z, mask=np.isnan(Z))
    masked_Z[Z < 0] = 0

    # Create a 3D plot for Z
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X / 1e3, Y / 1e3, masked_Z,
                           cmap='plasma_r', edgecolor='none')

    # Add color bar for reference
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=6, pad=0.01)
    colorbar.ax.tick_params(labelsize=14)

    # Set plot labels and title with larger font size
    ax.set_xlabel('X ($\\mu m$)', fontsize=20, labelpad=20)
    ax.set_ylabel('Y ($\\mu m$)', fontsize=20, labelpad=20)
    ax.set_zlabel('Z ($nm$)', fontsize=20,    labelpad=20)
    ax.set_zlim(0, 120)

    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=20, pad=10)
    ax.tick_params(axis='y', labelsize=20, pad=10)
    ax.tick_params(axis='z', labelsize=20, pad=10)

    ax.view_init(elev=15, azim=215)
    # Save the figure
    plt.savefig("out/surface_plot.png", dpi=600)

    dist_std = 1
    threshold = mean_Z + std_Z * dist_std

    # Label connected regions around each peak
    thresholded_Z = Z > threshold
    labeled_array, num_features = label(thresholded_Z)

    # Lists to store diameters and heights
    peak_diameters = []
    peak_heights = []

    # Loop through each peak and measure diameter
    for i in range(1, num_features + 1):
        peak_slice = find_objects(labeled_array == i)[0]

        # Extract the bounding box of the peak
        y_start, y_stop = peak_slice[0].start, peak_slice[0].stop
        x_start, x_stop = peak_slice[1].start, peak_slice[1].stop

        # Calculate approximate diameter as the maximum of width and height
        peak_height = y_stop - y_start
        peak_width = x_stop - x_start
        peak_diameter = max(peak_height, peak_width)

        # Get the maximum Z value within this peak region
        peak_max = np.nanmax(Z[y_start:y_stop, x_start:x_stop])

        if peak_max < 100:
            peak_diameters.append(peak_diameter)
            # Calculate the height from the mean to the top of the peak
            peak_height_from_mean = peak_max - mean_Z
            peak_heights.append(peak_height_from_mean)

    step = -0.01
    x = np.arange(200, 0 - step, step)
    F_B_total = np.zeros(len(x))

    v_Au = 0.42
    v_C = 0.2
    E_Au = 77 * 1e6
    E_C = 1220 * 1e6
    E = 1 / (((1 - v_Au**2) / E_Au) + ((1 - v_C**2) / E_C))

    # Print the diameters and heights of each peak
    for i, (diameter, height) in enumerate(zip(peak_diameters, peak_heights)):
        print(
            f"Peak {i + 1} has an approximate diameter: {diameter} units, "
            f"height from mean: {height: .2f} units"
        )
        R = (130 * diameter * 1e-9) / 2
        D = (height - x) * 1e-9
        D[D < 0] = 0
        F_B = ((4 * E / 3) * (R ** 0.5) * (D ** (3 / 2))) * 1e6
        F_B_total += F_B

    # Create a DataFrame
    data = pd.DataFrame({
        "x": x,
        "F1_total": F_B_total
    })

    # Save to CSV
    name = file_name.split(".")[0]
    output_file = f"out/Model/FB/{name}--FB-results.csv"
    print(output_file)
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    sys.exit()
    # Plot F_B_total vs x
    plt.figure(figsize=(10, 6))
    plt.plot(-1 * x, F_B_total, label="F_B vs x")
    plt.xlabel("z (nm)")
    plt.ylabel("F_B_total ($\\mu N$)")
    plt.title("Plot of F_B  vs z")
    plt.legend()
    plt.grid(True)
    # plt.show()


if __name__ == "__main__":
    main()
