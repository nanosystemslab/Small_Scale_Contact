import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse
import logging
import numpy as np
from scipy.stats import norm

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
                        default=None, required=False, help="path to input")
    parser.add_argument("--xlim", dest="xlim", nargs=2, type=float,
                        default=None, required=False,
                        help="Limits for x-axis (min max)")
    parser.add_argument("--ylim", dest="ylim", nargs=2, type=float,
                        default=None, required=False,
                        help="Limits for y-axis (min max)")
    parser.add_argument("--display", action="store_true", default=False,
                        help="Display plots or other visual outputs")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])
    return ret


def points_within_circle(data, circle_center_x, circle_center_y, radius):
    distance_squared = (
        (data['x'] - circle_center_x) ** 2 +
        (data['y'] - circle_center_y) ** 2
    )
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
    file_path = args['input'][0]  # Replace with your actual file path
    file_name = file_path.split("/")[-1]

    data = pd.read_csv(file_path, delim_whitespace=True,
                       header=None, names=['x', 'y', 'z'])

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

    # Variables for the circle center
    circle_center_x = data["x"].max() / 2
    circle_center_y = data["y"].max() / 2
    radius = 21.456 / 2

    # Define the circle's properties
    circle = Circle((circle_center_x, circle_center_y), radius,
                    color='red', fill=False)

    if args["display"]:
        plt.figure(figsize=(10, 6))
        plt.scatter(data['x'], data['y'], c=data['z'], cmap='viridis')
        ax = plt.gca()
        ax.add_patch(circle)
        plt.colorbar(label='Z Value')
        plt.xlabel('X Label')
        plt.ylabel('Y Label')
        plt.title('2D Plot of Truncated XYZ Data with Z as Color')

    # Filter data for points within the circle
    within_circle_data = data[points_within_circle(data, circle_center_x,
                                                   circle_center_y, radius)]

    z_values = within_circle_data['z']
    mu, std = norm.fit(z_values)

    if args["display"]:
        # Create a histogram of the z values
        plt.figure(figsize=(10, 6))
        count, bins, _ = plt.hist(z_values, bins=20, density=True, alpha=0.6,
                                  color='skyblue', edgecolor='black',
                                  label='Data Histogram')

        # Create a Gaussian curve based on the mean and stdev of the data
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)

        # Plot the Gaussian curve
        plt.plot(x, p, 'r', linewidth=2,
                 label=f'Gaussian Fit ($\mu={mu: .2f}$, $\sigma={std: .2f}$)')
        plt.xlabel('Z Value')
        plt.ylabel('Density')
        plt.title('Distribution of Z Values with Gaussian Fit')
        plt.legend()

    # Define lower and upper bounds for acceptable z values
    lower_bound = mu - 2 * std
    upper_bound = mu + 2 * std

    within_circle_data = within_circle_data[
        (within_circle_data['z'] >= lower_bound) &
        (within_circle_data['z'] <= upper_bound)
    ]

    # Z height for the circle plane
    circle_z_height = within_circle_data['z'].mean()

    # Generate points for the circle
    circle_x, circle_y, circle_z = generate_circle_points(circle_center_x,
                                                          circle_center_y,
                                                          radius,
                                                          circle_z_height)

    # Set mean plane hight
    within_circle_data['z'] = (within_circle_data['z'] - circle_z_height) * 1e3

    if args["display"]:
        # Plotting the 3D scatter plot for points within the circle
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(within_circle_data['x'], within_circle_data['y'],
                   within_circle_data['z'], c=within_circle_data['z'],
                   cmap='viridis')

        # Plot the circle at the given Z height
        ax.plot(circle_x, circle_y, 0, color='red', label='Circle at Z height')
        vertices = [list(zip(circle_x, circle_y, [0] * len(circle_x)))]
        circle_patch = Poly3DCollection(vertices, color='red', alpha=0.1)
        ax.add_collection3d(circle_patch)
        ax.set_xlabel('X ($\\mu m$)')
        ax.set_ylabel('Y ($\\mu m$)')
        ax.set_zlabel('Z (nm)')
        ax.set_title('3D Scatter Plot of XYZ Data within Circle Region')

    x = within_circle_data['x']
    y = within_circle_data['y']
    z = within_circle_data['z']
    n = int(np.sqrt(len(z)))

    # Create a grid of x and y points
    x_grid = np.linspace(x.min(), x.max(), n)
    y_grid = np.linspace(y.min(), y.max(), n)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Extract 'z' values from within_circle_data
    z_values = within_circle_data['z']
    z_values = np.array(z_values)

    # Array setup
    step = -2
    x = np.arange(200, 0 - step, step)

    # Ensure the array dimensions match
    num_rows = len(z_values)
    dist_asp = np.tile(np.linspace(200, 0, len(x)), (num_rows, 1))
    dist_cont = np.tile(np.linspace(200, 0, len(x)), (num_rows, 1))

    for i in range(num_rows):
        dist_asp[i, :] -= z_values[i]
        dist_cont[i, :] -= z_values[i]
        # Check for negative values and set them to 200 pm
        dist_asp[i, :] = np.where(dist_asp[i, :] < 0, 0.2e-9, dist_asp[i, :])

    # Coefficients
    AH = 27.2 * 10**-20
    R = (130 * 1e-9) / 2  # radius in meters

    # Calculation adjustment for distance
    dist_asp = dist_asp * 1e-9

    # Greenwood calculation
    F_1 = (
        (1 / 6) * AH *
        (R / dist_asp + R / (2 * R + dist_asp) +
            np.log(dist_asp / (2 * R + dist_asp)))
    )

    F1_total = np.sum(F_1, axis=0)
    F1_total = np.gradient(F1_total, step)
    F1_total = np.cumsum(F1_total) * 1e6

    x = np.linspace(-200, 0, len(F1_total))

    # Create a DataFrame
    data = pd.DataFrame({
        "x": x,
        "F1_total": F1_total
    })

    # Save to CSV
    name = file_name.split(".")[0]
    output_file = f"out/Model/F1/{name}--F1-results.csv"
    data.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    if args["display"]:
        plt.figure(figsize=(10, 6))
        plt.plot(x, F1_total, marker='o', linestyle='-',
                 color='b', label='F1 Total')
        plt.title(f'F1 for {name}')
        plt.xlabel('Step (x)')
        plt.ylabel('Summed Forces')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
