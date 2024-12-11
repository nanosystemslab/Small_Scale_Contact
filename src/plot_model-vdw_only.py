import argparse
import glob
import logging
import os
import sys
import types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting
import seaborn as sns
from scipy.integrate import quad
from scipy.misc import derivative

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
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])
    return ret


def load_file(filepath):
    df = pd.read_csv(filepath, encoding='unicode_escape',
                     sep='\t', skiprows=(0, 1, 2))
    print(df)
    col = df.columns.tolist()
    new_col = []
    for i in col:
        new_col.append(i.replace(" ", "_"))

    df.columns = new_col
    df.meta = types.SimpleNamespace()
    df.meta.filepath = filepath
    run = filepath.split("-")[-1]
    run = run.split("_")[0]
    df.meta.run = run
    return df


def load_multi(filepaths):
    data = []
    for data_file in filepaths:
        tran = load_file(data_file)
        data.append(tran)
    return data


def load_model_files(csv_dir):

    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))

    # Lists to store all x and y data
    x_data = []
    y_data = []

    # Load each CSV and extract x and y columns
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        col = df.columns.tolist()

        # Assuming the CSVs have columns named 'x' and 'y'
        x_data.append(df[col[0]].values)
        y_data.append(df[col[1]].values)

    # Convert lists to numpy arrays for computation
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # Calculate averages and standard deviations along the file axis (rows)
    x_mean = np.mean(x_data, axis=0)

    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)

    return x_mean, y_mean, y_std


def plot_data_multi_trace_poly(param_y="T", param_x="T", data_paths=None):
    """
    plot multi data trace from a file with std

    """
    log = logging.getLogger(__name__)
    log.debug("in")

    dfs = []
    for file in data_paths:
        print(f"loading...{file}")
        df = load_file(file)

        levels_up = os.path.abspath(os.path.join(file, "../"))
        keyword = 'Air_Indent'

        # Search for a file that contains the keyword in its name
        found_file_path = None
        for file_name in os.listdir(levels_up):
            if keyword in file_name:
                found_file_path = os.path.join(levels_up, file_name)
                break

        # Print the result
        if found_file_path:
            print(f"File found: {found_file_path}")
            df_air = load_file(found_file_path)

            # Assuming 'Load (µN)' is the column that needs normalization
            if 'Load_(µN)' in df.columns and 'Load_(µN)' in df_air.columns:
                # Interpolating the Air_Indent data to match experimental
                df_air_interpolated = df_air.reindex(df.index,
                                                     method='nearest')

                # Subtracting air indentation load from experimental load
                df['Load_(µN)'] = (
                    df['Load_(µN)'] - df_air_interpolated['Load_(µN)']
                )

                # Print or save the normalized dataframe
                print(df.head())  # For testing, you can see the top few rows
            else:
                print("Load (µN) column not found in data files")
        else:
            print("File containing the keyword does not exist not normalizing")

        run = df.meta.run

        df = df[df['Time_(s)'] >= 5]
        zero_norm = df[(df['Time_(s)'] >= 5) & (df['Time_(s)'] <= 50)]
        zero_norm = zero_norm['Load_(µN)'].mean()
        df['Load_(µN)'] = df['Load_(µN)'] - zero_norm
        df = df[df['Time_(s)'] >= 50]

        df.meta = types.SimpleNamespace()
        df.meta.run = run
        df.meta.file = file
        print(df)
        dfs.append(df)

    figsize = 4
    figdpi = 600
    hwratio = 16. / 9
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    hold_x = []
    hold_y = []
    with sns.axes_style("darkgrid"):
        for df in dfs:
            lab = df.meta.run
            file = df.meta.file
            df = df.iloc[np.linspace(0, len(df) - 1,
                                     num=int(len(df) * 0.1), dtype=int)]
            x = df[param_x]
            y = df[param_y]
            x = x.dropna()
            y = y.dropna()

            # Ensure both x and y have matching indices after dropping NaNs
            valid_index = x.index.intersection(y.index)
            x = x.loc[valid_index]
            y = y.loc[valid_index]

            hold_x.append(x)
            hold_y.append(y)

        df_combined = pd.DataFrame()
        for i, (x, y) in enumerate(zip(hold_x, hold_y)):
            x_reset = x.reset_index(drop=True)
            y_reset = y.reset_index(drop=True)
            df = pd.DataFrame({
                'X': x_reset,
                'Y': y_reset
            })
            df.set_index('X', inplace=True)
            df_combined = df_combined.join(df, how='outer', rsuffix=f'_{i}')

        # Sort by index if the x values are numeric and need to be in order
        df_combined.sort_index(inplace=True)

        # Perform linear interpolation to fill NaN values
        df_combined.interpolate(method='linear', inplace=True)

        # Fill any remaining NaNs at the start or end of the DataFrame
        df_combined.fillna(method='bfill', inplace=True)  # Backward fill
        df_combined.fillna(method='ffill', inplace=True)  # Forward fill

        # Display the combined and interpolated DataFrame
        df_combined['Average'] = df_combined.mean(axis=1)
        df_combined['STD'] = df_combined.std(axis=1)
        df_combined['SEM'] = (df_combined['STD'] /
                              np.sqrt(df_combined.notna().sum(axis=1)))

        df_combined = df_combined.iloc[::4]
        depth = df_combined.index  # Depth values (formerly 'X')
        load_average = df_combined['Average']  # Load average values
        load_std = df_combined['SEM']  # Load standard deviation values

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

        # Plot the average load on the given ax object
        ax.plot(depth, load_average, label='Experimental Load', color='blue')
        # Shaded STD area
        ax.fill_between(depth,
                        load_average - load_std, load_average + load_std,
                        color='blue', alpha=0.2, label='Standard Deviation')
        if param_x == "Time_(s)":
            xh = 160
            xl = 0
            xh = None
            xl = None
        elif param_x == "Depth_(nm)":
            xh = 0.1
            xl = -200
        else:
            xh = None
            xl = None

        def model(x):
            R = (130 * 1e-9) / 2
            N = 74357
            x = x * 1e-9
            Ah = 27.2 * 10**-20
            sigma = 13.204 * 1e-9

            def PDF(x):
                # Define PDF function
                return ((1 / (sigma * (2 * np.pi)**0.5)) *
                        np.exp((-1 / 2) * (x / sigma)**2))

            # Numerical integration for n
            z = 200 * 1e-9
            n_integral, _ = quad(PDF, z, np.inf)
            n = N * n_integral

            # Interaction energy at specific separation distance
            def W_i(x):
                return -((1 / 6) * Ah * (R / x + R / (2 * R + x) +
                                         np.log(x / (2 * R + x))))

            # Numerical derivative of W_i
            def dW_i(x):
                return derivative(W_i, x, dx=1e-10)

            # Total energy
            W = -(n * dW_i(x) + (N - n) * dW_i(x)) * 1e6
            return W

        # Array setup
        step = -0.1
        x = np.arange(200, 0 - step, step)
        W = model(abs(x))

        def model_2(x):
            x = x * 1e-9
            Ah = 27.2 * 10**-20
            D = 21.456 * 1e-6
            Area = np.pi * (D / 2)**2
            return -(Ah * Area) / (6 * np.pi * x**3) * 1e6

        W2 = model_2(abs(x))

        # Define indices for markers
        marker_indices = np.linspace(0, len(x) - 1, 10, dtype=int)
        marker_indices2 = np.linspace(0, len(x) - 1, 15, dtype=int)

        # Plot vdW Adhesion with markers and line
        ax.plot(-1 * x, W2, label='vdW Adhesion (HS)', color='#FF1493',
                linestyle='--', marker='d', markevery=marker_indices2)
        ax.plot(-1 * x, W, label='vdW Adhesion ($\\sigma$)', color='purple',
                linestyle='--', marker='^', markevery=marker_indices)
        ax.set_xlim([xl, xh])
        ax.set_xlabel("Separation Distance ($nm$)", fontsize=14)

        # Customize x-axis ticks to display absolute values
        ax.set_xticks(np.linspace(-200, 0, 10))  # Define tick positions
        ax.set_xticklabels(np.abs(ax.get_xticks()).astype(int), fontsize=14)
        ax.set_ylim([-7.5, 2])
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylabel("Load ($\\mu N$)", fontsize=14)
        ax.legend(loc='lower left', fontsize=14)
        # plt.show()

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = f"out/plot-{param_y}-vs-{param_x}-vdw_only.png"
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])
    plot_data_multi_trace_poly(param_y="Load_(µN)", param_x="Depth_(nm)",
                               data_paths=cmd_args['input'])


if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
       print("exited")


