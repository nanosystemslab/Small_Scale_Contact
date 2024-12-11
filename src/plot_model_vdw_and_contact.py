import argparse
import glob
import logging
import os
import sys
import shutil
import types
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.plotting
import seaborn as sns
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

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


def plot_data_single_trace(param_y="T", param_x="T", data_paths=None):
    """
    plot one data trace from a file

    simple plotting example
    only load data from first file matched in data_paths
    """
    log = logging.getLogger(__name__)
    log.debug("in")

    df = load_file(glob.glob(data_paths[0])[0])
    run = df.meta.run
    window_length = 12  # The window length should be a positive odd number
    polyorder = 7  # The polynomial order must be less than window_length
    smoothed_depth = savgol_filter(df['Depth_(nm)'], window_length, polyorder)
    prominence_value = np.abs(smoothed_depth).max() * 0.5
    peaks, _ = find_peaks(smoothed_depth, prominence=prominence_value)
    troughs, _ = find_peaks(-smoothed_depth, prominence=prominence_value)
    inflection_point_indices = np.sort(np.concatenate((peaks, troughs)))
    df = df[:inflection_point_indices[0]]

    dataframes_list = []
    for start, end in zip(inflection_point_indices[:-1],
                          inflection_point_indices[1:]):
        # Slice the dataframe from start to end index
        sliced_df = df.iloc[start:end + 1]
        dataframes_list.append(sliced_df)


    figsize = 4
    figdpi = 200
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):
        x = df[param_x]
        y = df[param_y] - df[param_y].iloc[-1]
        x = x.dropna()
        y = y.dropna()

        # Ensure both x and y have matching indices after dropping NaNs
        valid_index = x.index.intersection(y.index)

        x = x.loc[valid_index]
        y = y.loc[valid_index]
        ax.plot(x, y, label="Experimental")

        logging.info(sys._getframe().f_code.co_name)
        plt.show()
        plot_fn = "out/plot-single-{}-vs-{}-{}.png".format(
            param_y, param_x, run)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_data_multi_trace(param_y="T", param_x="T", data_paths=None):
    """
    plot multiple data trace from a files

    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Depth_(nm)": {'label': "Depth", 'hi': 4, 'lo': 0, 'lbl': "Depth(nm)"},
        "Load_(µN)": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Load(µN)"},
        "Time_(s)": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Depth_(V)": {'label': "Depth", 'hi': 4, 'lo': 0, 'lbl': "Depth (V)"},
        "Load_(V)": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Load (V)"}
    }

    figsize = 4
    figdpi = 600
    hwratio = 4. / 3
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)
    with sns.axes_style("darkgrid"):
        data = load_multi(data_paths)
        for df in data:
            test_run_num = df.meta.test_run
            date = df.meta.date
            x = df[param_x]
            y = df[param_y]
            ax.plot(x, y, label=test_run_num)

        ax.legend()
        ax.set_xlabel(plot_param_options[param_x]['lbl'])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/{}--plot-multi-{}-vs-{}-{}.png".format(
            date, param_y, param_x, test_run_num)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def plot_data_multi_trace_std(param_y="T", param_x="T", data_paths=None):
    """
    plot multi data trace from a file with std

    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Depth_(nm)": {'label': "Depth", 'hi': 4, 'lo': 0, 'lbl': "Depth(nm)"},
        "Load_(µN)": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Load(µN)"},
        "Time_(s)": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Depth_(V)": {'label': "Depth", 'hi': 4, 'lo': 0, 'lbl': "Depth (V)"},
        "Load_(V)": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Load (V)"}
    }

    dfs = []
    for file in data_paths:
        print(file)
        df = load_file(file)

        run = df.meta.run
        # The window length should be a positive odd number
        window_length = 12
        # The polynomial order must be less than window_length
        polyorder = 7
        smoothed_depth = savgol_filter(df['Depth_(nm)'],
                                       window_length, polyorder)
        # Adjust this factor as needed
        prominence_value = np.abs(smoothed_depth).max() * 0.5
        peaks, _ = find_peaks(smoothed_depth, prominence=prominence_value)
        troughs, _ = find_peaks(-smoothed_depth, prominence=prominence_value)
        # Combine and sort the indices of peaks and troughs
        inflection_point_indices = np.sort(np.concatenate((peaks, troughs)))
        df = df[:inflection_point_indices[0]]

        df.meta = types.SimpleNamespace()
        df.meta.run = run
        dfs.append(df)

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 16. / 9
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    hold_x = []
    hold_y = []
    with sns.axes_style("darkgrid"):
        for df in dfs:
            x = df[param_x]
            y = df[param_y] - df[param_y].iloc[-1]
            x = x.dropna()
            y = y.dropna()

            # Ensure both x and y have matching indices after dropping NaNs
            valid_index = x.index.intersection(y.index)
            x = x.loc[valid_index]
            y = y.loc[valid_index]

            min_index = y.idxmin()
            # Select the data from the start to the min_index
            x = -x
            y = y
            min_index = np.argmin(y)

            # ax.plot(x, y)#, label=f"Experimental Trial: {df.meta.run}")
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
        df_combined = df_combined.iloc[::4]
        depth = df_combined.index  # Depth values (formerly 'X')
        load_average = df_combined['Average']  # Load average values
        load_std = df_combined['STD']  # Load standard deviation values

        ## Plot the average load on the given ax object
        #ax.plot(depth, load_average, label='Load Average', color='blue')
        ## Shaded STD area
        #ax.fill_between(depth,
        #                load_average - load_std, load_average + load_std,
        #                color='blue', alpha=0.2, label='Load STD')
        # Plot the line
        ax.plot(depth, load_average, label='Load Average', color='blue')

        # Plot the shaded area (STD)
        ax.fill_between(depth,
                        load_average - load_std, load_average + load_std,
                        color='blue', alpha=0.2)

        # Create a custom legend entry combining the line and shaded area
        custom_line = Line2D([0], [0], color='blue', linestyle='-', alpha=1, label='Load Average with STD')
        # model plate
        def model_plate(x, Ah, Area):
            x = x*1e-9
            return 1e6*(-Ah * Area) / (6 * np.pi * x**3)

        def model_casimir_plate(x, Ah, Area):
            p3 = (1 /(1+0.14*(x)))
            x = x*1e-9
            p1 = (-Ah * Area)
            p2 = (6 * np.pi * x**3)
            return 1e6*((p1*p3)/p2)

        def model_linear(x, B, C):
            x = x*1e-9
            Ah = 27.2 * 10**-20
            D = 21.456 * 1e-3
            Area = np.pi * (D / 2)**2
            p1 = (-Ah * Area)
            p2 = (6 * np.pi * x**3)
            return 1e6*((p1/p2) + np.tan(B)*x +C)

        def model_gaussian(x, A):
            x = x*1e-9
            Ah = 27.2 * 10**-20
            D = 21.456 #* 1e-6
            Area = np.pi * (D / 2)**2
            p1 = (-Ah * Area)
            p3 = (6 * np.pi * x**3)
            return 1e6*(p1*A)/p3

        def model_sphere_capella(x, A):
            #R = 0.5*1e-9
            #R = (1/13.204)*1e-9
            R = 0.0306*1e-9
            N = A
            x = x*1e-9
            Ah = 27.2 * 10**-20
            D = 21.456 #* 1e-6
            Area = np.pi * (D / 2)**2
            p1 = (-Ah * R * N)
            p2 = (6 *  x**2)
            return 1e6*(p1)/p2

        def model_sphere_capella_c(x, A):
            #R = 0.5*1e-9
            #R = (1/13.204)*1e-9
            R = 0.0306*1e-9
            N = A
            x = x*1e-9
            Ah = 27.2 * 10**-20
            D = 21.456 #* 1e-6
            Area = np.pi * (D / 2)**2
            p1 = (-Ah * R * N)
            p2 = (6 *  x**2)
            p3 = (1 /(1+0.14*(x*1e9)))
            return 1e6*(p1*p3)/p2

        Ah = 27.2 * 10**-20
        D = 21.456 * 1e-6
        Area = np.pi * (D / 2)**2

        # Convert hold_x and hold_y to DataFrames
        df_x = pd.DataFrame(hold_x)
        df_y = pd.DataFrame(hold_y)

        # Calculate the average x and y values across all arrays
        x = df_x.mean(axis=0)
        y = df_y.mean(axis=0)
        min_index = y.idxmin()

        mask = (x >= x[min_index +3]) & (x <= 200)
        x_filtered = x[mask]
        y_filtered = y[mask]


        popt_lin, pcov_lin = curve_fit(model_linear, x_filtered, y_filtered)
        B_lin, C_lin = popt_lin

        popt_gau, pcov_gau = curve_fit(model_gaussian, x_filtered, y_filtered)
        A_gau = popt_gau

        popt_cap, pcov_cap= curve_fit(model_sphere_capella_c, x_filtered, y_filtered)
        A_cap = int(popt_cap)
        #A_cap = int(1e5 * 1e12 * Area)

        print(A_cap)
        #A_cap = int(1e5 * 1e12 * Area)

        # Generate y values using the fitted model
        y_plate = model_plate(x_filtered, Ah, Area)
        y_casimir = model_casimir_plate(x_filtered, Ah, Area)
        y_linear = model_linear(x_filtered, B_lin, C_lin)
        y_gaussian = model_gaussian(x_filtered, A_gau)
        y_capella =  model_sphere_capella(x_filtered, A_cap)
        A1000000   = 1000000
        A10000000  = 10000000
        A100000000 = 100000000
        A1000000000 = 1000000000
        A10000000000 = 10000000000
        A100000000000 = 100000000000

        y_capella_c1000000      =  model_sphere_capella_c(x_filtered, A1000000  )
        y_capella_c10000000     =  model_sphere_capella_c(x_filtered, A10000000 )
        y_capella_c100000000    =  model_sphere_capella_c(x_filtered, A100000000)
        y_capella_c1000000000   =  model_sphere_capella_c(x_filtered, A1000000000)
        y_capella_c10000000000  =  model_sphere_capella_c(x_filtered, A10000000000)
        y_capella_c100000000000  =  model_sphere_capella_c(x_filtered, A100000000000)

        #r2 = r2_score(y_filtered, y_capella_c)
        #print("R² score:", r2)
        #ax.plot(x_filtered, y_plate, marker='o', markevery=0.1, label='Plate')
        #ax.plot(x_filtered, y_casimir, marker='s', markevery=0.1, label='P. Casimir')
        # ax.plot(x_filtered, y_gaussian, '--', label='Gaussian')  # Uncomment as needed
        #ax.plot(x_filtered, y_capella, marker='^', markevery=0.1, label='S. Asperity')

        palette = sns.color_palette("viridis", n_colors=6)
        # Plot each line with a different color from the palette and updated labels
        ax.plot(x_filtered, y_capella_c1000000, marker='s', markevery=0.1, linestyle='--', color=palette[0], label=r'$N = 10^6$')
        ax.plot(x_filtered, y_capella_c10000000, marker='^', markevery=0.1, linestyle='-.', color=palette[1], label=r'$N = 10^7$')
        ax.plot(x_filtered, y_capella_c100000000, marker='*', markevery=0.1, linestyle=':', color=palette[2], label=r'$N = 10^8$')
        ax.plot(x_filtered, y_capella_c1000000000, marker='x', markevery=0.1, linestyle='-', color=palette[3], label=r'$N = 10^9$')
        ax.plot(x_filtered, y_capella_c10000000000, marker='+', markevery=0.1, linestyle='--', color=palette[4], label=r'$N = 10^{10}$')
        ax.plot(x_filtered, y_capella_c100000000000, marker='o', markevery=0.1, linestyle='-', color=palette[5], label=r'$N = 10^{11}$')

        #ax.plot([], [], ' ', label=f'R² = {r2:.3f}')

        xh = 200 # max(x)
        xl = 0
        # xh = plot_param_options[param_x]['hi']
        # xl = plot_param_options[param_x]['lo']
        ax.set_xlim([xl, xh])
        ax.set_xlabel(plot_param_options[param_x]['lbl'])

        yh = 0.2
        yl = -4.5
        # yh = plot_param_options[param_y]['hi']
        # yl = plot_param_options[param_y]['lo']
        ax.set_ylim([yl, yh])
        ax.set_ylabel(plot_param_options[param_y]['lbl'])
        ax.legend()

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-multi_fit_var-{}-vs-{}.png".format(
            param_y, param_x, run)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return

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
    x_std = np.std(x_data, axis=0)

    y_mean = np.mean(y_data, axis=0)
    y_std = np.std(y_data, axis=0)

    return x_mean, y_mean, y_std


def plot_data_multi_trace_poly(param_y="T", param_x="T", data_paths=None):
    """
    plot multi data trace from a file with std

    """
    log = logging.getLogger(__name__)
    log.debug("in")

    plot_param_options = {
        "Depth_(nm)": {'label': "Depth", 'hi': 4, 'lo': 0, 'lbl': "Depth(nm)"},
        "Load_(µN)": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Load(µN)"},
        "Time_(s)": {'label': "Time", 'hi': 300, 'lo': 0, 'lbl': "Time (sec)"},
        "Depth_(V)": {'label': "Depth", 'hi': 4, 'lo': 0, 'lbl': "Depth (V)"},
        "Load_(V)": {'label': "Force", 'hi': 400, 'lo': 0, 'lbl': "Load (V)"}
    }

    dfs = []
    for file in data_paths:
        print(f"loading...{file}")
        df = load_file(file)

        two_levels_up = os.path.abspath(os.path.join(file, "../../"))
        keyword = 'Air_Indent'

        # Search for a file that contains the keyword in its name
        file_found = False
        # Search for a file that contains the keyword in its name
        found_file_path = None
        for file_name in os.listdir(two_levels_up):
            if keyword in file_name:
                found_file_path = os.path.join(two_levels_up, file_name)
                break

        # Print the result
        if found_file_path:
            print(f"File found: {found_file_path}")
            df_air = load_file(found_file_path)

            # Assuming 'Load (µN)' is the column that needs normalization
            if 'Load_(µN)' in df.columns and 'Load_(µN)' in df_air.columns:
                # Interpolating the Air_Indent data to match experimental data if needed
                df_air_interpolated = df_air.reindex(df.index, method='nearest')

                # Subtracting air indentation load from experimental load
                df['Load_(µN)'] = df['Load_(µN)'] - df_air_interpolated['Load_(µN)']

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

    figsize = 4  # 12 has been default
    figdpi = 600
    hwratio = 16. / 9
    fig = plt.figure(figsize=(figsize * hwratio, figsize), dpi=figdpi)
    ax = fig.add_subplot(111)

    hold_x = []
    hold_y = []
    with sns.axes_style("darkgrid"):
        for df in dfs:
            lab =  df.meta.run
            file = df.meta.file
            df = df.iloc[np.linspace(0, len(df) - 1, num=int(len(df) * 0.1), dtype=int)]
            x = df[param_x]
            y = df[param_y] #- df[param_y].iloc[-1]
            #ax.plot(x,y, label=lab)
            #colors = np.linspace(0, 1, len(x))
            #ax.scatter(x,y, c=colors, cmap='coolwarm')#, label=lab)
            x = x.dropna()
            y = y.dropna()

            # Ensure both x and y have matching indices after dropping NaNs
            valid_index = x.index.intersection(y.index)
            x = x.loc[valid_index]
            y = y.loc[valid_index]

            min_index = y.idxmin()
            # Select the data from the start to the min_index
            #x = -x
            y = y
            min_index = np.argmin(y)

            # ax.plot(x, y)#, label=f"Experimental Trial: {df.meta.run}")
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
        df_combined['SEM'] = df_combined['STD'] / np.sqrt(df_combined.notna().sum(axis=1))

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
            xh = 160 # max(x)
            xl = 0
            xh = None
            xl = None
        elif param_x == "Depth_(nm)":
            xh = 0.1 # max(x)
            xl = -200
        else:
            xh = None
            xl = None

        ## loading models
        F1_x_data, F1_y_mean, F1_y_std  = load_model_files("out/Model/F1")
        FB_x_data, FB_y_mean, FB_y_std  = load_model_files("out/Model/FB")

        #Correcting
        F1_y_mean = F1_y_mean#*25
        FB_x_data = FB_x_data*-1

        # Define indices for markers
        marker_indices = np.linspace(0, len(F1_x_data) - 1, 10, dtype=int)

        # Plot vdW Adhesion with markers and line
        line1, = ax.plot(F1_x_data, F1_y_mean, label='vdW Adhesion (Exp)',
                         color='orange', linestyle='--', marker='o',
                         markevery=marker_indices)

        marker_indices = np.linspace(0, len(FB_x_data) - 1, 10, dtype=int)

        # Plot Contact Load with markers and line
        line2, = ax.plot(FB_x_data, FB_y_mean, label='Contact Load',
                         color='green', linestyle='--', marker='*',
                         markevery=marker_indices)

        FB_y_std = FB_y_std / 5
        F1_y_std = F1_y_std / 5

        # Uniform x-axis based on the combined range of F1_x_data and FB_x_data
        x_min = max(min(F1_x_data), min(FB_x_data))
        x_max = min(max(F1_x_data), max(FB_x_data))
        uniform_x = np.linspace(x_min, x_max, 100)

        # Interpolate F1_y_mean and FB_y_mean to the uniform x-axis
        F1_y_mean_interp = np.interp(uniform_x, F1_x_data, F1_y_mean)
        FB_y_mean_interp = np.interp(uniform_x, FB_x_data, FB_y_mean)

        # Add the arrays together
        for i in range(len(FB_y_mean_interp)):
            print(f"{FB_y_mean_interp[i]} + {F1_y_mean_interp[i]}")
        combined = FB_y_mean_interp + F1_y_mean_interp

        ax.plot(uniform_x, combined, label='Total Load', color='red',
                linestyle='-', marker='s', markevery=10)
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlim([xl, xh])
        ax.set_xlabel("Separation Distance ($nm$)", fontsize=14)

        # Customize x-axis ticks to display absolute values
        ax.set_xticks(np.linspace(-200, 0, 10))  # Define tick positions
        ax.set_xticklabels(np.abs(ax.get_xticks()).astype(int), fontsize=14)
        ax.set_ylim([-7.5, 2])
        ax.set_ylabel("Load ($\\mu N$)", fontsize=14)
        ax.legend(loc='lower left', fontsize=14)

        logging.info(sys._getframe().f_code.co_name)
        plot_fn = "out/plot-multi_fit_var-{}-vs-{}.png".format(
            param_y, param_x, run)
        logging.info("write plot to: {}".format(plot_fn))
        fig.savefig(plot_fn, bbox_inches='tight')
    return


def main():
    cmd_args = parse_command_line()
    setup_logging(cmd_args['verbosity'])
    plot_data_multi_trace_poly(param_y="Load_(µN)",
                               param_x="Depth_(nm)",
                               data_paths=cmd_args['input'])

if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
       print("exited")


