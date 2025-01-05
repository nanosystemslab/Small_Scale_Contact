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
    parser.add_argument("--plot_stats", action="store_true", 
                        help="plot statistics")
    ret = vars(parser.parse_args())
    ret["verbosity"] = max(0, 30 - 10 * ret["verbosity"])
    return ret


def load_file(filepath):
    df = pd.read_csv(filepath, encoding='unicode_escape',
                     sep='\t', skiprows=(0, 1, 2))
    logging.info(df)
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


def calc_stats(df, exp: str, model:str, plot=False):
    working_df = df.copy()

    # Calculate Percentage Error
    working_df['Percentage_Error'] = (
        (working_df[model] - working_df[exp]).abs()
        / (working_df[exp])
    ) * 100

    # Calculate Percentage Difference
    working_df['Percentage_Difference'] = (
        (working_df[model] - working_df[exp]).abs()
        / ((working_df[model] + working_df[exp]) / 2)
    ) * 100

    # Calculate Root Mean Square Error (RMSE)
    working_df['Squared_Error'] = (working_df[exp] - working_df[model]) ** 2
    rmse = (working_df['Squared_Error'].mean()) ** 0.5

    # Calculate Mean Absolute Error (MAE)
    working_df['Absolute_Error'] = (working_df[model] - working_df[exp]).abs()
    mae = working_df['Absolute_Error'].mean()

    # Add RMSE, MAE, and R² to the DataFrame as metadata
    working_df['RMSE'] = rmse
    working_df['MAE'] = mae

    # Add formatted statements
    result_summary = (
        f"For comparing {exp} vs {model}:\n"
        f"Mean Percentage Error: {working_df['Percentage_Error'].mean():.2f}\n"
        f"Mean Percentage Difference: {working_df['Percentage_Difference'].mean():.2f}\n"
        f"Root Mean Square Error (RMSE): {working_df['RMSE'].mean():.2f}\n"
        f"Mean Absolute Error (MAE): {working_df['MAE'].mean():.2f}\n"
    )
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(working_df.index, working_df['Percentage_Difference'],
                 label='Percentage Difference', color='blue', linewidth=1.5)
        plt.scatter(working_df.index, working_df['Percentage_Error'],
                 label='Percentage Error', color='red', linewidth=1.5)
        plt.axhline(0, color='gray', linestyle='--', linewidth=0.8, label='Zero Line')
        plt.title(f'Percentage Difference vs X for {exp} and {model}', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Percentage Difference (%)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    return result_summary

def plot_data_multi_trace_poly(param_y="T", param_x="T", data_paths=None, plot_stats=False):
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
        logging.info(f"loading...{file}")
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
            logging.info(f"File found: {found_file_path}")
            df_air = load_file(found_file_path)

            # Assuming 'Load (µN)' is the column that needs normalization
            if 'Load_(µN)' in df.columns and 'Load_(µN)' in df_air.columns:
                # Interpolating the Air_Indent data to match experimental data if needed
                df_air_interpolated = df_air.reindex(df.index, method='nearest')

                # Subtracting air indentation load from experimental load
                df['Load_(µN)'] = df['Load_(µN)'] - df_air_interpolated['Load_(µN)']

                # Print or save the normalized dataframe
                logging.info(df.head())  # For testing, you can see the top few rows
            else:
                logging.warning("Load (µN) column not found in data files")
        else:
            logging.warning("File containing the keyword does not exist not normalizing")

        run = df.meta.run

        df = df[df['Time_(s)'] >= 5]
        zero_norm = df[(df['Time_(s)'] >= 5) & (df['Time_(s)'] <= 50)]
        zero_norm = zero_norm['Load_(µN)'].mean()
        df['Load_(µN)'] = df['Load_(µN)'] - zero_norm
        df = df[df['Time_(s)'] >= 50]

        df.meta = types.SimpleNamespace()
        df.meta.run = run
        df.meta.file = file
        logging.info(df)
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

        # Experimental Data
        exp_df = pd.DataFrame({
            'load_average': load_average
        })
        # Get the minimum and maximum index values from combined_df
        min_index_combined = -200
        max_index_combined =  -12

        # Filter exp_df to keep rows with indices within the range of combined_df
        filtered_exp_df = exp_df[(exp_df.index >= min_index_combined) & (exp_df.index <= max_index_combined)]
        min_index_value = filtered_exp_df.loc[min(filtered_exp_df.index), 'load_average']
        filtered_exp_df = filtered_exp_df['load_average'] - min_index_value

        # Set x as x from exp
        uniform_x = filtered_exp_df.index

        # Interpolate F1_y_mean and FB_y_mean to the uniform x-axis
        F1_y_mean_interp = np.interp(uniform_x, F1_x_data, F1_y_mean)
        FB_y_mean_interp = np.interp(uniform_x, FB_x_data, FB_y_mean)
        combined = FB_y_mean_interp + F1_y_mean_interp
        combined_df = pd.DataFrame({
            'Uniform_X': uniform_x,
            'Combined': combined,
            'vdw_model': F1_y_mean_interp
        })
        combined_df.set_index('Uniform_X', inplace=True)
        combined_df.index.name = 'X'


        # Display the filtered DataFrame
        combined_result = pd.merge(
            combined_df,
            filtered_exp_df,
            left_index=True,
            right_index=True,
            how='inner'  # Use 'inner' to include only rows with matching indices
        )

        # Generate Statistics
        vdw_model_stats = calc_stats(combined_result, exp = "load_average", model = "vdw_model", plot=plot_stats)
        com_model_stats = calc_stats(combined_result, exp = "load_average", model = "Combined", plot=plot_stats)
        logging.info(vdw_model_stats)
        logging.info(com_model_stats)

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
                               data_paths=cmd_args['input'],
                               plot_stats=cmd_args['plot_stats'])

if "__main__" == __name__:
    try:
        sys.exit(main())
    except KeyboardInterrupt:
       print("exited")


