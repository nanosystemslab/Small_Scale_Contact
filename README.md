# Small_Scale_Contact

## Overview
**Small_Scale_Contact** is a project developed by the Nano Systems Lab to study and simulate the contact mechanics at a small scale. This repository contains the codebase and associated files for performing various simulations and analyses on nano-scale contact points.

## Directory Structure
```
Small_Scale_Contact/
├── LICENSE
├── README.md
├── data
│   ├── Adhesion-Experiment 
│   ├── NI-TIP-surface/
├── out
│   ├── Model/
│   ├── plot-Load_(µN)-vs-Depth_(nm)-vdw_only.png
│   └── plot-multi_fit_var-Load_(µN)-vs-Depth_(nm).png
├── output.txt
├── poetry.lock
├── pyproject.toml
└── src
    ├── gen_tip_surface_data-contact.py
    ├── gen_tip_surface_data_vdw.py
    ├── plot_model-vdw_only.py
    └── plot_model_vdw_and_contact.py
```

## Features
- **Contact Mechanics Simulation**: Analyze and visualize contact points at nano-scale.
- **Material Properties**: Customize material properties for different simulations.
- **Data Export**: Export simulation results in various formats for further analysis.

## Installation
### Collect Repository
To clone the repository: 
```sh
git clone https://github.com/nanosystemslab/Small_Scale_Contact.git
cd Small_Scale_Contact
```

### Using pip
install dependencies using pip:
```sh
pip install .
```

### Using Poetry
Alternatively, if you prefer using Poetry for dependency management, you can run:
```sh
poetry install
```

## Using a Poetry Shell
To start a new shell session with your project's dependencies, use the following command:
```sh
poetry shell
```
This will activate a new shell where all the dependencies specified in your `pyproject.toml` are available.

## Usage
1. **Prepare Input Data**: Ensure your input data files are correctly formatted and placed in the `data` directory.
2. **Run Simulations**: Use the scripts in the `src` directory to generate surface data and plot models.
```sh
python3 src/gen_tip_surface_data-contact.py -i data/NI-TIP-surface/NI-Tip-1.asc 
python3 src/gen_tip_surface_data_vdw.py -i data/NI-TIP-surface/NI-Tip-1.asc 
python3 src/plot_model-vdw_only.py -i data/Au_2024-10-24/2024-10-*txt
python3 src/plot_model_vdw_and_contact.py -i data/Au_2024-10-24/2024-10-*txt
```
3. **View Results**: Access the output files in the `out` directory and analyze the generated data.

## Results
1. Results of Adhesion Model
![Load vs Depth (vdw only)](out/plot-Load_(µN)-vs-Depth_(nm)-vdw_only.png)

2. Experimental Results
![Multi Fit Load vs Depth](out/plot-multi_fit_var-Load_(µN)-vs-Depth_(nm).png)


## License
This project is licensed under the GPL-3.0-or-later License - see the [LICENSE](LICENSE) file for details.
