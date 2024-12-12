# Small_Scale_Contact
### Code DOI: [![Zenodo-Code-DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14426972.svg)](https://doi.org/10.5281/zenodo.14426972)
### Data DOI: [![Zenodo-Data-DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14422143.svg)](https://doi.org/10.5281/zenodo.14422143)

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

## Collect Data
Download the data from [Zenodo](https://zenodo.org/records/14422143) and unzip it into the `data` directory.

### Data Citation

If you use this data, please cite it using the following BibTeX entry:
```bibtex
@dataset{nakamura_2024_14422143,
  author       = {Nakamura, Matthew and
                  Heyes, Corrisa},
  title        = {Small Scale Force Dataset},
  month        = dec,
  year         = 2024,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.14422143},
  url          = {https://doi.org/10.5281/zenodo.14422143}
}
```

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

## Results From Paper
### Figure 3. Experimental Results vs Adhesion Model
![Load vs Depth (vdw only)](out/plot-Load_(µN)-vs-Depth_(nm)-vdw_only.png)
Graph represents 71 approaches to a nominally flat gold sample. The average detected load on the transducer pre-contact and its standard deviation are represented by a blue line and region, respectively. We compare this experimental data with two versions of the van der Waals adhesion model. The half-space model (w.r.t. Paper eq. 2) assuming perfectly flat, smooth surfaces is shown in pink diamonds. The asperities model with fixed radius R = 65 nm and normally distributed surface heights as a function of RMS roughness $\sigma$ (w.r.t. Paper eq. 4) is shown in purple triangles. This experimental data indicates a distinct error with respect to the pre-contact adhesion model we are attempting to validate.

### Figure 5. Experimental Results vs Adhesion Model + Contact
![Multi Fit Load vs Depth](out/plot-multi_fit_var-Load_(µN)-vs-Depth_(nm).png)
Experimental approach data from fig. 3 is again represented in blue. A model of asperity contact loading is shown in green stars following [Bhushan1998](https://link.springer.com/article/10.1023/A:1019186601445). The van der Waals approaching contact adhesion model (w.r.t. Paper eq. 4) applied to measured surface asperity heights is shown in yellow circles. Finally, the corrected total interaction load, accounting for the early contact of outlier asperities protruding from the nanoindenter probe, is shown in red squares.

### Code Citation

If you use this data, please cite it using the following BibTeX entry:
```bibtex
@software{matthew_nakamura_2024_14426972,
  author       = {Matthew Nakamura},
  title        = {nanosystemslab/Small\_Scale\_Contact: Initial
                   Release of Small Scale Contact Project Data
                   Processing (v1.0.0)
                  },
  month        = dec,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.14426972},
  url          = {https://doi.org/10.5281/zenodo.14426972}
}
```

## License
This project is licensed under the GPL-3.0-or-later License - see the [LICENSE](LICENSE) file for details.
