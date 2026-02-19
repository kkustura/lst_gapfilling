

# Land Surface Temperature (LST) Gapfilling

Pipeline for gap-filling missing values in Land Surface Temperature (LST) datasets using machine-learning–based reconstruction methods.
The repository provides tools for (i) downloading thermal (ECOSTRESS), meteorological (INCA) and optical (Sentinel-2) data, (ii) preparing input features and training predictive models, and (iii) generating spatially continuous LST images from incomplete satellite observations. For a complete description of the methodology, please refer to the official publication:

K. Kustura, D. Conti, M. Sammer, and M. Riffler, Harnessing Multi-Source Data and Deep Learning for High-Resolution Land Surface Temperature Gap-Filling Supporting Climate Change Adaptation Activities, Remote Sensing, 2025, 17(2), 318. https://doi.org/10.3390/rs17020318

---


## Overview of the scripts

## Main Scripts Overview

The main entry points are in the `src/` subdirectory:

- `main_data.py`: Data acquisition and preprocessing (ECOSTRESS, S2, INCA)
- `main_train.py`: Input data preparation and model training (work in progress)
- `main_predict.py`: (to be added) Model inference/prediction

### main_data.py
Handles downloading and preprocessing of all datasets. Key features:
- Supports ECOSTRESS, Sentinel-2, and INCA meteorological data
- Download and preprocess actions (resampling, reprojection)
- Flexible CLI: select dataset, action, date range, tiles, and resolution
- Requires configuration and secrets files (see below)

### main_train.py
Prepares input arrays and trains the CNN model. Key features:
- Generates timestamp lists for training/validation
- Prepares input arrays from preprocessed data
- Trains a pixelwise CNN (Keras/TensorFlow)
- Modular: supports different datasets, patch sizes, and training configs
- CLI: select actions (timestamps, arrays, train), tiles, and more

### main_predict.py
To be added: will handle model inference on new data.

---


## How do I get set up?


## Setup Instructions

1. **Clone the repository.**

	```bash
	git clone https://<USERNAME>@bitbucket.org/geoville/land_surface_temperature.git
	cd land_surface_temperature
	```

2. **Create and activate environment.**

	Use the provided `requirements.txt` to install dependencies (pip or conda):

	```bash
	conda create -n lst python=3.10
	conda activate lst
	pip install -r requirements.txt
	```

	> Note: Large datasets are used. Ensure sufficient disk space and memory.

3. **Configure credentials and settings.**

	- Copy or create your configuration files in `src/config/`:
	  - `config.yml`: main pipeline and dataset settings
	  - `secrets.ini`: credentials for data access (see below)

---

1. **Clone the repository.** Clone from the Bitbucket using your credentials.

```bash
git clone https://<USERNAME>@bitbucket.org/geoville/land_surface_temperature.git
```

2. <TBD!!!!-not there yet> **Create conda environment.** Use `requirements.txt` provided in the project directory to create a conda environment:

```bash
cd <project_directory>
conda create -n lst
conda activate lst
pip install -r requirements.txt
```


## Configuration Files

### config.yml
Defines data directories, dataset parameters, and training settings. Example fields:

```yaml
data_directory: output
inca_datasets: ['T2M', 'GL', 'RH2M', 'UU', 'VV']
s2_datasets: ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'SCL']
resolution: 70
year: 2025
month_range: [8, 8]
# ...
```

### secrets.ini
Required for authentication to data providers. **Sections to include:**

- `[s3]`: Copernicus Data Space S3 credentials (for Sentinel-2)
	- `access_key`, `secret_key`, `host_base`, `host_bucket`, etc.
- `[earthdata]`: NASA Earthdata credentials (for ECOSTRESS)
	- `username`, `password`

Example:
```ini
[s3]
access_key = <your-access-key>
secret_key = <your-secret-key>
host_base = eodata.dataspace.copernicus.eu
host_bucket = eodata.dataspace.copernicus.eu

[earthdata]
username = <your-username>
password = <your-password>
```

> **Important:**
> - The `[s3]` section is required for Sentinel-2 downloads. See the [Copernicus Data Space S3 API documentation](https://documentation.dataspace.copernicus.eu/APIs/S3.html) for details.
> - The `[earthdata]` section is required for ECOSTRESS downloads. Register at https://urs.earthdata.nasa.gov/ if you do not have an account.

---

## Data Download and Preprocessing

Run `main_data.py` with appropriate arguments. Example:

```bash
python src/main_data.py --config src/config/config.yml --dataset S2 --action download --start_date 2025-08-01 --end_date 2025-08-15 --tiles 32TPT --secrets src/config/secrets.ini
```

See `src/main_data.py --help` for all options.

## Model Training

Run `main_train.py` to generate timestamps, prepare arrays, and train the model:

```bash
python src/main_train.py --config src/config/config.yml --action timestamps arrays train --tiles 32TPT
```

---


