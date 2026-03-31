# DACNet User Guide

## 0. Installation and Quick Start

### 0.1 Environment Setup

It is recommended to use **Python 3.9+**.
Install the required packages with:

```bash
pip install -r requirements.txt
```

> Please install the CUDA-compatible PyTorch build that matches your environment before running the command above.

### 0.2 Quick Start

#### Run all datasets

```bash
python run_all_experiments.py --root_path /path/to/datasets
```

#### Run a single dataset

```bash
python run_single_experiments.py --root_path /path/to/datasets --dataset ecl --gpu 0
```


## 1. Model-Related Parameters

The experiment settings in this project are mainly controlled by `run.py` and the files in `configs/*.yml`.
Each dataset has its own configuration file in the `configs` folder, for example: `configs/ecl.yml` and `configs/traffic.yml`.

Each configuration file usually contains two parts:

- `fixed`: dataset-specific fixed parameters
- `pred_len_configs`: internal parameters selected according to the external `pred_len`

### 1.1 Example Configuration Structure

```yaml
fixed:
  root_path: ./datasets/
  data_path: ECL.csv
  model: DACNet_In
  model_id: ECL
  data: custom
  enc_in: 321
  cycle: 168
  learning_rate: 0.01
  lradj: type4
  sim_mode: l1
  mix: 0
  seq_len: 96
  backbone: mlp

pred_len_configs:
  96:
    use_norm: 1
    D_cp: 16
    D_de: 16
    D_mix: 8
    d_model: 512
```

### 1.2 Main Parameter Descriptions

| Parameter | Description | Example / Options |
|---|---|---|
| `root_path` | Root directory of the dataset | `./datasets/` |
| `data_path` | Data file name | `ECL.csv`, `Traffic.csv` |
| `model` | Model name | `DACNet_In` |
| `model_id` | Experiment or dataset identifier | `ECL`, `ETTh1` |
| `data` | Dataset loading type | `custom`, `ETTh1`, `Solar` |
| `enc_in` | Input feature dimension / number of channels | 7, 21, 321, 862 |
| `cycle` | Period length of the sequence | 24, 96, 144, 168 |
| `learning_rate` | Learning rate | `0.005`, `0.01` |
| `lradj` | Learning rate adjustment strategy | `type1`, `type4` |
| `sim_mode` | Similarity computation mode | `l1`, `l2`, `dot`, `cosine` |
| `mix` | Whether to enable the mixing module | `0`: off, `1`: on |
| `seq_len` | Input historical sequence length | `96` |
| `backbone` | Backbone network type | `mlp`, `linear` |
| `pred_len` | Prediction length | `96`, `192`, `336`, `720` |
| `use_norm` | Normalization type | `1`, `2` |
| `D_cp` | Compression module dimension | e.g. `16` |
| `D_de` | Denoising module dimension | e.g. `16`, `32` |
| `D_mix` | Mixing module dimension | e.g. `8` |
| `d_model` | Hidden dimension of the model | `64`, `256`, `512`, `1024` |
| `loss` | Loss function | `mae`, `mse` |
| `gpu` | GPU device id | `0` |
| `random_seed` | Random seed | `2024`, `2025` |

> If you want to adjust parameters for a specific dataset under different prediction lengths, directly edit the corresponding `.yml` file in the `configs` directory.

---

## 2. One-Click Running Scripts

### 2.1 `run_all_experiments.py`: Run All Datasets

This script will:

1. Traverse all dataset configuration files in the `configs/` folder;
2. Read the `fixed` parameters for each dataset;
3. Select the corresponding `pred_len_configs[pred_len]` based on `PRED_LEN_LIST` in the script;
4. Merge the parameters and call `run.py` for training and testing;
5. Summarize the results into the output result files.

#### Usage

```bash
python run_all_experiments.py --root_path /path/to/datasets
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `--root_path` | Yes | Root directory of the datasets |

> It is recommended to always pass `root_path` explicitly to ensure the program uses the correct data directory.

---

### 2.2 `run_single_experiments.py`: Run a Specific Dataset

This script is used to run experiments on a single dataset.
It automatically reads `configs/{dataset}.yml` according to the input `dataset`, then selects the corresponding internal parameters together with the configured prediction length, and finally calls `run.py`.

At the same time, the script runs the experiment under two loss settings:

- `loss = mae`
- `loss = mse`

In other words, each execution will call `run.py` **twice**.

#### Usage

```bash
python run_single_experiments.py --root_path /path/to/datasets --dataset ecl
```

To specify a GPU, you can additionally pass:

```bash
python run_single_experiments.py --root_path /path/to/datasets --dataset ecl --gpu 0
```

#### Arguments

| Argument | Required | Description |
|---|---|---|
| `--root_path` | Yes | Root directory of the datasets |
| `--dataset` | Yes | Dataset name, such as `ecl`, `etth1`, `traffic` |
| `--gpu` | No | GPU device id, default is `0` |

---

## 3. Configuration Files

All dataset-related parameter files are stored in:

```bash
configs/
```

Currently included:

- `ecl.yml`
- `etth1.yml`
- `etth2.yml`
- `ettm1.yml`
- `ettm2.yml`
- `exchange.yml`
- `solar.yml`
- `traffic.yml`
- `weather.yml`

If needed, you can directly edit the corresponding `.yml` file to:

- change the default hyperparameters of a dataset;
- adjust `d_model`, `D_cp`, `D_de`, `D_mix`, `use_norm`, and other settings for different `pred_len` values;
- switch settings such as `backbone`, `mix`, and `learning_rate`.

---

## 4. Running Suggestions

- First make sure the dataset directory structure is correct, and pass the data root directory explicitly with `--root_path`;
- Use `run_all_experiments.py` for batch experiments;
- Use `run_single_experiments.py` for single-dataset tuning;
- Prefer editing parameters in `configs/*.yml` for easier unified management and experiment reproducibility.
