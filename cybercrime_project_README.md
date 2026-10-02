# Cybercrime Time-Series Analysis in India

## Project Overview

This project analyzes the yearly number of cybercrime cases reported in India and uses time-series forecasting techniques to study the trend and forecast future cases.

The project follows the methodology described in the original dissertation:

- Graphical representation of cybercrime cases
- Linear growth model
- Quadratic growth model
- Exponential/log model
- Moving-average analysis
- Augmented Dickey-Fuller (ADF) test
- ACF and PACF analysis
- ARIMA model fitting
- 60:40 training/testing evaluation
- Holt Linear Exponential Smoothing
- Model comparison using forecast errors
- Forecasting cybercrime cases for 2022–2036

## Dataset

The national dataset contains annual cybercrime case counts for India from **2002 to 2021**.

| Year | Cases |
|---:|---:|
| 2002 | 808 |
| 2003 | 471 |
| 2004 | 347 |
| 2005 | 481 |
| 2006 | 453 |
| 2007 | 556 |
| 2008 | 464 |
| 2009 | 696 |
| 2010 | 1322 |
| 2011 | 2213 |
| 2012 | 3477 |
| 2013 | 5693 |
| 2014 | 9622 |
| 2015 | 11592 |
| 2016 | 12317 |
| 2017 | 21796 |
| 2018 | 27248 |
| 2019 | 44546 |
| 2020 | 50035 |
| 2021 | 52974 |

The project also contains templates for population and state-level data. These should be replaced with the corresponding source data if state-wise or population-adjusted analysis is required.

## Project Structure

```text
cybercrime_project_python/
│
├── cybercrime_time_series_python.py
├── national_cybercrime.csv
├── population_india_template.csv
├── state_cybercrime_template.csv
├── requirements.txt
└── README.md
```

## Requirements

Python 3.9 or newer is recommended.

Install the required packages:

```bash
pip install -r requirements.txt
```

The main packages used are:

- pandas
- numpy
- matplotlib
- statsmodels
- scipy
- scikit-learn

## How to Run

### 1. Open the project in VS Code

Open the project folder:

```text
cybercrime_project_python
```

### 2. Open the VS Code terminal

Run:

```bash
pip install -r requirements.txt
```

### 3. Run the Python program

```bash
python cybercrime_time_series_python.py
```

The program reads `national_cybercrime.csv`, performs the analysis, fits the time-series models, evaluates them, and generates forecasts and plots.

## Methodology

### 1. Exploratory Analysis

The yearly cybercrime cases are plotted against time to examine the overall growth pattern.

The project considers:

- Linear growth
- Quadratic growth
- Exponential/logarithmic growth

### 2. Moving Average

A **3-period moving average** is used to smooth short-term fluctuations and identify the underlying trend.

### 3. Stationarity Testing

The Augmented Dickey-Fuller (ADF) test is used to determine whether the time series is stationary.

The dissertation reports:

| Series | ADF p-value |
|---|---:|
| Log original series | 0.3452 |
| First difference | 0.4013 |
| Second difference | 0.01983 |

Since the second-differenced series has a p-value below 0.05, the analysis uses:

```text
d = 2
```

### 4. ACF and PACF

The ACF and PACF plots are examined to identify possible ARIMA parameters.

The dissertation identifies the strongest relevant lag as approximately lag 1 for both ACF and PACF, leading to:

```text
p = 1
q = 1
```

Therefore, the selected model is:

```text
ARIMA(1, 2, 1)
```

### 5. Train/Test Split

The data are divided chronologically:

```text
Training period: 2002–2013
Testing period: 2014–2021
```

This gives a 60:40 split.

The model is fitted using the training period and evaluated against the observed testing period.

## Model Comparison

The dissertation reports the following test-set errors:

| Model | ME | RMSE | MAE |
|---|---:|---:|---:|
| ARIMA(1,2,1) | 68.38299 | 268.9753 | 187.4139 |
| ARIMA(0,2,1) | 171.5175 | 330.7058 | 257.7731 |
| ARIMA(1,2,0) | 112.4893 | 294.8925 | 220.9235 |
| Holt Linear | 125.0424 | 468.1245 | 359.9674 |

The project uses ARIMA(1,2,1) as the selected forecasting model, consistent with the dissertation.

## ARIMA Model

The model is represented as:

```text
(1 - Φ₁B)(1 - B)²Y(t) = (1 + θ₁B)e(t)
```

where:

- `B` = backshift operator
- `Φ₁` = AR parameter
- `θ₁` = MA parameter
- `d = 2` = order of differencing
- `e(t)` = error term

The reported AIC and BIC for ARIMA(1,2,1) are:

```text
AIC = 353.0588
BIC = 354.8395
```

## Forecast

The selected ARIMA model is used to forecast cybercrime cases from **2022 to 2036**.

Reported forecasts:

| Year | Forecast |
|---:|---:|
| 2022 | 59480.24 |
| 2023 | 65599.96 |
| 2024 | 71761.56 |
| 2025 | 77918.63 |
| 2026 | 84076.18 |
| 2027 | 90233.68 |
| 2028 | 96391.19 |
| 2029 | 102548.70 |
| 2030 | 108706.21 |
| 2031 | 114863.72 |
| 2032 | 121021.22 |
| 2033 | 127178.73 |
| 2034 | 133336.24 |
| 2035 | 139493.75 |
| 2036 | 145651.25 |

For 2024, the reported prediction intervals are:

```text
95% PI: [53384.353, 90138.77]
80% PI: [59745.351, 83777.77]
```

## Output

After execution, the script generates analysis outputs such as:

- Time-series plots
- Growth-model plots
- Moving-average plot
- ADF test results
- ACF plot
- PACF plot
- ARIMA model results
- Train/test forecast comparison
- Holt model results
- Model error comparison
- Future forecast
- Forecast interval plots

The generated files are saved in the project's output directory.

## Important Note About Reproducibility

The original dissertation provides the national yearly observations and reported model results, but some supporting datasets referenced for population and state-level analysis are not included as complete raw datasets in the dissertation.

Therefore:

1. `national_cybercrime.csv` contains the national series used for the time-series analysis.
2. The population and state-level CSV files are templates.
3. Exact numerical results may vary slightly depending on the Python/statsmodels version and model-fitting implementation.
4. The reported dissertation values are included in this README for reference; the Python script performs the calculations rather than simply hard-coding those results.

## Author / Project Context

This Python implementation is a computational version of the academic project on:

**Time-Series Analysis of Cybercrime Cases in India**

The implementation is intended for academic learning, analysis, visualization, and forecasting.
