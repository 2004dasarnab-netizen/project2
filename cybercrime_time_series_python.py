# ================================================================
# EVOLUTION OF CYBER-CRIME IN INDIA: A TIME SERIES ANALYSIS
# Python reproduction of the University of Calcutta project
# Source period: 2002-2021
# ================================================================
# Install once:
# pip install pandas numpy matplotlib scipy statsmodels
#
# The national totals below are transcribed from Table 2 / page 28
# and the dataset table on page 8 of the supplied dissertation.
# The state/UT-level analysis expects a CSV named
# "state_cybercrime.csv" with columns:
# State/UT,2002,2003,...,2021
#
# Optional population analysis expects:
# population_india.csv with columns: Year,Population
# ================================================================

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import Holt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# -----------------------------
# 1. SETTINGS
# -----------------------------
OUTPUT = Path("cybercrime_outputs")
OUTPUT.mkdir(exist_ok=True)

YEARS = np.arange(2002, 2022)
CASES = np.array([
    808, 471, 347, 481, 453, 556, 464, 696, 1322, 2213,
    3477, 5693, 9622, 11592, 12317, 21796, 27248, 44546,
    50035, 52974
], dtype=float)

DATA = pd.DataFrame({"Year": YEARS, "CyberCrime": CASES})

# Zone definitions exactly following the dissertation's methodology.
ZONES = {
    "North": [
        "Himachal Pradesh", "Punjab", "Uttarakhand", "Uttar Pradesh",
        "Haryana", "Jammu & Kashmir", "Chandigarh", "Delhi UT"
    ],
    "South": [
        "Andhra Pradesh", "Karnataka", "Kerala", "Tamil Nadu", "Telangana",
        "Lakshadweep", "Puducherry", "A & N Islands"
    ],
    "West": [
        "Rajasthan", "Gujarat", "Goa", "Maharashtra", "D&N Haveli", "Daman & Diu"
    ],
    "East": ["Bihar", "Odisha", "Jharkhand", "West Bengal"],
    "North-East": [
        "Assam", "Sikkim", "Meghalaya", "Tripura", "Manipur",
        "Mizoram", "Nagaland", "Arunachal Pradesh"
    ],
    "Central": ["Madhya Pradesh", "Chhattisgarh"]
}

# -----------------------------
# 2. BASIC DATA DISPLAY
# -----------------------------
print("\nNATIONAL CYBERCRIME DATA")
print(DATA.to_string(index=False))

DATA.to_csv(OUTPUT / "national_cybercrime_2002_2021.csv", index=False)

# -----------------------------
# 3. FIGURE 1 - LINE PLOT
# -----------------------------
plt.figure(figsize=(10, 6))
plt.plot(DATA["Year"], DATA["CyberCrime"], marker="o")
plt.title("Cyber-Crime Cases in India (2002-2021)")
plt.xlabel("Year")
plt.ylabel("Number of Cyber-Crime Cases")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig1_national_line_plot.png", dpi=300)
plt.close()

# -----------------------------
# 4. OPTIONAL POPULATION / RATE ANALYSIS
# -----------------------------
# The dissertation describes a cybercrime-rate plot per 10,000 people.
# Because the exact population series is not printed in the dissertation's
# dataset table, the code reads it from population_india.csv instead of
# inventing values.

def population_analysis():
    path = Path("population_india.csv")
    if not path.exists():
        print("\n[INFO] population_india.csv not found.")
        print("       Skipping Fig.2 and Fig.3. Add Year,Population to enable them.")
        return

    pop = pd.read_csv(path)
    required = {"Year", "Population"}
    if not required.issubset(pop.columns):
        raise ValueError("population_india.csv must contain Year and Population columns")

    merged = DATA.merge(pop, on="Year", how="inner")
    # Population is assumed to be number of people.
    merged["Rate_per_10000"] = merged["CyberCrime"] / merged["Population"] * 10000

    plt.figure(figsize=(10, 6))
    plt.plot(merged["Year"], merged["Rate_per_10000"], marker="o")
    plt.title("Cyber-Crime Rate per 10,000 People in India (2002-2021)")
    plt.xlabel("Year")
    plt.ylabel("Cyber-Crime Cases per 10,000 People")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig2_cybercrime_rate.png", dpi=300)
    plt.close()

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(merged["Year"], merged["Population"])
    ax[0].set_ylabel("Population")
    ax[0].set_title("Population and Cyber-Crime Cases")
    ax[1].plot(merged["Year"], merged["CyberCrime"])
    ax[1].set_ylabel("Cyber-Crime Cases")
    ax[1].set_xlabel("Year")
    for a in ax:
        a.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig3_population_vs_cybercrime.png", dpi=300)
    plt.close()

    merged.to_csv(OUTPUT / "population_cybercrime_analysis.csv", index=False)

population_analysis()

# -----------------------------
# 5. ZONAL ANALYSIS
# -----------------------------

def clean_state_name(x):
    x = str(x).strip()
    replacements = {
        "Delhi": "Delhi UT",
        "A & N Islands": "A & N Islands",
        "Andaman & Nicobar Islands": "A & N Islands",
        "D & N Haveli": "D&N Haveli",
        "D&N Haveli": "D&N Haveli",
        "Daman & Diu": "Daman & Diu",
        "Dadra & Nagar Haveli": "D&N Haveli",
        "Pondicherry": "Puducherry",
        "Puducherry": "Puducherry",
    }
    return replacements.get(x, x)


def load_state_data():
    path = Path("state_cybercrime.csv")
    if not path.exists():
        print("\n[INFO] state_cybercrime.csv not found.")
        print("       Skipping Figures 4.1-4.7 (zonal analysis).")
        print("       Required format: State/UT,2002,2003,...,2021")
        return None

    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "State/UT"})
    df["State/UT"] = df["State/UT"].map(clean_state_name)

    year_cols = [str(y) for y in YEARS]
    missing = [c for c in year_cols if c not in df.columns]
    if missing:
        raise ValueError(f"state_cybercrime.csv is missing year columns: {missing}")

    for c in year_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def build_zonal_data(state_df):
    zone_rows = []
    for zone, states in ZONES.items():
        subset = state_df[state_df["State/UT"].isin(states)]
        row = {"Zone": zone}
        for year in YEARS:
            row[str(year)] = subset[str(year)].sum()
        zone_rows.append(row)
    wide = pd.DataFrame(zone_rows)
    long = wide.melt(id_vars="Zone", var_name="Year", value_name="Cases")
    long["Year"] = long["Year"].astype(int)
    return wide, long

state_df = load_state_data()
if state_df is not None:
    zonal_wide, zonal_long = build_zonal_data(state_df)
    zonal_long.to_csv(OUTPUT / "zonal_cybercrime.csv", index=False)

    for zone in ZONES:
        d = zonal_long[zonal_long["Zone"] == zone]
        plt.figure(figsize=(10, 5))
        plt.plot(d["Year"], d["Cases"], marker="o")
        plt.title(f"{zone} Zone Cyber-Crime Cases (2002-2021)")
        plt.xlabel("Year")
        plt.ylabel("Cyber-Crime Cases")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(OUTPUT / f"zonal_{zone.lower().replace('-', '_')}.png", dpi=300)
        plt.close()

    # Fig.4.7: proportion of total cybercrime in each zone
    prop = zonal_wide.set_index("Zone").T
    prop.index = prop.index.astype(int)
    prop["Total"] = prop.sum(axis=1)
    prop_pct = prop.drop(columns="Total").div(prop["Total"], axis=0) * 100
    prop_pct.to_csv(OUTPUT / "zonal_proportions.csv")

    plt.figure(figsize=(11, 6))
    for zone in ZONES:
        plt.plot(prop_pct.index, prop_pct[zone], marker="o", label=zone)
    plt.title("Proportion of Cyber-Crime Cases by Zone (2002-2021)")
    plt.xlabel("Year")
    plt.ylabel("Proportion (%)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / "fig4_7_zonal_proportion.png", dpi=300)
    plt.close()

# -----------------------------
# 6. MATHEMATICAL CURVE FITTING
# -----------------------------

t = DATA["Year"].to_numpy(dtype=float)
y = DATA["CyberCrime"].to_numpy(dtype=float)

# 6.1 Linear growth: Y = a + b*t
X_linear = np.column_stack([np.ones(len(t)), t])
beta_linear = np.linalg.lstsq(X_linear, y, rcond=None)[0]
yhat_linear = X_linear @ beta_linear
r2_linear = 1 - np.sum((y - yhat_linear) ** 2) / np.sum((y - y.mean()) ** 2)

print("\nLINEAR GROWTH MODEL")
print(f"a = {beta_linear[0]:.6f}")
print(f"b = {beta_linear[1]:.6f}")
print(f"R-squared = {r2_linear:.7f}")

plt.figure(figsize=(10, 6))
plt.scatter(t, y, label="Observed")
plt.plot(t, yhat_linear, linewidth=2, label="Linear fit")
plt.title("Linear Growth Model")
plt.xlabel("Year")
plt.ylabel("Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_9_linear_growth.png", dpi=300)
plt.close()

# 6.2 Quadratic growth: Y = a + b*t + c*t^2
X_quad = np.column_stack([np.ones(len(t)), t, t**2])
beta_quad = np.linalg.lstsq(X_quad, y, rcond=None)[0]
yhat_quad = X_quad @ beta_quad
r2_quad = 1 - np.sum((y - yhat_quad) ** 2) / np.sum((y - y.mean()) ** 2)

print("\nQUADRATIC GROWTH MODEL")
print(f"a = {beta_quad[0]:.6f}")
print(f"b = {beta_quad[1]:.6f}")
print(f"c = {beta_quad[2]:.6f}")
print(f"R-squared = {r2_quad:.7f}")

plt.figure(figsize=(10, 6))
plt.scatter(t, y, label="Observed")
plt.plot(t, yhat_quad, linewidth=2, label="Quadratic fit")
plt.title("Quadratic Growth Curve")
plt.xlabel("Year")
plt.ylabel("Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_11_quadratic_growth.png", dpi=300)
plt.close()

# 6.3 Exponential growth using log(Y) = A + b*t
log_y = np.log(y)
X_exp = sm_matrix = np.column_stack([np.ones(len(t)), t])
beta_exp = np.linalg.lstsq(X_exp, log_y, rcond=None)[0]
log_yhat = X_exp @ beta_exp
yhat_exp = np.exp(log_yhat)
r2_exp_log = 1 - np.sum((log_y - log_yhat) ** 2) / np.sum((log_y - log_y.mean()) ** 2)

# R-squared reported in the dissertation is the R-squared of the log-transformed regression.
print("\nEXPONENTIAL GROWTH MODEL")
print(f"log-intercept A = {beta_exp[0]:.8f}")
print(f"growth coefficient b = {beta_exp[1]:.8f}")
print(f"Model: Y = exp({beta_exp[0]:.8f} + {beta_exp[1]:.8f} * Year)")
print(f"R-squared on log(Y) = {r2_exp_log:.7f}")

plt.figure(figsize=(10, 6))
plt.scatter(t, y, label="Observed")
plt.plot(t, yhat_exp, linewidth=2, label="Exponential fit")
plt.title("Exponential Growth Model")
plt.xlabel("Year")
plt.ylabel("Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_13_exponential_growth.png", dpi=300)
plt.close()

curve_table = pd.DataFrame({
    "Year": YEARS,
    "Observed": y,
    "Linear_Predicted": yhat_linear,
    "Quadratic_Predicted": yhat_quad,
    "Exponential_Predicted": yhat_exp
})
curve_table.to_csv(OUTPUT / "curve_fitting_predictions.csv", index=False)

# -----------------------------
# 7. THREE-PERIOD MOVING AVERAGE
# -----------------------------
DATA["MA_3"] = DATA["CyberCrime"].rolling(window=3).mean()

plt.figure(figsize=(10, 6))
plt.plot(DATA["Year"], DATA["CyberCrime"], marker="o", label="Observed")
plt.plot(DATA["Year"], DATA["MA_3"], linewidth=2, label="3-period moving average")
plt.title("Cyber-Crime Cases: 3-Period Moving Average")
plt.xlabel("Year")
plt.ylabel("Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_14_three_period_moving_average.png", dpi=300)
plt.close()

# -----------------------------
# 8. AUGMENTED DICKEY-FULLER TEST
# -----------------------------
# The dissertation log-transforms the series before the ADF test and uses lag=1.

def run_adf(series, name):
    result = adfuller(series.dropna(), maxlag=1, autolag=None, regression="c")
    print(f"\nADF TEST: {name}")
    print(f"ADF statistic: {result[0]:.8f}")
    print(f"p-value: {result[1]:.8f}")
    print(f"used lag: {result[2]}")
    return result

log_series = pd.Series(np.log(y), index=YEARS)
first_diff = log_series.diff().dropna()
second_diff = first_diff.diff().dropna()

adf_original = run_adf(log_series, "log(CyberCrime)")
adf_first = run_adf(first_diff, "first difference of log(CyberCrime)")
adf_second = run_adf(second_diff, "second difference of log(CyberCrime)")

adf_table = pd.DataFrame([
    ["log(Y)", adf_original[0], adf_original[1], adf_original[2]],
    ["D(log(Y))", adf_first[0], adf_first[1], adf_first[2]],
    ["D2(log(Y))", adf_second[0], adf_second[1], adf_second[2]],
], columns=["Series", "ADF_Statistic", "p_value", "Lag"])
adf_table.to_csv(OUTPUT / "adf_results.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(second_diff.index, second_diff.values, marker="o")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Second-Differenced Log Cyber-Crime Series")
plt.xlabel("Year")
plt.ylabel("D2(log(CyberCrime))")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_16_stationary_series.png", dpi=300)
plt.close()

# -----------------------------
# 9. ACF AND PACF
# -----------------------------
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
plot_acf(second_diff, lags=7, ax=ax)
ax.set_title("ACF Plot - Stationary Series")
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_17_acf.png", dpi=300)
plt.close()

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
plot_pacf(second_diff, lags=7, ax=ax, method="ywm")
ax.set_title("PACF Plot - Stationary Series")
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_18_pacf.png", dpi=300)
plt.close()

# -----------------------------
# 10. ARIMA MODEL IDENTIFICATION
# -----------------------------
# The dissertation reports candidate models and AIC/BIC values.
# Here we fit the same candidate orders using statsmodels.

candidate_orders = [
    (0,2,1), (1,2,0), (1,2,1), (2,2,1), (1,2,2),
    (2,2,2), (3,2,1), (3,2,2), (3,2,3), (4,2,4)
]

arima_rows = []
for order in candidate_orders:
    try:
        model = ARIMA(y, order=order, trend="n").fit()
        arima_rows.append({
            "Model": f"ARIMA{order}",
            "p": order[0], "d": order[1], "q": order[2],
            "AIC": model.aic, "BIC": model.bic
        })
    except Exception as e:
        print(f"Could not fit ARIMA{order}: {e}")

arima_selection = pd.DataFrame(arima_rows).sort_values("AIC")
arima_selection.to_csv(OUTPUT / "arima_aic_bic_python.csv", index=False)
print("\nPYTHON ARIMA AIC/BIC RESULTS (statsmodels)")
print(arima_selection.to_string(index=False))

# AIC/BIC values printed in the supplied dissertation (Table 3).
# They are retained separately because different software/estimation
# conventions can produce slightly different information criteria.
reported_aic_bic = pd.DataFrame([
    ["ARIMA(0,2,1)", 353.5963, 355.3777],
    ["ARIMA(1,2,0)", 354.9853, 357.6564],
    ["ARIMA(1,2,1)", 353.0588, 354.8395],
    ["ARIMA(2,2,1)", 356.6817, 360.2432],
    ["ARIMA(1,2,2)", 356.8899, 360.4513],
    ["ARIMA(2,2,2)", 358.5747, 363.0265],
    ["ARIMA(3,2,1)", 358.0466, 362.4984],
    ["ARIMA(3,2,2)", 359.1973, 364.5395],
    ["ARIMA(3,2,3)", 357.3975, 363.6301],
    ["ARIMA(4,2,4)", 359.8691, 367.8825],
], columns=["Model", "AIC_reported", "BIC_reported"])
reported_aic_bic.to_csv(OUTPUT / "table3_aic_bic_from_dissertation.csv", index=False)

# -----------------------------
# 11. TRAIN / TEST SPLIT: 2002-2013 / 2014-2021
# -----------------------------
train_mask = YEARS <= 2013
test_mask = YEARS >= 2014
train = y[train_mask]
test = y[test_mask]
test_years = YEARS[test_mask]

plt.figure(figsize=(10, 5))
plt.plot(YEARS[train_mask], train, marker="o", label="Training data")
plt.plot(YEARS[test_mask], test, marker="o", label="Testing data")
plt.axvline(2013.5, linestyle="--", linewidth=1)
plt.title("Training and Testing Split")
plt.xlabel("Year")
plt.ylabel("Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_19_train_test_split.png", dpi=300)
plt.close()

# -----------------------------
# 12. ERROR METRICS
# -----------------------------
def error_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    return {
        "ME": np.mean(err),
        "RMSE": np.sqrt(np.mean(err**2)),
        "MAE": np.mean(np.abs(err))
    }

# -----------------------------
# 13. TEST FORECASTS FOR THE THREE MAIN ARIMA MODELS
# -----------------------------
arima_test_results = []

for order, filename in [
    ((1,2,1), "fig4_20_arima_121_test.png"),
    ((0,2,1), "fig4_21_arima_021_test.png"),
    ((1,2,0), "fig4_22_arima_120_test.png")
]:
    model = ARIMA(train, order=order, trend="n").fit()
    forecast_obj = model.get_forecast(steps=len(test))
    pred = forecast_obj.predicted_mean
    ci = np.asarray(forecast_obj.conf_int(alpha=0.20))
    metrics = error_metrics(test, pred)
    arima_test_results.append({"Model": f"ARIMA{order}", **metrics})

    plt.figure(figsize=(10, 6))
    plt.plot(YEARS[train_mask], train, label="Train")
    plt.plot(test_years, test, marker="o", label="Actual test")
    plt.plot(test_years, pred, marker="o", label=f"ARIMA{order} forecast")
    plt.fill_between(test_years, ci[:,0], ci[:,1], alpha=0.20)
    plt.title(f"ARIMA{order} Forecast on Test Data")
    plt.xlabel("Year")
    plt.ylabel("Cyber-Crime Cases")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT / filename, dpi=300)
    plt.close()

arima_test_table = pd.DataFrame(arima_test_results)
arima_test_table.to_csv(OUTPUT / "arima_test_errors.csv", index=False)
print("\nARIMA TEST ERRORS")
print(arima_test_table.to_string(index=False))

# -----------------------------
# 14. HOLT'S LINEAR EXPONENTIAL SMOOTHING
# -----------------------------
holt_model = Holt(train, initialization_method="estimated", damped_trend=False).fit(optimized=True)
holt_pred = holt_model.forecast(len(test))
holt_metrics = error_metrics(test, holt_pred)

plt.figure(figsize=(10, 6))
plt.plot(YEARS[train_mask], train, label="Train")
plt.plot(test_years, test, marker="o", label="Actual test")
plt.plot(test_years, holt_pred, marker="o", label="Holt forecast")
plt.title("Holt's Linear Exponential Smoothing Forecast")
plt.xlabel("Year")
plt.ylabel("Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_23_holt_test.png", dpi=300)
plt.close()

holt_table = pd.DataFrame([{"Model": "Holt Linear", **holt_metrics}])
holt_table.to_csv(OUTPUT / "holt_test_errors.csv", index=False)

# -----------------------------
# 15. SELECTED MODEL: ARIMA(1,2,1)
# -----------------------------
# The dissertation selects ARIMA(1,2,1), citing the lowest AIC/BIC among
# its candidate models and the test-set comparison.
final_model = ARIMA(y, order=(1,2,1), trend="n").fit()
print("\nFINAL ARIMA(1,2,1) MODEL SUMMARY")
print(final_model.summary())

# -----------------------------
# 16. 15-YEAR FORECAST: 2022-2036
# -----------------------------
forecast_obj = final_model.get_forecast(steps=15)
forecast_mean = forecast_obj.predicted_mean
forecast_ci_80 = np.asarray(forecast_obj.conf_int(alpha=0.20))
forecast_ci_95 = np.asarray(forecast_obj.conf_int(alpha=0.05))
future_years = np.arange(2022, 2037)

forecast_table = pd.DataFrame({
    "Year": future_years,
    "Point_Forecast": forecast_mean,
    "Lower_80": forecast_ci_80[:, 0],
    "Upper_80": forecast_ci_80[:, 1],
    "Lower_95": forecast_ci_95[:, 0],
    "Upper_95": forecast_ci_95[:, 1],
})
forecast_table.to_csv(OUTPUT / "table4_forecast_2022_2036.csv", index=False)

plt.figure(figsize=(11, 6))
plt.plot(YEARS, y, marker="o", label="Observed")
plt.plot(future_years, forecast_mean, marker="o", label="ARIMA(1,2,1) forecast")
plt.fill_between(
    future_years,
    forecast_ci_95[:,0],
    forecast_ci_95[:,1],
    alpha=0.12,
    label="95% confidence interval"
)
plt.fill_between(
    future_years,
    forecast_ci_80[:,0],
    forecast_ci_80[:,1],
    alpha=0.20,
    label="80% confidence interval"
)
plt.axvline(2021, linestyle="--", linewidth=1)
plt.title("Forecast of Cyber-Crime Cases in India: 2022-2036")
plt.xlabel("Year")
plt.ylabel("Number of Cyber-Crime Cases")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT / "fig4_24_final_15_year_forecast.png", dpi=300)
plt.close()

# -----------------------------
# 17. REPORTED RESULTS FROM THE DISSERTATION
# -----------------------------
# These values reproduce the dissertation's printed Table 4 exactly.
# They are kept separately from the fresh statsmodels forecast because
# different software/estimation settings can give different forecasts.
reported_forecast = pd.DataFrame({
    "Year": list(range(2022, 2037)),
    "Point_Forecast": [59480.24, 65599.96, 71761.56, 77918.63, 84076.18,
                       90233.68, 96391.19, 102548.70, 108706.21, 114863.72,
                       121021.22, 127178.73, 133336.24, 139493.75, 145651.25],
    "Lower_80": [54518.346, 57303.126, 59745.351, 61831.504, 63577.546,
                  65002.011, 66122.577, 66955.001, 67513.165, 67809.304,
                  67854.258, 67657.684, 67228.248, 66573.771, 65701.354],
    "Upper_80": [64442.13, 73896.80, 83777.77, 94005.75, 104574.82,
                  115465.36, 126659.81, 138142.40, 149899.25, 161918.13,
                  174188.19, 186699.78, 199444.23, 212413.72, 225601.15],
    "Lower_95": [51891.678, 52911.047, 53384.353, 53315.495, 52726.224,
                  51645.169, 50099.340, 48112.835, 45706.885, 42900.204,
                  39709.366, 36149.145, 32232.791, 27972.268, 23378.433],
    "Upper_95": [67068.80, 78288.88, 90138.77, 102521.76, 115426.14,
                  128822.20, 142683.05, 156984.57, 171705.53, 186827.23,
                  202333.08, 218208.32, 234439.69, 251015.22, 267924.07]
})
reported_forecast.to_csv(OUTPUT / "table4_reported_from_dissertation.csv", index=False)

# Reported test-set errors from the dissertation (pages 35-39).
reported_errors = pd.DataFrame([
    ["ARIMA(1,2,1)", 68.38299, 268.9753, 187.4139],
    ["ARIMA(0,2,1)", 171.5175, 330.7058, 257.7731],
    ["ARIMA(1,2,0)", 112.4893, 294.8925, 220.9235],
    ["Holt Linear", 125.0424, 468.1245, 359.9674]
], columns=["Model", "ME_reported", "RMSE_reported", "MAE_reported"])
reported_errors.to_csv(OUTPUT / "reported_test_errors_from_dissertation.csv", index=False)

# -----------------------------
# 17. PROJECT RESULTS SUMMARY
# -----------------------------
summary = pd.DataFrame({
    "Model": ["Linear", "Quadratic", "Exponential (log-linear)", "ARIMA(1,2,1)", "Holt Linear"],
    "R2_or_metric": [r2_linear, r2_quad, r2_exp_log, np.nan, np.nan],
    "RMSE_test": [np.nan, np.nan, np.nan,
                   arima_test_table.loc[arima_test_table["Model"]=="ARIMA(1, 2, 1)", "RMSE"].iloc[0]
                   if any(arima_test_table["Model"]=="ARIMA(1, 2, 1)") else np.nan,
                   holt_metrics["RMSE"]]
})
summary.to_csv(OUTPUT / "project_model_summary.csv", index=False)

# -----------------------------
# 18. SAVE ALL KEY RESULTS
# -----------------------------
with open(OUTPUT / "model_equations.txt", "w", encoding="utf-8") as f:
    f.write("LINEAR MODEL\n")
    f.write(f"Y = {beta_linear[0]:.8f} + {beta_linear[1]:.8f} * Year\n")
    f.write(f"R-squared = {r2_linear:.8f}\n\n")
    f.write("QUADRATIC MODEL\n")
    f.write(f"Y = {beta_quad[0]:.8f} + {beta_quad[1]:.8f} * Year + {beta_quad[2]:.8f} * Year^2\n")
    f.write(f"R-squared = {r2_quad:.8f}\n\n")
    f.write("EXPONENTIAL MODEL (log-linear form)\n")
    f.write(f"log(Y) = {beta_exp[0]:.8f} + {beta_exp[1]:.8f} * Year\n")
    f.write(f"Y = exp({beta_exp[0]:.8f} + {beta_exp[1]:.8f} * Year)\n")
    f.write(f"R-squared on log(Y) = {r2_exp_log:.8f}\n\n")
    f.write("SELECTED ARIMA MODEL\n")
    f.write("ARIMA(1,2,1)\n")
    f.write("(1 - Phi1 B)(1 - B)^2 Y_t = (1 + theta1 B)e_t\n")

print("\n============================================================")
print("PROJECT ANALYSIS COMPLETE")
print(f"All generated files are in: {OUTPUT.resolve()}")
print("============================================================")
