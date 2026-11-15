from turtle import width
from wsgiref import types

import astropy as astr
import astropy.units as u
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def setup_lc_data(filename):
    with astr.io.fits.open(filename) as Raw_data:
        data = Raw_data[1].data
        df = pd.DataFrame(data)
    return df


def plot_light_curve(lc):
    lc.plot(label='PDCSAP Flux')
    plt.title("TESS Light Curve")
    plt.xlabel("Time (BTJD)")
    plt.ylabel("Normalized Flux")
    plt.savefig("TESS_Light_Curve.png")


def transit_orbit(lc):
    period = np.linspace(1, 20, 10000)
    bls = lc.to_periodogram('bls', period=period, frequency_factor=500)
    bls.plot(title='BLS Periodogram', xlabel='Period (days)', ylabel='Power')
    plt.savefig("first_BLS_Periodogram.png")
    planet_A_period = bls.period_at_max_power
    print("Analysis of planets orbital periods:")
    print(f"Planet A Period: {planet_A_period}ays")
    planet_A_t0 = bls.transit_time_at_max_power
    planet_A_duration = bls.duration_at_max_power
    planet_A_mask = bls.get_transit_mask(period=planet_A_period, transit_time=planet_A_t0, duration=planet_A_duration)

    masked_lc = lc[~planet_A_mask]
    bls = masked_lc.to_periodogram('bls', period=period, frequency_factor=500)
    bls.plot(title='BLS Periodogram', xlabel='Period (days)', ylabel='Power')
    plt.savefig("second_BLS_Periodogram.png")
    planet_B_period = bls.period_at_max_power
    print(f"Planet B Period: {planet_B_period}ays")
    planet_B_t0 = bls.transit_time_at_max_power
    planet_B_duration = bls.duration_at_max_power
    planet_B_mask = bls.get_transit_mask(period=planet_B_period, transit_time=planet_B_t0, duration=planet_B_duration)
    
    combined_masked_lc = masked_lc[~planet_B_mask]
    bls = combined_masked_lc.to_periodogram('bls', period=period, frequency_factor=500)
    bls.plot(title='BLS Periodogram', xlabel='Period (days)', ylabel='Power')
    plt.savefig("Third_BLS_Periodogram.png")
    planet_C_period = bls.period_at_max_power
    print(f"Planet C Period: {planet_C_period}ays")
    return planet_A_period, planet_B_period, planet_C_period


def analyze_data(data):
    num_columns = len(data.columns)
    num_rows = len(data)
    num_missing = data.isnull().sum().sum()
    mean_missing = data.isnull().mean().mean()
    num_missing_row = data.isnull().any(axis=1).sum() 
    num_missing_col = data.isnull().any(axis=0).sum()
    print(f"Number of columns: {num_columns}")
    print(f"Number of rows: {num_rows}")
    print(f"Number of missing values: {num_missing}")
    print(f"Mean missing values percentage: {mean_missing}")
    print(f"Missing values row: {num_missing_row}")
    print(f"Missing values column: {num_missing_col}")


def cleaning_tess_summary_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(columns={data.columns[0]: data.columns[0][1:]})
    data = data.dropna()
    data = data.loc[:, ["tic_id", "Period"]]
    return data


def cleaning_NASA_planet_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.loc[data["sy_snum"] == 1, ["pl_name", "pl_orbper", "pl_orbsmax", "pl_trandur", "st_rad", "st_teff"]]
    data = data.dropna()
    return data


def cleaned_data_summary(data: pd.Series) -> None:
    print(f"{data.name} Description:\n {data.describe()}")


def set_stellar_type(data):
    if data["st_teff"] < 3900:
        return "M"
    elif data["st_teff"] < 5300:
        return "K"
    elif data["st_teff"] < 6000:
        return "G"
    elif data["st_teff"] < 7300:
        return "F"
    elif data["st_teff"] < 10000:
        return "A"
    elif data["st_teff"] < 33000:
        return "B"
    else:
        return "O"


def shortest_range_of_50_percent(data: pd.Series) -> pd.Series:
    data = np.sort(data.to_numpy())
    half = int(np.ceil(len(data) * 0.5))
    range = data[half - 1:] - data[:len(data) - half + 1]
    index_lower_bound = np.argmin(range)
    return pd.Series({"range": data[index_lower_bound + half - 1] - data[index_lower_bound], "lower_bound": data[index_lower_bound], "upper_bound": data[index_lower_bound + half - 1]})


def ratio_analysis(data: pd.DataFrame) -> pd.DataFrame:
    data["ratio"] = data["pl_trandur"] / data["pl_orbper"]
    ratio_summary = data.groupby("stellar_spectype")["ratio"].apply(shortest_range_of_50_percent)
    print("Ratio Analysis:")
    print(ratio_summary)


def plot_transit_orbit_comparison(measured_transit_orbit, documented_transit_orbit):
    plt.figure()
    x = np.arange(3)
    width = 0.35
    planet_names = ['Planet A', 'Planet B', 'Planet C']
    fig, ax = plt.subplots()
    sns.barplot(ax=ax, x=x - 0.2, y=measured_transit_orbit, color='blue',
                width=width, native_scale=True, label='Measured')
    sns.barplot(ax=ax, x=x + 0.2, y=documented_transit_orbit, color='red',
                width=width, native_scale=True, label='Documented')
    ax.set_title("Measured vs Documented Transit Orbits")
    ax.set_xticks(x)
    ax.set_xticklabels(planet_names)
    ax.set_ylabel("Period (days)")
    ax.legend(title="Orbit Type")
    plt.savefig("Measured_vs_Documented_Transit_Orbits.png")
    period_diff = []
    for measured, documented in zip(measured_transit_orbit, documented_transit_orbit):
        period_diff.append(abs(measured - documented))
    plt.figure()
    sns.relplot(x=planet_names, y=period_diff,
                hue=[f"Planet A: {documented_transit_orbit[0]:.3f} days",
                     f"Planet B: {documented_transit_orbit[1]:.3f} days",
                     f"Planet C: {documented_transit_orbit[2]:.3f} days"],)
    plt.xlabel("Planets")
    plt.ylabel("Period Difference (days)")
    plt.title("Transit Orbit Period Difference")
    plt.savefig("Transit_Orbit_Period_Difference.png")


def plot_stellar_type_distribution(data):
    plt.figure()
    sns.countplot(x='stellar_spectype', data=data, order=['O', 'B', 'A', 'F', 'G', 'K', 'M'])
    plt.title("Stellar Type Distribution")
    plt.xlabel("Stellar Type")
    plt.ylabel("Count")
    plt.savefig("Stellar_Type_Distribution.png")


"""
def plot_stellar_type_vs_orbital_period(data):
    plt.figure()
    sns.boxplot(x='stellar_spectype', y='pl_orbper', data=data, order=['O', 'B', 'A', 'F', 'G', 'K', 'M'])
    plt.title("Stellar Type vs Orbital Period")
    plt.xlabel("Stellar Type")
    plt.ylabel("Orbital Period (days)")
    plt.savefig("Stellar_Type_vs_Orbital_Period.png")
"""


def plot_transit_duration_vs_orbital_period(data):
    types = ["O", "B", "A", "F", "G", "K", "M"]
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    for ax, star_type in zip(axes.flat, types):
        subset = data.loc[data["stellar_spectype"] == star_type]
        sns.scatterplot(ax=ax, data=subset, x="pl_orbper", y="pl_trandur")
        ax.set_title(f"Type {star_type}")
        ax.set_xlabel("Orbital Period (days)")
        ax.set_ylabel("Transit Duration (hours)")
    plt.savefig("transit Duration vs orbital Period.png")


def plot_ratio_distribution(data):
    data["ratio"] = data["pl_trandur"] / data["pl_orbper"]
    plt.figure()
    sns.displot(data=data, x="ratio", col="stellar_spectype", bins=50, log_scale=True, facet_kws={"sharex": False, "sharey": False})
    plt.savefig("ratio_distribution.png")


def main():
    tess_lightcurve_filename = "hlsp_tess-data-alerts_tess_phot_00307210830-s02_tess_v1_lc.fits"
    tess_TOI_summary_filename = "hlsp_tess-data-alerts_tess_phot_alert-summary-s01+s02+s03+s04_tess_v9_spoc.csv"
    NASA_planet_filename = "PS_2026.11.05_19.20.58.csv"
    tess_summary_data = pd.read_csv(tess_TOI_summary_filename)
    NASA_planet_data = pd.read_csv(NASA_planet_filename, comment='#')
    lc_data = setup_lc_data(tess_lightcurve_filename)
    print("Light Curve Data analysis:")
    analyze_data(lc_data)
    print("TESS Summary Data analysis:")
    analyze_data(tess_summary_data)
    print("NASA Planet Data analysis:")
    analyze_data(NASA_planet_data)
    lc = lk.read(tess_lightcurve_filename)
    lc.remove_nans()
    plot_light_curve(lc)
    lc.flatten(window_length=901).remove_outliers()
    lc.normalize()
    NASA_planet_data = cleaning_NASA_planet_data(NASA_planet_data)
    lc_data.apply(cleaned_data_summary)
    NASA_planet_data.apply(cleaned_data_summary)
    tess_summary_data = cleaning_tess_summary_data(tess_summary_data)
    print(f"TESS Summary Data Description:\n"
          f"Number of stars: {len(tess_summary_data['tic_id'].unique())}\n" +
          f"Planet period description:\n {tess_summary_data['Period'].describe()}")
    measured_transit_orbit = sorted(transit_orbit(lc))
    documented_transit_orbit = sorted(tess_summary_data[tess_summary_data['tic_id'] == 307210830]['Period'].to_list())
    measured_transit_orbit = [period.to_value(u.day) for period in measured_transit_orbit]
    plot_transit_orbit_comparison(measured_transit_orbit, documented_transit_orbit)
    NASA_planet_data['stellar_spectype'] = NASA_planet_data.apply(set_stellar_type, axis=1)
    plot_stellar_type_distribution(NASA_planet_data)
    #plot_stellar_type_vs_orbital_period(NASA_planet_data)
    plot_transit_duration_vs_orbital_period(NASA_planet_data)
    plot_ratio_distribution(NASA_planet_data)
    ratio_analysis(NASA_planet_data)


if __name__ == "__main__":
    main()