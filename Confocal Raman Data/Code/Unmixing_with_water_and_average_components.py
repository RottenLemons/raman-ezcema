# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Raman_DB as rm
import argparse
import matplotlib as mpl
from sklearn import preprocessing
import matplotlib.gridspec as gridspec


# %%
parser = argparse.ArgumentParser()
parser.add_argument("input_file", help="Input CSV file")
args = parser.parse_args()

with open(args.input_file, "r") as file:
    rawdata = pd.read_csv(args.input_file, header=None).to_numpy()

# folder for results to be saved in
folder_name1 = os.path.splitext(args.input_file)[0]
subject_name = [args.input_file.split("\\")[1]]

if not os.path.exists(folder_name1):
    os.makedirs(folder_name1)


# %%
# sort data file
sorteddata_671, sorteddata_785 = rm.AT_sort(rawdata)

# change firstrow of wavelength to wavenumber
wavelength = rawdata[0, :]
wavenumber_671 = rm.wl2wn(wavelength, central_wl=671)
wavenumber_785 = rm.wl2wn(wavelength, central_wl=784.5)


# %%
# preprocessing2 func from Raman_DB does both baseline correction and smoothing
baselinecorr_785, bkg_785 = rm.preprocessing2(sorteddata_785, 1e4, 0.0001)
baselinecorr_671, bkg_671 = rm.preprocessing2(sorteddata_671, 1e6, 0.001)

# inserting wavenumber to processed data file
processed_671 = np.insert(baselinecorr_671, 0, wavenumber_671, axis=0)
processed_785 = np.insert(baselinecorr_785, 0, wavenumber_785, axis=0)


# %%
# selecting wavenumber range
step = 5  # step size 5 micrometer
LW_L = 550  # FingerPrint region lower limit
LW_H = 1750  # FP region upper limit
HW_L = 2800  # HighWavenumber region lower limit
HW_H = 3800  # HW region upper limit

selected_671_data = rm.select_wn_range(processed_671, HW_L, HW_H)
selected_785_data = rm.select_wn_range(processed_785, LW_L, LW_H)
selected_wavenumber_671 = selected_671_data[0, :]
selected_wavenumber_785 = selected_785_data[0, :]


# %%
# Import pure spectra of individual skin components data
# os.chdir(r"./Pure Components Spectra")

pure_system = pd.read_csv("./Pure Components Spectra/AT_system_v3.csv", header=None).to_numpy()  # 0  AT_system_v3
components = pure_system

etalon = pd.read_csv("./Pure Components Spectra/AT_etalon_v3.csv", header=None).to_numpy()  # 1
components = np.vstack([components, etalon[1, :]])

pure_cera2 = pd.read_csv("./Pure Components Spectra/AT_Ceramide2_1822_v3.csv", header=None).to_numpy()  # 2
components = np.vstack([components, pure_cera2[1, :]])

pure_cera3 = pd.read_csv("./Pure Components Spectra/AT_Ceramide3_v3.csv", header=None).to_numpy()  # 3
components = np.vstack([components, pure_cera3[1, :]])

pure_kera = pd.read_csv("./Pure Components Spectra/AT_Keratin_v3.csv", header=None).to_numpy()  # 4
components = np.vstack([components, pure_kera[1, :]])

pure_chol = pd.read_csv("./Pure Components Spectra/AT_Cholesterol_v3.csv", header=None).to_numpy()  # 5
components = np.vstack([components, pure_chol[1, :]])

pure_lac = pd.read_csv("./Pure Components Spectra/AT_Lactic_acid_liquid_v3.csv", header=None).to_numpy()  # 6
components = np.vstack([components, pure_lac[1, :]])

pure_pca = pd.read_csv("./Pure Components Spectra/AT_PCA_v3.csv", header=None).to_numpy()  # 7
components = np.vstack([components, pure_pca[1, :]])

pure_uca = pd.read_csv("./Pure Components Spectra/AT_Urocanic_acid_v3.csv", header=None).to_numpy()  # 8
components = np.vstack([components, pure_uca[1, :]])

pure_urea = pd.read_csv("./Pure Components Spectra/AT_Urea_v3.csv", header=None).to_numpy()  # 9
components = np.vstack([components, pure_urea[1, :]])

pure_water = pd.read_csv("./Pure Components Spectra/AT_water_v3.csv", header=None).to_numpy()  # 10
components = np.vstack([components, pure_water[1, :]])

pure_mela = pd.read_csv("./Pure Components Spectra/AT_Melanin_v3.csv", header=None).to_numpy()  # 11
components = np.vstack([components, pure_mela[1, :]])


# %%
# pure components for unmixing
wn_785 = rm.wl2wn(components[0, :], central_wl=786)
components[0] = wn_785  # replacing the first row to wavenumber
components_select = rm.select_wn_range(components, LW_L - 11, LW_H - 9)

no_components = len(components) - 2  # minus the pure_system and wavnumber in the first 2 rows


# %%
# Set the directory to save the plots in
# directory = os.path.split(args.input_file)[0]
folder_name2 = "plots"
# os.chdir(r"../EC001/5.40.27 pm_17_5_001_eczema")
plot_dir = os.path.join(folder_name1, folder_name2)

# Create the directory if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


# %%
# Override the default matplotlib save method
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(plot_dir, fig_id + ".jpg")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, dpi=600)


# %%
# Spectral plot of the raw spectra (before preprocessing)
mpl.rcParams["font.size"] = 14
start_plot = 0
stop_plot = 17
inc = 2  # step increment for plot
string = []
fig = plt.figure(constrained_layout=False)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[1.3, 1], wspace=0.12)
ax1 = fig.add_subplot(spec[0, 0])
for i in range(start_plot, stop_plot, inc):
    ax1.plot(wavenumber_785, sorteddata_785[i + 1, :] - 300 * i, linewidth=1.5)
    string.append(str(i))

ax2 = fig.add_subplot(spec[0, 1])
for i in range(start_plot, stop_plot, inc):
    ax2.plot(wavenumber_671, sorteddata_671[i + 1, :] - 300 * i, linewidth=1.5)

ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax1.set_xlim(LW_L, LW_H)
ax2.set_xlim(HW_L, HW_H)
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
fig.text(0.5, 0.03, "wavenumber (cm-1)", ha="center", va="center")
ax1.set_ylabel("Intensity (a.u.)")
plt.suptitle("Raw Spectrum", y=0.95)
save_fig(fig_id=f"Raw Spectrum")


# %%
# Spectral plot of the raw data (individual spectrum)
fig = plt.figure(constrained_layout=False)
ax1 = fig.add_subplot(spec[0, 0])
ax1.plot(wavenumber_785, sorteddata_785[12, :], linewidth=1.5)
ax2 = fig.add_subplot(spec[0, 1])
ax2.plot(wavenumber_671, sorteddata_671[12, :], linewidth=1.5)
ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax1.set_xlim(LW_L, LW_H)
ax2.set_xlim(HW_L, HW_H)
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
fig.text(0.5, 0.03, "wavenumber (cm-1)", ha="center", va="center")
ax1.set_ylabel("Intensity (a.u.)")
plt.suptitle("Raw Spectrum (Individual)", y=0.95)
save_fig(fig_id=f"Raw Spectrum-Individual")


# %%
# Backgound Plot (individual spectrum)
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["font.size"] = 16
plt.figure()
plt.plot(wavenumber_785, sorteddata_785[14])
plt.plot(wavenumber_785, bkg_785[14])
plt.plot(wavenumber_671, sorteddata_671[14])
plt.plot(wavenumber_671, bkg_671[14])
plt.title("Spectrum Fitting")
plt.ylabel("Intensity (a.u.)")
plt.xlabel("Wavenumber (cm-1)")
plt.yticks([])
save_fig(fig_id=f"Spectrum Fitting")


# %%
# Spectra plot of the merged spectra (after preprocessing)
mpl.rcParams["font.size"] = 14
start_plot = 0
stop_plot = 17
inc = 2  # step increment for plot
string = []
fig = plt.figure(constrained_layout=False)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[1.3, 1], wspace=0.12)
ax1 = fig.add_subplot(spec[0, 0])
for i in range(start_plot, stop_plot, inc):
    ax1.plot(selected_wavenumber_785, selected_785_data[i + 1, :] - 300 * i, linewidth=1.5)
    string.append(str(i))

ax2 = fig.add_subplot(spec[0, 1])
for i in range(start_plot, stop_plot, inc):
    ax2.plot(selected_wavenumber_671, selected_671_data[i + 1, :] - 300 * i, linewidth=1.5)

ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax1.set_xlim(LW_L, LW_H)
ax2.set_xlim(HW_L, HW_H)
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax1.axvline(x=1625, c="r")
ax1.axvline(x=1720, c="r")
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
fig.text(0.5, 0.03, "wavenumber (cm-1)", ha="center", va="center")
ax1.set_ylabel("Intensity (a.u.)")
save_fig(fig_id=f"Merged Spectra")


# %%
# Spectra plot of the merged spectra (after preprocessing) #FN
mpl.rcParams["font.size"] = 14
start_plot = 0
stop_plot = 17
inc = 2  # step increment for plot
string = []
fig = plt.figure(constrained_layout=False)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[1.3, 1], wspace=0.12)
ax1 = fig.add_subplot(spec[0, 0])
for i in range(start_plot, stop_plot, inc):
    ax1.plot(selected_wavenumber_785, selected_785_data[i + 1, :] - 300 * i, linewidth=1.5)
    string.append(str(i))

ax2 = fig.add_subplot(spec[0, 1])
for i in range(start_plot, stop_plot, inc):
    ax2.plot(selected_wavenumber_671, selected_671_data[i + 1, :] - 300 * i, linewidth=1.5)

ax1.set_yticklabels([])
ax2.set_yticklabels([])
ax1.set_xlim(LW_L, LW_H)
ax2.set_xlim(HW_L, HW_H)
ax1.spines["right"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.axvline(x=3020, c="r")
ax2.axvline(x=3120, c="r")
ax1.yaxis.tick_left()
ax2.yaxis.tick_right()
fig.text(0.5, 0.03, "wavenumber (cm-1)", ha="center", va="center")
ax1.set_ylabel("Intensity (a.u.)")
save_fig(fig_id=f"Merged Spectra_2")


# %%
## Unmixing of the fingerprint region (785nm)

# normalising data
normalised_785_data = preprocessing.normalize(processed_785, axis=1, copy=True)
normalised_785_data[0] = wn_785
select_nm_785 = rm.select_wn_range(normalised_785_data, LW_L, LW_H)
final_785 = pd.DataFrame(normalised_785_data)


# %%
# For each spctra in the fingerprint region (each depth), use the intensity from the range 1400 to 1520nm
# calculate the difference between the max intensity and the average intensity.
extract_785 = pd.DataFrame(rm.select_wn_range(normalised_785_data, 1625, 1720))
avg_785 = extract_785.iloc[1:, :].mean(axis=1)
max_int_785 = extract_785.iloc[1:, :].max(axis=1)
max_idx_785 = (max_int_785 - avg_785).idxmax()


# %%
results = np.zeros(
    [len(range(max([1, max_idx_785 - 2]), 18)), no_components - 3]
)  # minus the those components you don't want to calculate the relative conc.


# %%
for i in range(max([1, max_idx_785 - 2]), 18):
    data_fp = final_785.iloc[[i], :]
    data_fp = pd.concat([pd.DataFrame(wn_785).T, data_fp], axis=0)

    data_fp = data_fp.values
    data_fp = rm.select_wn_range(data_fp, LW_L, LW_H)  # select the range

    xx_results, xx_simu = rm.unmix_spec_2D(
        components_select, data_fp
    )  # results is the coeff while simu is the fitted spectra

    # Relative Concentration to Keratin
    AT_kera = xx_results[:, 4]
    AT_cera2_0 = xx_results[:, 2] / AT_kera
    AT_cera3_0 = xx_results[:, 3] / AT_kera
    AT_lac_0 = xx_results[:, 6] / AT_kera
    AT_pca_0 = xx_results[:, 7] / AT_kera
    AT_uca_0 = xx_results[:, 8] / AT_kera
    AT_urea_0 = xx_results[:, 9] / AT_kera
    AT_mela_0 = xx_results[:, 11] / AT_kera
    AT_chol_0 = xx_results[:, 5] / AT_kera
    AT_kera[AT_kera == 0] = 1e-9

    print("Spectra {}".format(i))
    print("Ceramide2:")
    print(AT_cera2_0[0])
    print("Ceramide3:")
    print(AT_cera3_0[0])
    print("Urocanic Acid:")
    print(AT_uca_0[0])
    print("PCA:")
    print(AT_pca_0[0])
    print("Urea:")
    print(AT_urea_0[0])
    print("Lactic Acid:")
    print(AT_lac_0[0])
    print("\n")

    # Plotting of data and fitted spectra

    mpl.rcParams["font.size"] = 16
    fig = plt.figure()
    plt.plot(components_select[0, :], np.transpose(select_nm_785[i, :]))
    plt.plot(components_select[0, :], xx_simu[1, :], ":")
    plt.legend(["Original spectrum", "Fitted spectrum"], bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.yticks([])
    plt.ylabel("Intensity (a.u.)")
    plt.xlabel("Wavenumber (cm-1)")
    plt.title("Raw and fitted spectra")
    depth = str(i)
    fig.suptitle(f"Depth: {depth}")
    plt.subplots_adjust(top=0.85)
    save_fig(fig_id=f"Raw_and_fitted_spectra{depth}")

    results[i - max_idx_785, :] = np.transpose(
        np.vstack([AT_cera2_0,AT_cera3_0, AT_lac_0, AT_pca_0, AT_uca_0, AT_urea_0, AT_mela_0, AT_chol_0])
    )


# %%
concat_results = pd.DataFrame(
    results, columns=["Ceramide 2", "Ceramide 3", "Lactic Acid", "PCA", "Uric Acid", "Urea", "Melanin", "Cholesterol"]
)
save_results = concat_results.iloc[1:, :]


# %%
results_directory = folder_name1
results_path = os.path.join(results_directory, "results.csv")
save_results.to_csv(results_path, index=True)


# %%
## Averaging of skin components for all the selected depths

avg_components = concat_results.iloc[1:, :].mean(axis=0)
avg_components.replace([np.inf, -np.inf], np.nan, inplace=True)
avg_components_785 = pd.DataFrame(avg_components)
avg_components_785 = avg_components_785.T
avg_components_785 = avg_components_785.iloc[:, 0:8]
avg_components_785.index = subject_name
results_path = os.path.join(results_directory, "average components.csv")
avg_components_785.to_csv(results_path, index=True)


# %%
## Calculating water content using the 671nm region

# spectra with max intensity within region is taken as skin surface
# For each spctra in the highwavenumber region (each depth), use the intensity from the range 3350 to 3550nm
extract_671 = pd.DataFrame(rm.select_wn_range(processed_671, 3020, 3120))

# Use peak intensity to determine skin surface
max_int_671 = extract_671.iloc[1:, :].max(axis=1)
max_idx_671 = max_int_671.idxmax()


# %%
water_671 = pd.DataFrame()

for i in range(max([1, max_idx_671 - 2]), 18):
    # HW REGION ONLY (for water profile)
    prep = pd.DataFrame(processed_671)
    data_hw = prep.iloc[[i], :]
    data_hw = pd.concat([pd.DataFrame(wavenumber_671).T, data_hw], axis=0)

    # Water Ratio (Sum)
    sum_waterAT = np.zeros(len(data_hw) - 1)
    waterAT = rm.select_wn_range(data_hw.values, 3350, 3550)

    for j in range(len(data_hw) - 1):
        sum_waterAT[j] = np.sum(waterAT[j + 1, :])

    watermass_0 = sum_waterAT
    water_671 = pd.concat([water_671, pd.DataFrame(watermass_0)], axis=1)

water_671.columns = range(max([1, max_idx_671 - 2]), 18)
water_671.index = subject_name


# %%
results_path = os.path.join(results_directory, "Watermass.csv")
water_671.to_csv(results_path, index=True)
