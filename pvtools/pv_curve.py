import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from pathlib import Path

import matplotlib.pyplot as plt

import logging


# object class called a pv curve
class PVCurve:
    def __init__(
            self,
            psis: np.ndarray,
            masses: np.ndarray,
            dry_mass: float,
            psi_unit: str = 'MPa',
            mass_unit: str = 'g',
            dry_mass_unit: str = 'g',
            leaf_area: float = np.nan,
            height: float = 0,
            bkp: int = 0,
            tlp_conf: float = 0.05,
            psi_sign_convention='negative',
            validate_data: bool = True):
        """
        PVCurve object that stores data and basic calculations for a pressure-volume curve.
        Required inputs:
            psis: numpy array of leaf water potentials (MPa)
            masses: numpy array of masses (g)
            dry_mass: mass of dry leaf (g)
        Optional inputs:
            psi_unit: unit of the psi measurements (default is MPa)
            mass_unit: unit of the wet and dry mass measurements (default is g)
            leaf_area: leaf area (cm^2)
            height: height of the sample (m)
            bkp: breakpoint index (default is 0, which will be calculated -- alternatively, provide the # of points that should be included in the first segment)
            tlp_conf: confidence interval for the turgor loss point (default is 0.05 - aka, a 95% confidence interval)
            psi_sign_convention: 'negative' (default) or 'positive' -- indicates whether the input psis are negative values (as is standard) or positive values (e.g., pressure bomb outputs)
            validate_data: whether to validate the input data (default is True). recommended to leave as True unless you are sure the data is clean.
        """
        # TODO: #2 remodel init so that each calculation takes place in a function

        # input data...
        self.psis = psis
        self.masses = masses
        self.dry_mass = dry_mass
        self.psi_unit = psi_unit
        self.mass_unit = mass_unit
        self.dry_mass_unit = dry_mass_unit
        self.leaf_area = leaf_area
        self.height = height
        self.bkp = bkp
        self.tlp_conf = tlp_conf
        self.psi_sign_convention = psi_sign_convention
        self.validate_data = validate_data

        # if psis, masses, or dry mass are not provided/None, raise an error
        if self.psis is None or self.masses is None or self.dry_mass is None:
            raise ValueError("psis, masses, and dry_mass must all be provided.")
        
        # drop data points that don't have both psi and mass values
        valid_idx = ~np.isnan(self.psis) & ~np.isnan(self.masses)
        if len(self.psis[valid_idx]) < len(self.psis):
            logging.warning("Dropping data points with missing Ψ or mass values...")
            self.psis = self.psis[valid_idx]
            self.masses = self.masses[valid_idx]
        if len(self.psis) < 4:
            raise ValueError("Need at least four non-NaN data points to create a PVCurve. Sorry!")

        # if psi units are not MPa, convert to MPa
        if self.psi_unit == None:
            self.psi_unit = 'MPa'
            logging.info("No psi unit provided. Assuming MPa...")
        if self.psi_unit.lower() == 'bar':
            self.psis = self.psis / 10.0
            logging.info("Converted Ψ values from bar to MPa for internal processing. New values: " + str(self.psis[0:5]) + "...")
        elif self.psi_unit.lower() == 'kpa':
            self.psis = self.psis / 1000.0
            logging.info("Converted Ψ values from kPa to MPa for internal processing. New values: " + str(self.psis[0:5]) + "...")
        elif self.psi_unit.lower() == 'mpa':
            pass
        else:
            raise ValueError("Unsupported psi unit. Please use 'MPa', 'bar', or 'kPa'.")
        
        # if mass units are not g, convert to g
        if self.mass_unit == None:
            self.mass_unit = 'g'
            logging.info("No mass unit provided. Assuming grams...")
        if self.mass_unit.lower() == 'mg':
            self.masses = self.masses / 1000.0
            self.dry_mass = self.dry_mass / 1000.0
            logging.info("Converted mass values from mg to g for internal processing. New values: " + str(self.masses[0:5]) + "...")
        elif self.mass_unit.lower() == 'g':
            pass
        else:
            raise ValueError("Unsupported mass unit. Please use 'g' or 'mg'.")
        
        # same for dry mass...
        if self.dry_mass_unit == None:
            self.dry_mass_unit = self.mass_unit
            logging.info("No dry mass unit provided. Assuming " + self.mass_unit + " (same as mass unit)...")
        if self.dry_mass_unit.lower() == 'mg':
            self.dry_mass = self.dry_mass / 1000.0
            logging.info("Converted dry mass value from mg to g for internal processing. New value: " + str(self.dry_mass) + "...")
        elif self.dry_mass_unit.lower() == 'g':
            pass
        else:
            raise ValueError("Unsupported dry mass unit. Please use 'g' or 'mg'.")

        # handle psi sign convention
        if self.psi_sign_convention.lower() == 'positive':
            self.psis = -1 * self.psis
        elif self.psi_sign_convention.lower() == 'negative':
            # if all of the psis are positive, though, let's just fix it
            if np.all(self.psis >= 0):
                logging.warning("All Ψ values are positive. Flipping sign to negative for all points...")
                self.psis = -1 * self.psis
        else:
            raise ValueError("Unsupported psi_sign_convention. Please use 'negative' or 'positive'.")

        # validate the data
        if self.validate_data:
            self._validate_data()

        # sort everything according to the mass in reverse order
        sort_idx = np.argsort(self.masses)[::-1]
        self.psis = self.psis[sort_idx]
        self.masses = self.masses[sort_idx]
        
        # if the sorting changed something, let the user know
        if not np.array_equal(sort_idx, np.arange(len(self.masses))):
            logging.warning("Data was sorted into decreasing mass order.")

        # basic calculations
        # if there is a psi that's exactly zero, nudge it a bit
        if np.any(self.psis == 0):
            logging.warning("One or more Ψ values are exactly zero. Nudging these values slightly negative to avoid division by zero...")
            self.psis[self.psis == 0] = -1e-6
        self.inverse_psis = -1 / self.psis
        self.water_mass = self.masses - self.dry_mass

        # get breakpoint if not provided
        if bkp == 0:
            logging.info("No breakpoint provided. Calculating breakpoint...")
            self.bkp = get_breakpoint(self.psis, self.masses, self.dry_mass, psi_unit = self.psi_unit, mass_unit = self.mass_unit, dry_mass_unit = self.dry_mass_unit,
                                      leaf_area = self.leaf_area, height = self.height, psi_sign_convention = self.psi_sign_convention, validate_data = self.validate_data)

        # water mass at full turgor (Ψ = 0)
        slope, intercept = np.polyfit(self.water_mass[:self.bkp], self.psis[:self.bkp], 1)
        self.water_FT_slope = slope   # used in osmotic potential calculation
        self.water_FT = -intercept / self.water_FT_slope

        # saturated water content (SWC)
        self.swc = self.water_FT / self.dry_mass

        # relative water content (RWC)
        self.rwc = self.water_mass / self.water_FT

        # osmotic potential
        slope, intercept = np.polyfit(1-self.rwc[self.bkp:], self.inverse_psis[self.bkp:], 1)
        self.os_pot_FT_slope = slope
        self.os_pot_FT_inv = intercept
        self.os_pot_FT = -1/self.os_pot_FT_inv

        # solute potential + turgor pressure split
        self.sol_pot = -1 / (self.os_pot_FT_inv+self.os_pot_FT_slope*(1-self.rwc))
        self.turgor_pressure = self.psis - self.sol_pot

        # turgor loss point slope
        slope, intercept = np.polyfit(self.sol_pot[:self.bkp], self.turgor_pressure[:self.bkp], 1)
        self.tlp_slope = slope

        # turgor loss point confidence interval
        x = self.turgor_pressure[:self.bkp]
        x = sm.add_constant(x)
        y = self.sol_pot[:self.bkp]
        model = sm.OLS(y, x).fit()
        y_hat = model.get_prediction([1,0]).summary_frame(alpha=self.tlp_conf)
        self.tlp_conf_int = (y_hat['mean_ci_lower'].values[0], y_hat['mean_ci_upper'].values[0])
        self.tlp = y_hat['mean'].values[0]

        # rwc at turgor loss point
        slope, intercept = np.polyfit(self.rwc[:self.bkp], self.turgor_pressure[:self.bkp], 1)
        self.bulk_elastic_total = slope
        self.rwc_tlp = -intercept / slope

        # apoplastic water fraction
        self.awf = 1-(-self.os_pot_FT_inv/self.os_pot_FT_slope)
        self.rwc_tlp_sym = (self.rwc_tlp-self.awf)/(1-self.awf)
        self.bulk_elastic_total_sym = self.bulk_elastic_total*(1-self.awf)

        # hydraulic capacitance before TLP
        slope, intercept = np.polyfit(self.psis[:self.bkp], self.rwc[:self.bkp], 1)
        self.ct_before = slope
        self.ct_before_massnorm = self.ct_before * self.swc
        self.ct_before_areanorm = self.ct_before * self.water_FT / self.leaf_area

        # water storage capacity
        self.capacity_massnorm = (1-self.rwc_tlp) * self.swc
        self.capacity_areanorm = (1-self.rwc_tlp) * self.water_FT / self.leaf_area
        # water storage capacity normalized by gravity...
        self.capacity_massnorm_gravity = ((-0.0098 * self.height+self.water_FT_slope*self.water_FT)/self.water_FT_slope-self.rwc_tlp*self.water_FT)/self.dry_mass
        self.capacity_areanorm_gravity = ((-0.0098 * self.height+self.water_FT_slope*self.water_FT)/self.water_FT_slope-self.rwc_tlp*self.water_FT)/self.leaf_area

        # hydraulic capacitance after TLP
        slope, intercept = np.polyfit(self.psis[self.bkp:], self.rwc[self.bkp:], 1)
        self.ct_after = slope
        self.ct_after_massnorm = self.ct_after * self.swc
        self.ct_after_areanorm = self.ct_after * self.water_FT / self.leaf_area

        # correlation coefficient
        self.r2_beforetlp = np.corrcoef(self.psis[:self.bkp], self.water_mass[:self.bkp])[0,1]**2
        self.r2_aftertlp = np.corrcoef(self.inverse_psis[self.bkp:], self.rwc[self.bkp:])[0,1]**2
        self.r2 = (self.r2_beforetlp * self.bkp + self.r2_aftertlp * (len(self.psis)-self.bkp)) / len(self.psis)
    
    def _validate_data(self):
        if self.psis.shape != self.masses.shape:
            raise ValueError("Ψ and mass data are not the same shape. If there are missing measurements, please fill in with np.nan.")
        if self.bkp != 0:
            if self.bkp < 0 or self.bkp > len(self.psis):
                raise ValueError("Breakpoint is either negative or larger than the number of points in the dataset.")
            if self.bkp < 2 or self.bkp > len(self.psis) - 2:
                raise ValueError("Breakpoint must leave at least 2 points on each side.")
        if np.any(self.masses < self.dry_mass):
            raise ValueError("One or more mass measurements are less than the dry mass. Please check your data.")


    # remove point from curve based on index
    def remove_point(self, idx: int):
        """
        Remove points from the PVCurve object based on index.
        """
        new_psis = np.delete(self.psis, idx)
        new_masses = np.delete(self.masses, idx)
        # when creating a new curve, bkp is set back to 0 so it will recalculate
        new_pvcurve = PVCurve(new_psis, new_masses, self.dry_mass, psi_unit = self.psi_unit, mass_unit = self.mass_unit, dry_mass_unit = self.dry_mass_unit,
        leaf_area = self.leaf_area, height = self.height, bkp = 0, tlp_conf = self.tlp_conf, psi_sign_convention = self.psi_sign_convention, validate_data = self.validate_data)
        return new_pvcurve

    # print function
    def __repr__(self):
        return "TLP: {:.3f} MPa \nR2: {:.3f}".format(self.tlp, self.r2)   

    # length function (number of points in the curve)
    def __len__(self):
        return len(self.psis)

    # indexing functions
    def __getitem__(self, idx):
        return (self.psis[idx], self.masses[idx])
    def __setitem__(self, idx, value):
        self.psis[idx], self.masses[idx] = value
    
    # save out to a csv file
    def save_csv(self, filename: str):
        """
        Save the PVCurve object to a CSV file.
        """

        # check that the filename ends in .csv; if not throw exception
        if not filename.endswith('.csv'):
            raise ValueError("Filename must end in .csv")

        # create a dictionary of the data
        data = {
            'Ψ (MPa)': self.psis,
            'Mass (g)': self.masses,
            'Dry Mass (g)': [self.dry_mass] + [np.nan]*(len(self.psis)-1),
            '-1/Ψ (MPa^-1)': self.inverse_psis,
            'Water mass (g)': self.water_mass,
            'RWC': self.rwc,
            'Solute potential (MPa)': self.sol_pot,
            'Turgor pressure (MPa)': self.turgor_pressure
        }

        calculations = {
            'Number of points used for \'before TLP\'': self.bkp,
            'Water at FT (g)': self.water_FT,
            'Saturated Water Content': self.swc,
            'Osmotic potential at FT (MPa)': self.os_pot_FT,
            'TLP (MPa)': self.tlp,
            'Bulk elastic modulus (MPa)': self.bulk_elastic_total,
            'RWC at TLP': self.rwc_tlp,
            'Bulk elastic modulus (symplastic) (MPa)': self.bulk_elastic_total_sym,
            'RWC at TLP (symplastic)': self.rwc_tlp_sym,
            'Apoplastic Water Fraction': self.awf,
            'Hydraulic capacitance before TLP (g MPa^-1)': self.ct_before_massnorm,
            'Hydraulic capacitance before TLP (cm^2 MPa^-1)': self.ct_before_areanorm,
            'Water storage capacity (g)': self.capacity_massnorm,
            'Water storage capacity (cm^2)': self.capacity_areanorm,
            'Water storage capacity (g) (gravity corrected)': self.capacity_massnorm_gravity,
            'Water storage capacity (cm^2) (gravity corrected)': self.capacity_areanorm_gravity,
            'Hydraulic capacitance after TLP (g MPa^-1)': self.ct_after_massnorm,
            'Hydraulic capacitance after TLP (cm^2 MPa^-1)': self.ct_after_areanorm,
            'R2 before TLP': self.r2_beforetlp,
            'R2 after TLP': self.r2_aftertlp,
            'R2': self.r2
        }

        # create a dataframe with the data and calculations
        df_data = pd.DataFrame(data)
        df_calculations = pd.DataFrame([calculations.keys(), calculations.values()]).T
        df_calculations.columns = ['Variable Name', 'Value']
        df = pd.concat([df_data, df_calculations], axis=1)
        df.to_csv(filename, index=False)

        absolute_path = os.path.abspath(filename)
        logging.info(f"Data saved to {absolute_path}")

    def to_dataframe(self):
        """
        Store all the array-like data in a pandas dataframe.
        """
        data = {
            'Ψ (MPa)': self.psis,
            'Mass (g)': self.masses,
            '-1/Ψ (MPa^-1)': self.inverse_psis,
            'Water mass (g)': self.water_mass,
            'RWC': self.rwc,
            'Solute potential (MPa)': self.sol_pot,
            'Turgor pressure (MPa)': self.turgor_pressure
        }

        df = pd.DataFrame(data)
        return df
            

    
    
    # save out to an excel sheet
    def save_excel(self, filename: str):
        """
        Save the PVCurve object to an Excel file.
        """

        # check that the filename ends in .xlsx; if not throw exception
        if not filename.endswith('.xlsx') and not filename.endswith('.xls') and not filename.endswith('.xlsm'):
            raise ValueError("Filename must end in .xlsx, .xls, or .xlsm")

        # create a dictionary of the data
        data = {
            'Ψ (MPa)': self.psis,
            'Mass (g)': self.masses,
            'Dry Mass (g)': [self.dry_mass] + [np.nan]*(len(self.psis)-1),
            '-1/Ψ (MPa^-1)': self.inverse_psis,
            'Water mass (g)': self.water_mass,
            'RWC': self.rwc,
            'Solute potential (MPa)': self.sol_pot,
            'Turgor pressure (MPa)': self.turgor_pressure
        }

        calculations = {
            'Number of points used for \'before TLP\'': self.bkp,
            'Water at FT (g)': self.water_FT,
            'Saturated Water Content': self.swc,
            'Osmotic potential at FT (MPa)': self.os_pot_FT,
            'TLP (MPa)': self.tlp,
            'Bulk elastic modulus (MPa)': self.bulk_elastic_total,
            'RWC at TLP': self.rwc_tlp,
            'Bulk elastic modulus (symplastic) (MPa)': self.bulk_elastic_total_sym,
            'RWC at TLP (symplastic)': self.rwc_tlp_sym,
            'Apoplastic Water Fraction': self.awf,
            'Hydraulic capacitance before TLP (g MPa^-1)': self.ct_before_massnorm,
            'Hydraulic capacitance before TLP (cm^2 MPa^-1)': self.ct_before_areanorm,
            'Water storage capacity (g)': self.capacity_massnorm,
            'Water storage capacity (cm^2)': self.capacity_areanorm,
            'Water storage capacity (g) (gravity corrected)': self.capacity_massnorm_gravity,
            'Water storage capacity (cm^2) (gravity corrected)': self.capacity_areanorm_gravity,
            'Hydraulic capacitance after TLP (g MPa^-1)': self.ct_after_massnorm,
            'Hydraulic capacitance after TLP (cm^2 MPa^-1)': self.ct_after_areanorm,
            'R2 before TLP': self.r2_beforetlp,
            'R2 after TLP': self.r2_aftertlp,
            'R2': self.r2
        }

        # create an excel spreadsheet where each column is a different variable from data
        # then there's a blank column
        # and then a column with the keys from calculations and the values from calculations
        df_data = pd.DataFrame(data)
        df_calculations = pd.DataFrame([calculations.keys(), calculations.values()]).T
        df_calculations.columns = ['Variable Name', 'Value']
        df = pd.concat([df_data, pd.DataFrame(columns=['']), df_calculations], axis=1)
        df.to_excel(filename, index=False)

        absolute_path = os.path.abspath(filename)
        logging.info(f"Data saved to {absolute_path}")



    def plot(self):
        """
        Plot the pressure-volume curve.
        """
        fig, ax = plt.subplots(1,2, figsize=(12,6))

        # plot the pressure-volume curve
        ax[0].axvline(1, color='black', linewidth=0.25)
        ax[0].plot(self.rwc, self.inverse_psis, color='black')
        ax[0].plot(self.rwc[:self.bkp], self.inverse_psis[:self.bkp], 'o', color='blue', label='Before TLP')
        ax[0].plot(self.rwc[self.bkp:], self.inverse_psis[self.bkp:], 'o', color='red', label='After TLP')        
        slope, intercept = np.polyfit(self.rwc[self.bkp:], self.inverse_psis[self.bkp:], 1)
        x_range = np.arange(1, np.min(self.rwc), -0.01)
        ax[0].plot(x_range, slope*x_range+intercept, '--', color='black', linewidth=0.5)
        ax[0].invert_xaxis()
        ax[0].set_title('Pressure-Volume Curve')
        ax[0].set_xlabel('RWC')
        ax[0].set_ylabel('-1/Ψ (MPa)')
        ax[0].legend()

        # plot in linear space
        ax[1].axhline(0, color='black', linewidth=0.25)
        ax[1].scatter(self.water_FT, 0, color='black', label='Water at FT', facecolor='none')
        ax[1].plot(self.water_mass, self.psis, color='black')
        ax[1].plot(self.water_mass[:self.bkp], self.psis[:self.bkp], 'o', color='blue', label='Before TLP')
        ax[1].plot(self.water_mass[self.bkp:], self.psis[self.bkp:], 'o', color='red', label='After TLP')
        slope, intercept = np.polyfit(self.water_mass[:self.bkp], self.psis[:self.bkp], 1)
        x_range = np.arange(np.min(self.water_mass), np.max(self.water_mass), 0.01)
        ax[1].plot(x_range, slope*x_range+intercept, '--', color='black', linewidth=0.5)
        ax[1].set_title('Saturated Water Content')
        ax[1].set_xlabel('Water Mass (g)')
        ax[1].set_ylabel('Ψ (MPa)')
        ax[1].legend()

        plt.tight_layout()
        plt.show()


    def set_breakpoint(self, bkp: int):
        """
        Set the breakpoint for the PVCurve object and recalculate everything
        """
        new_PVcurve = PVCurve(self.psis, self.masses, self.dry_mass, psi_unit = self.psi_unit, mass_unit = self.mass_unit, dry_mass_unit = self.dry_mass_unit,
                              leaf_area = self.leaf_area, height = self.height, bkp = bkp, tlp_conf = self.tlp_conf, psi_sign_convention = self.psi_sign_convention, validate_data = self.validate_data)
        return new_PVcurve
    
    
    def remove_outliers(self, method: str = 'regression', **kwargs):
        """
        Remove outliers from the PVCurve object. Mostly for catching transcription errors!
        inputs:
            plot: whether to plot the outlier removal process (default is False) 
            method: method for outlier removal (default is 'regression')
                'regression': remove points outside of a confidence interval around the piecewise regression lines
                'local_mad': remove points based on local median absolute deviation
        outputs:
            new_pvcurve: PVCurve object with outliers removed according to the specified method        
        """  
        ## TODO: #1 add a plot option to show the outliers that were removed

        if method == 'regression':
            # Remove outliers using a confidence interval around the piecewise regression lines.
            if 'alpha' not in kwargs:
                logging.warning("No 'alpha' keyword argument provided for regression outlier removal. Using default alpha=0.05 for 95% confidence interval.")
            alpha = kwargs.get('alpha', 0.05)
            
            new_pvcurve = regression_removal(self, alpha=alpha)

        elif method == 'local_mad':
            # Remove outliers using local median absolute deviation.
            
            if 'window' not in kwargs:
                logging.warning("No 'window' keyword argument provided for local MAD outlier removal. Using default window=5 (i.e., 5 data points will be used to calculate local median absolute deviation).") 
            window = kwargs.get('window', 5)
            if window < 3 or window % 2 == 0:
                raise ValueError("Window size must be a positive odd integer greater than or equal to three.")
            if window > len(self.psis):
                raise ValueError("Window size cannot be larger than the number of data points in the PVCurve.") 

            if 'cutoff' not in kwargs:
                logging.warning("No 'cutoff' keyword argument provided for local MAD outlier removal. Using default cutoff=4 (i.e., points will be removed if they are more than 4 median absolute deviations from the local median).")
            cutoff = kwargs.get('cutoff', 4)
            if cutoff <= 0:
                raise ValueError("Cutoff must be a positive number.")
            if 'protect_edges' not in kwargs:
                logging.warning("No 'protect_edges' keyword argument provided for local MAD outlier removal. Using default protect_edges=2 (i.e., first and last two points will not be considered for outlier removal).")
            protect_edges = kwargs.get('protect_edges', 2)
            if protect_edges < 0:
                raise ValueError("protect_edges must be a non-negative integer.")
            if protect_edges * 2 >= len(self.psis):
                raise ValueError("protect_edges cannot be larger than or equal to half the number of data points in the PVCurve. Otherwise you're just not removing outliers at all :)")

            new_pvcurve = local_mad_removal(self, window=window, cutoff=cutoff)
        else:
            raise ValueError(f"Unsupported outlier removal method ({method}). Please use 'regression' or 'local_mad'.")

        return new_pvcurve
        

    def add_point(self, psi: float, mass: float):
        """
        Add a point to the PVCurve object and recalculate everything.
        """
        new_psis = np.append(self.psis, psi)
        new_masses = np.append(self.masses, mass)
        new_pvcurve = PVCurve(new_psis, new_masses, self.dry_mass, psi_unit = self.psi_unit, mass_unit = self.mass_unit, dry_mass_unit = self.dry_mass_unit,
                              leaf_area = self.leaf_area, height = self.height, bkp = self.bkp, tlp_conf = self.tlp_conf, psi_sign_convention = self.psi_sign_convention, validate_data = self.validate_data)
        return new_pvcurve
    
    def add_points(self, psis: np.ndarray, masses: np.ndarray):
        """
        Add multiple points to the PVCurve object and recalculate everything.
        """
        new_psis = np.append(self.psis, psis)
        new_masses = np.append(self.masses, masses)
        new_pvcurve = PVCurve(new_psis, new_masses, self.dry_mass, psi_unit = self.psi_unit, mass_unit = self.mass_unit, dry_mass_unit = self.dry_mass_unit,
                              leaf_area = self.leaf_area, height = self.height, bkp = self.bkp, tlp_conf = self.tlp_conf, psi_sign_convention = self.psi_sign_convention, validate_data = self.validate_data)
        return new_pvcurve
    
    def get_breakpoint(self, plot=False):
        """
        Get the breakpoint for the PVCurve object.
        """
        if plot:
            get_breakpoint(self.psis, self.masses, self.dry_mass, psi_unit = self.psi_unit, mass_unit = self.mass_unit, dry_mass_unit = self.dry_mass_unit,
                              leaf_area = self.leaf_area, height = self.height, psi_sign_convention = self.psi_sign_convention, validate_data = self.validate_data, plot=True)
        return self.bkp





### outlier removal support functions

# linear regression confidence interval
def regression_removal(pvcurve: PVCurve, alpha: float = 0.05):
    """
    Remove outliers from a PVCurve object using regression-based method.
    inputs:
        pvcurve: PVCurve object
        alpha: significance level for confidence interval (default is 0.05 for 95% confidence interval)
    outputs:
        new_pvcurve: PVCurve object with outliers removed
    """
    if alpha <= 0 or alpha >= 1:
        raise ValueError("Confidence interval must be between 0 and 1 (e.g., conf=0.05 for 95% confidence interval).")

    # before TLP (regress water mass and psi)
    x = pvcurve.masses[:pvcurve.bkp]
    x = sm.add_constant(x)
    y = pvcurve.psis[:pvcurve.bkp]
    model = sm.OLS(y, x).fit()
    y_hat = model.get_prediction(x).summary_frame(alpha=alpha)
    mask1 = (y_hat['obs_ci_lower'].to_numpy() > y) | (y_hat['obs_ci_upper'].to_numpy() < y)
    outliers1_idx = np.where(mask1)[0].tolist()

    # after TLP (regress -1/psi and RWC)
    x = pvcurve.inverse_psis[pvcurve.bkp:]
    x = sm.add_constant(x)
    y = pvcurve.rwc[pvcurve.bkp:]
    model = sm.OLS(y, x).fit()
    y_hat = model.get_prediction(x).summary_frame(alpha=alpha)
    mask2 = (y_hat['obs_ci_lower'].to_numpy() > y) | (y_hat['obs_ci_upper'].to_numpy() < y)
    outliers2_idx = (pvcurve.bkp + np.where(mask2)[0]).tolist()
    logging.info('Detected {} outlier(s) using regression method.'.format(len(outliers1_idx) + len(outliers2_idx)))
    logging.info("Recalculating breakpoint after outlier removal...")
    new_pvcurve = pvcurve.remove_point(outliers1_idx + outliers2_idx)

    return new_pvcurve
        
# local median absolute deviation
def local_mad_removal(pvcurve: PVCurve, window: int = 5, cutoff: float = 4, protect_edges: int = 2):
    """
    Remove outliers from a PVCurve object using local median absolute deviation method.
    inputs:
        pvcurve: PVCurve object
        window: size of the moving window (default is 5)
        cutoff: number of median absolute deviations to use as cutoff (default is 4)
        protect_edges: number of points at each edge to omit from outlier detection (default is 2)
    outputs:
        new_pvcurve: PVCurve object with outliers removed
    """
    # TODO: add some way to address outliers on the edges (e.g., first and last few points). right now they are just skipped.

    n = len(pvcurve)
    half_window = window // 2
    outlier_indices = []
    for i in range(n):
        if i < protect_edges or i >= n - protect_edges:
            continue  # skip edge points. again, it'd be nice to have a way to deal with these

        # define the window
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)

        # masses first...
        window_data = pvcurve.masses[start:end]
        # calculate the median and MAD
        median = np.median(window_data)
        mad = np.median(np.abs(window_data - median))
        if mad == 0:
            continue  # avoid division by zero
        # calculate the modified z-score
        modified_z_score = 0.6745 * (pvcurve.masses[i] - median) / mad
        # check if the point is an outlier
        if np.abs(modified_z_score) > cutoff:
            outlier_indices.append(i)
            continue

        # now psis
        window_data = pvcurve.psis[start:end]
        median = np.median(window_data)
        mad = np.median(np.abs(window_data - median))
        if mad == 0:
            continue  # avoid division by zero
        modified_z_score = 0.6745 * (pvcurve.psis[i] - median) / mad
        if np.abs(modified_z_score) > cutoff:
            outlier_indices.append(i)

    logging.info('Detected {} outlier(s) using local MAD method.'.format(len(outlier_indices)))
    psis = np.delete(pvcurve.psis, outlier_indices)
    masses = np.delete(pvcurve.masses, outlier_indices)
    # bkp is set back to 0 so it will recalculate
    logging.info("Recalculating breakpoint after outlier removal...")
    new_pvcurve = PVCurve(psis, masses, pvcurve.dry_mass, psi_unit = pvcurve.psi_unit, mass_unit = pvcurve.mass_unit, dry_mass_unit = pvcurve.dry_mass_unit,
                        leaf_area = pvcurve.leaf_area, height = pvcurve.height, bkp = 0, tlp_conf = pvcurve.tlp_conf, psi_sign_convention = pvcurve.psi_sign_convention, validate_data = pvcurve.validate_data)

    return new_pvcurve


def get_breakpoint(psis, masses, dry_mass, psi_unit='MPa', mass_unit='g', dry_mass_unit='g', 
                   leaf_area=np.nan, height=0, psi_sign_convention='negative', validate_data=True,
                   return_r2s=False, plot=False):
    """
    Function to find the breakpoint for a pressure-volume curve.
    Here we define the breakpoint based on the fit that results in the maximum R2 value.
    inputs:
        psis: numpy array of leaf water potentials (MPa)
        masses: numpy array of masses (g)
        dry_mass: mass of dry leaf (g)
    outputs:
        breakpoint: index of the breakpoint
    """    

    # drop nan values to avoid screwing with np.polyfit later
    valid = np.isfinite(psis) & np.isfinite(masses)
    psis = psis[valid]
    masses = masses[valid]
    
    # if we don't have at least four points tell the user we can't calculate a breakpoint
    if len(psis) < 4:
        raise ValueError("Need at least four non-NaN data points to calculate a breakpoint.")


    # initialize variables
    possible_bkps = np.arange(2, len(psis)-1)  # need at least two points on either side of the breakpoint
    r2s = []
    tlps = []

    # loop through all possible breakpoints
    for i in possible_bkps:
        PVCurve_temp = PVCurve(
            psis, masses, dry_mass,
            bkp=i,
            psi_unit=psi_unit,
            mass_unit=mass_unit,
            dry_mass_unit=dry_mass_unit,
            leaf_area=leaf_area,
            height=height,
            psi_sign_convention=psi_sign_convention,
            validate_data=validate_data,
        )
        r2s.append(PVCurve_temp.r2)
        tlps.append(PVCurve_temp.tlp)
    
    # find the breakpoint that maximizes R2
    breakpoint = np.argmax(r2s) + 2


    if plot:
        bad_breakpoints = [i for i in possible_bkps if i != breakpoint]
        bad_r2s = [r2s[i-2] for i in bad_breakpoints]
        bad_tlps = [tlps[i-2] for i in bad_breakpoints]

        fig, ax = plt.subplots(1,2, figsize=(12,6))

        ax[0].plot(possible_bkps, r2s, color='black', zorder=0, linewidth=0.5) 
        ax[0].scatter(bad_breakpoints, bad_r2s, color='black', marker='o', zorder=1) 
        ax[0].scatter(breakpoint, r2s[breakpoint-2], marker='*', color='red', zorder=1, s=100)
        ax[0].set_title('R² vs. Breakpoint')
        ax[0].set_xlabel('Breakpoint (# of points used for pre-TLP)')
        ax[0].set_ylabel('R²')

        ax[1].plot(possible_bkps, tlps, color='black', zorder=0, linewidth=0.5)
        ax[1].scatter(bad_breakpoints, bad_tlps, color='black', marker='o', zorder=1)
        ax[1].scatter(breakpoint, tlps[breakpoint-2], marker='*', color='red', zorder=1, s=100)
        ax[1].set_title('Turgor Loss Point Estimate')
        ax[1].set_xlabel('Breakpoint (# of points used for pre-TLP)')
        ax[1].set_ylabel('Turgor Loss Point (MPa)')

        plt.tight_layout()
        plt.show()

    logging.info(f"Breakpoint found at index {breakpoint} with R² of {r2s[breakpoint-2]:.3f}")

    if return_r2s:
        return breakpoint, r2s
    return breakpoint

def _read_df(
        df: pd.DataFrame,
        psi_column: str = None,
        mass_column: str = None,
        dry_mass_column: str = None,
        dry_mass: float = None,
        psi_units: str = None,
        mass_units: str = None,
        dry_mass_units: str = None,
        bkp: int = 0,
    ) -> PVCurve:
    """
    Helper function to read a dataframe into a PVCurve object.
    inputs:
        df: pandas DataFrame containing the data
        psi_column: name of the column containing Ψ values (if None, will try to infer)
        mass_column: name of the column containing mass values (if None, will try to infer)
        dry_mass_column: name of the column containing dry mass values (if None, will try to infer)
        dry_mass: value of the dry mass (if None, will try to infer from column)
        psi_units: units of the Ψ values (if None, will default to MPa or try to infer from column name)
        mass_units: units of the mass values (if None, will default to g or try to infer from column name)
        dry_mass_units: units of the dry mass value (if None, will default to g or try to infer from column name)
        bkp: breakpoint index (if 0, will try to infer)
    outputs:
        pvcurve: PVCurve object with the data from the CSV file
    """

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # function for inferring column names with fuzzy matching
    norm_cols = {c.strip().lower(): c for c in df.columns}
    def _infer_column(candidates):
        for cand in candidates:
            key = cand.strip().lower()
            if key in norm_cols:
                return norm_cols[key]
        return None

    if psi_column is None:
        reasonable_psi_names = [
            "psi", "ψ", "Ψ", "water potential", "water potential (mpa)", "y (mpa)", "Ψ (mpa)", "ψ (mpa)",
            "water potential (bar)", "y (bar)", "Ψ (bar)", "ψ (bar)",
            "water potential (kpa)", "y (kpa)", "Ψ (kpa)", "ψ (kpa)"
        ]
        psi_column = _infer_column(reasonable_psi_names)
        if psi_column is None:
            psi_column = "Ψ (MPa)"
        if psi_column in df.columns:
            logging.info(f"No value given for psi_column. Using column '{psi_column}' for Ψ values...")
        else:
            raise ValueError("No value given for psi_column and no reasonable column name found for Ψ values.")
    # raise an error of the psi_column is not in the data sheet
    if psi_column not in df.columns:
        raise ValueError(f"Column '{psi_column}' not found in dataframe columns.")

    # if there are fewer than 2 non-nan values for psi, raise an error at this point
    if df[psi_column].dropna().shape[0] < 2:
        raise ValueError(f"Column '{psi_column}' has fewer than 2 non-nan values. Is this the correct column for Ψ values?")
    


    # if 'bar' or 'kpa' is in the column name, set the units appropriately
    if psi_units is None:
        if 'bar' in psi_column.lower():
            logging.info(f"Detected 'bar' in psi_column name '{psi_column}'. Assuming units are in bar.")
            psi_units = 'bar'
        elif 'kpa' in psi_column.lower():
            logging.info(f"Detected 'kpa' in psi_column name '{psi_column}'. Assuming units are in kPa.")
            psi_units = 'kPa'
        else: 
            logging.info(f"Assuming Ψ units are in MPa.")
            psi_units = 'MPa'

    # infer mass column based on df keys
    if mass_column is None:
        reasonable_mass_names = ["mass", "mass (g)", "weight", "weight (g)", "wet mass", "wet mass (g)", "mass (mg)", "weight (mg)", "wet mass (mg)"]
        mass_column = _infer_column(reasonable_mass_names) or "Mass (g)"
        if mass_column in df.columns:
            logging.info(f"No value given for mass_column. Using column '{mass_column}' for mass values...")
        else:
            raise ValueError("No value given for mass_column and no reasonable column name found for mass values.")
    # raise an error of the mass_column is not in the data sheet
    if mass_column not in df.columns:
        raise ValueError(f"Column '{mass_column}' not found in dataframe columns.")

    # detecting units for mass
    if mass_units is None:
        if 'mg' in mass_column.lower():
            logging.info(f"Detected 'mg' in mass_column name '{mass_column}'. Assuming units are in mg.")
            mass_units = 'mg'
        else:
            logging.info(f"Assuming mass units are in grams.")
            mass_units = 'g'

    # if there are fewer than 2 non-nan values in mass data, raise an error at this point
    if df[mass_column].dropna().shape[0] < 2:
        raise ValueError(f"Column '{mass_column}' has fewer than 2 non-nan values. Is this the correct column for mass values?")

    if dry_mass is None:
        if dry_mass_column is None:
            dm_col = _infer_column(["dry mass (g)", "dry mass", "dry weight (g)", "dry weight", "dry", "dry (g)", "dry mass (mg)", "dry weight (mg)", "dry (mg)"])
            dry_mass_column = dm_col

        if dry_mass_column is None or dry_mass_column not in df.columns:
            raise ValueError(
                "Dry mass not provided and no dry mass column found. "
                "Pass dry_mass=... or specify dry_mass_column=..."
            )

        if dry_mass_column not in df.columns:
            raise ValueError(f"Column '{dry_mass_column}' not found in dataframe columns.")
        
        dm_vals = df[dry_mass_column].dropna().values
        if dm_vals.size == 0:
            raise ValueError(f"Dry mass column '{dry_mass_column}' exists but contains no values.")

        # Consistency check
        if dm_vals.size > 1:
            logging.warning(
                f"Multiple dry mass values found under '{dry_mass_column}' ({dm_vals[:5]}...). "
                "Using the first value."
            )
        dry_mass = float(dm_vals[0])
        logging.info(f"Using dry mass value of {dry_mass} from '{dry_mass_column}' column.")

    if dry_mass_units is None:
        if dry_mass_column and 'mg' in dry_mass_column.lower():
            logging.info(f"Detected 'mg' in dry_mass_column name '{dry_mass_column}'. Assuming units are in mg.")
            dry_mass_units = 'mg'
        else: 
            logging.info(f"No unit for dry mass found. Assuming dry mass units are the same as mass units.")
            dry_mass_units = mass_units
    
    # create the PVCurve object
    pvcurve = PVCurve(df[psi_column].values, df[mass_column].values, dry_mass, bkp=bkp, psi_unit=psi_units, mass_unit=mass_units, dry_mass_unit=dry_mass_units)
    return pvcurve


def read_csv(
        filename: str,
        **kwargs
    ) -> PVCurve:
    """
    Read a CSV file into a PVCurve object.
    inputs:
        filename: path to the CSV file
    outputs:
        pvcurve: PVCurve object with the data from the CSV file
    """

    # read in the data
    df = pd.read_csv(filename)
    # convert all column names to stripped strings for easier matching
    df.columns = df.columns.astype(str).str.strip()
    pvcurve = _read_df(
        df,
        **kwargs
    )
    return pvcurve

    

def read_excel(
        filename: str,
        **kwargs
    ) -> PVCurve:
    """
    Read an Excel file into a PVCurve object.
    inputs:
        filename: path to the Excel file
        psi_column: name of the column containing Ψ values (if None, will try to infer)
        mass_column: name of the column containing mass values (if None, will try to infer)
        dry_mass_column: name of the column containing dry mass values (if None, will try to infer)
        dry_mass: value of the dry mass (if None, will try to infer from column)
        psi_units: units of the Ψ values (if None, will default to MPa or try to infer from column name)
        mass_units: units of the mass values (if None, will default to g or try to infer from column name)
        dry_mass_units: units of the dry mass value (if None, will default to g or try to infer from column name)
    outputs:
        pvcurve: PVCurve object with the data from the Excel file
    """
    # read in the data
    df = pd.read_excel(filename)
    # convert all column names to stripped strings for easier matching
    df.columns = df.columns.astype(str).str.strip()
    pvcurve = _read_df(
        df,
        **kwargs
    )
    return pvcurve

def read(filename: str, **kwargs) -> PVCurve:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return read_csv(filename, **kwargs)
    if suffix in [".xlsx", ".xls", ".xlsm"]:
        return read_excel(filename, **kwargs)
    raise ValueError(f"Unsupported file type: {suffix}")