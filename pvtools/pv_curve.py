import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# object class called a pv curve
class PVCurve:
    def __init__(self, psis: np.ndarray, masses: np.ndarray, dry_mass: float, leaf_area: float = np.nan, height: float = 0, bkp: int = 0):
        """
        PVCurve object that stores data and basic calculations for a pressure-volume curve.
        Required inputs:
            psis: numpy array of leaf water potentials (MPa)
            masses: numpy array of masses (g)
            dry_mass: mass of dry leaf (g)
        Optional inputs:
            leaf_area: leaf area (cm^2)
            height: height of the sample (m)
            bkp: breakpoint index (default is 0, which will be calculated -- alternatively, provide the # of points that should be included in the first segment)
        """
        # check that the size of the water potential and mass data are the same
        if psis.shape != masses.shape:
            raise ValueError("Ψ and mass data are not the same shape. If there are missing measurements, please fill in with np.nan.")
        # check that the breakpoint is within the range of the data
        if bkp < 0 or bkp > len(psis):
            raise ValueError("Breakpoint is either negative or larger than the number of points in the dataset.")


        # input data...
        self.psis = psis
        self.masses = masses
        self.dry_mass = dry_mass
        self.leaf_area = leaf_area
        self.height = height
        self.bkp = bkp

        # basic calculations
        self.inverse_psis = -1 / psis
        self.water_mass = masses - dry_mass

        # get breakpoint if not provided
        if bkp == 0:
            self.bkp = get_breakpoint(psis, masses, dry_mass)

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

        # turgor loss point
        slope, intercept = np.polyfit(self.sol_pot[:self.bkp], self.turgor_pressure[:self.bkp], 1)
        self.tlp_slope = slope
        self.tlp = intercept / self.tlp_slope

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
    
    # print function
    def __repr__(self):
        return "TLP: {:.3f} MPa \nR2: {:.3f}".format(self.tlp, self.r2)   
    
    # save out to an excel sheet
    def save_excel(self, filename: str):
        """
        Save the PVCurve object to an Excel file.
        """

        # check that the filename ends in .xlsx; if not throw exception
        if not filename.endswith('.xlsx'):
            raise ValueError("Filename must end in .xlsx")

        # create a dictionary of the data
        data = {
            'Ψ (MPa)': self.psis,
            'Mass (g)': self.masses,
            '-1/Ψ (MPa^-1)': self.inverse_psis,
            'Water mass (g)': self.water_mass,
            'RWC': self.rwc,
            'Solute potential (MPa)': self.sol_pot,
            'Turgor pressure (MPa)': self.turgor_pressure
        }

        calculations = {
            'Number of points used for \'before TLP\'': self.bkp,
            'Dry Mass (g)': self.dry_mass,
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
        print(f"Data saved to {absolute_path}")

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
        new_PVcurve = PVCurve(self.psis, self.masses, self.dry_mass, self.leaf_area, self.height, bkp)
        return new_PVcurve











def get_breakpoint(psis, masses, dry_mass, return_r2s=False, plot=False):
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
    # initialize variables
    possible_bkps = np.arange(2, len(psis)-1)  # need at least two points on either side of the breakpoint
    r2s = []
    tlps = []

    # loop through all possible breakpoints
    for i in possible_bkps:
        PVCurve_temp = PVCurve(psis, masses, dry_mass, bkp=i)
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


    if return_r2s:
        return breakpoint, r2s
    return breakpoint