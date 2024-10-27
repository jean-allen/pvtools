import numpy as np
import pvtools.pv_curve as pvc
import matplotlib.pyplot as plt

# test data for PVCurve object from Williams P-V
mass = np.array([4.168, 4.143, 4.127, 4.102, 4.081, 4.048, 4.008, 3.996, 3.974, 3.927, 3.897])
psis = np.array([-0.069, -0.138, -0.241, -0.345, -0.517, -0.724, -1.000, -1.172, -1.344, -1.655, -1.793])
dry_mass  = 1.73

# create a PVCurve object
pv_curve = pvc.PVCurve(psis, mass, dry_mass, bkp=9)

# test the PVCurve object against reference values in the Williams P-V curve analyzer
# as long as we're within 0.1 of the reference value, we'll consider the test passed
assert np.isclose(pv_curve.water_FT, 2.433, rtol=0.1)
assert np.isclose(pv_curve.water_FT_slope, 6.762, rtol=0.1)
assert np.isclose(pv_curve.swc, 1.406, rtol=0.1)
assert np.isclose(pv_curve.os_pot_FT_inv, 0.967, rtol=0.1)
assert np.isclose(pv_curve.os_pot_FT, -1.035, rtol=0.1)
assert np.isclose(pv_curve.os_pot_FT_slope, -3.733, rtol=0.1)
assert np.isclose(pv_curve.tlp_slope, 1.901, rtol=0.1)
assert np.isclose(pv_curve.tlp, -1.55, rtol=0.1)
assert np.isclose(pv_curve.bulk_elastic_total, 10.789, rtol=0.1)
assert np.isclose(pv_curve.rwc_tlp, 0.906, rtol=0.1)
assert np.isclose(pv_curve.awf, 0.741, rtol=0.1)
assert np.isclose(pv_curve.rwc_tlp_sym, 0.636, rtol=0.1)
assert np.isclose(pv_curve.bulk_elastic_total_sym, 2.794, rtol=0.1)
assert np.isclose(pv_curve.ct_before, 0.061, rtol=0.1)
assert np.isclose(pv_curve.ct_before_massnorm, 0.085, rtol=0.1)
assert np.isclose(pv_curve.capacity_massnorm, 0.132, rtol=0.1)
assert np.isclose(pv_curve.capacity_massnorm_gravity, 0.132, rtol=0.1)
assert np.isclose(pv_curve.ct_after, 0.09, rtol=0.1)
assert np.isclose(pv_curve.ct_after_massnorm, 0.127, rtol=0.1)

print(pv_curve)

pv_curve.plot()
plt.show()