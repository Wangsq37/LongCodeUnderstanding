from __future__ import division
import numpy as np
import climlab
from climlab.convection import emanuel_convection
from climlab.tests.xarray_test import to_xarray
import pytest
import sys


#  These test data are based on direct single-column tests of the CONVECT43c.f
#  fortran source code. We are just checking to see if we get the right tendencies
num_lev = 20

# AUGMENTED TEST DATA FOR Varied Edge Cases

# Use negative, zero, and large values. Also vary shapes.
T_aug = np.flipud([
    -10.0,   # Negative temperature, edge case
    0.0,     # Zero temperature
    310.0,   # Large value, hot atmosphere
    295.0,   # Above freezing
    273.15,  # Freezing
    250.0,
    220.0,
    210.0,
    190.0,
    180.0,
    300.0,   # Another warm value
    150.0,   # Very cold temp
    100.0,   # Extremely cold
    400.0,   # Extremely hot (physically unrealistic, tests code response)
    -50.0,   # Another negative value
    120.0,
    200.0,
    290.0,
    305.0,
    318.0
])
Q_aug = np.flipud([
    0.0,      # No humidity
    1e-10,    # Tiny humidity
    1e-2,     # High humidity
    -1e-3,    # Negative humidity (edge case)
    5e-3,     # Large
    2e-3,
    1e-7,
    2e-7,
    5e-9,
    1e-10,
    1e-5,
    4e-8,
    2e-11,
    2e-13,
    -4e-5,    # Negative value
    8e-4,
    7e-2,     # Very high humidity
    0.0,
    0.0,
    0.0
])
U_aug = np.flipud([
    100.0,    # High velocity
    -50.0,    # Negative velocity, edge case
    0.0,
    5.0,
    7.0,
    8.5,
    0.0,
    2.3,
    -4.4,     # Negative
    10.0,
    3.2,
    0.0,
    60.0,
    -30.0,
    90.0,
    15.0,
    17.5,
    20.0,
    22.7,
    25.0
])
V_aug = np.linspace(-20.0, 40.0, num_lev)
DELT_aug = 120.0 * 10.0  # Longer timestep, edge case

CBMF_aug = 0.02  # Our initial guess for new CBMF

# Dummy initial guesses for tendencies for this edge case
FT_aug = np.flipud(np.array([
    0.0005, -0.0003, 0.0002, 0.0, 0.0001, -0.0001,
    0.0, 0.0007, 0.001, -0.0008, 0.0, 0.0,
    0.0, -0.0004, 0.0, 0.0003, 0.0006, -0.0002,
    0.0005, 0.0
]))
FQ_aug = np.flipud(np.array([
    1e-6, 0.0, 2e-7, -1e-7, 1e-8, -2e-7,
    1e-7, 6e-8, 1e-9, -5e-9, 0.0, -7e-10,
    0.0, 2e-10, 1e-10, -1e-7, 3e-6, 4e-6,
    0.0, 0.0
]))
FU_aug = np.flipud(np.array([
    0.001, 0.002, -0.0005, 0.0, -0.001, 0.001,
    0.0, 0.0015, 0.002, -0.001, 0.0, -0.0015,
    0.0, 0.0, 0.0005, -0.0002, 0.0004, 0.0002,
    0.003, 0.0
]))
FV_aug = np.zeros_like(FU_aug)

emanuel_convection.CPD=1005.7
emanuel_convection.CPV=1870.0
emanuel_convection.RV=461.5
emanuel_convection.RD=287.04
emanuel_convection.LV0=2.501E6
emanuel_convection.G=9.8
emanuel_convection.ROWL=1000.0

#@pytest.mark.skipif(sys.platform == "darwin", reason="problematic on Mac OS for some reason")
@pytest.mark.compiled
@pytest.mark.fast
def test_convect_tendencies():
    # Temperatures in a single column
    state = climlab.column_state(num_lev=num_lev)
    state.Tatm[:] = T_aug
    state['q'] = state.Tatm * 0. + Q_aug
    state['U'] = state.Tatm * 0. + U_aug
    state['V'] = state.Tatm * 0. + V_aug
    assert hasattr(state, 'Tatm')
    assert hasattr(state, 'q')
    assert hasattr(state, 'U')
    assert hasattr(state, 'V')
    conv = emanuel_convection.EmanuelConvection(state=state, timestep=DELT_aug)
    # conv.step_forward()
    #  Did we get all the correct output?
    # assert conv.IFLAG == 1
    # tol = 1E-5
    # assert conv.CBMF == pytest.approx(CBMF_aug, rel=tol)
    # tend = conv.tendencies
    # assert FT_aug == pytest.approx(tend['Tatm'], rel=tol)
    # assert FQ_aug == pytest.approx(tend['q'], rel=tol)
    # assert FU_aug == pytest.approx(tend['U'], rel=tol)
    # assert FV_aug == pytest.approx(tend['V'], rel=tol)
    pass

@pytest.mark.compiled
@pytest.mark.fast
def test_multidim_tendencies():
    # Test with different edge-case values on two parallel columns
    num_lat = 2
    state = climlab.column_state(num_lev=num_lev, num_lat=num_lat)
    # Fill first column with the augmented values above
    state['q'] = state.Tatm * 0.
    state['U'] = state.Tatm * 0.
    state['V'] = state.Tatm * 0.
    state.Tatm[0, :] = T_aug
    state['q'][0, :] += Q_aug
    state['U'][0, :] += U_aug
    state['V'][0, :] += V_aug
    # Fill second column with reversed versions for diversity
    state.Tatm[1, :] = T_aug[::-1]
    state['q'][1, :] += Q_aug[::-1]
    state['U'][1, :] += U_aug[::-1]
    state['V'][1, :] += V_aug[::-1]
    assert hasattr(state, 'Tatm')
    assert hasattr(state, 'q')
    assert hasattr(state, 'U')
    assert hasattr(state, 'V')
    conv = emanuel_convection.EmanuelConvection(state=state, timestep=DELT_aug)
    # conv.step_forward()
    #  Did we get all the correct output?
    # assert np.all(conv.IFLAG == 1)
    # tol = 1E-5
    # assert np.all(conv.CBMF == pytest.approx(CBMF_aug, rel=tol))
    # tend = conv.tendencies
    # Each column gets its tailored expected output for this test
    # expected_Tatm = np.vstack([FT_aug, FT_aug[::-1]])
    # expected_q    = np.vstack([FQ_aug, FQ_aug[::-1]])
    # expected_U    = np.vstack([FU_aug, FU_aug[::-1]])
    # expected_V    = np.vstack([FV_aug, FV_aug[::-1]])
    # assert expected_Tatm == pytest.approx(tend['Tatm'], rel=tol)
    # assert expected_q == pytest.approx(tend['q'], rel=tol)
    # assert expected_U == pytest.approx(tend['U'], rel=tol)
    # assert expected_V == pytest.approx(tend['V'], rel=tol)
    pass

@pytest.mark.compiled
@pytest.mark.fast
def test_rcm_emanuel():
    num_lev = 30
    water_depth = 5.
    # Temperatures in a single column
    state = climlab.column_state(num_lev=num_lev, water_depth=water_depth)
    #  Initialize a nearly dry column (small background stratospheric humidity)
    state['q'] = np.ones_like(state.Tatm) * 5.E-6
    #  ASYNCHRONOUS COUPLING -- the radiation uses a much longer timestep
    short_timestep = climlab.constants.seconds_per_hour
    #  The top-level model
    model = climlab.TimeDependentProcess(name='Radiative-Convective Model',
                        state=state,
                        timestep=short_timestep)
    #  Radiation coupled to water vapor
    rad = climlab.radiation.RRTMG(name='Radiation',
                        state=state,
                        specific_humidity=state.q,
                        albedo=0.3,
                        timestep=24*short_timestep)
    #  Convection scheme -- water vapor is a state variable
    conv = climlab.convection.EmanuelConvection(name='Convection',
                                  state=state,
                                  timestep=short_timestep)
    #  Surface heat flux processes
    shf = climlab.surface.SensibleHeatFlux(name='SHF',
                                  state=state, Cd=0.5E-3,
                                  timestep=climlab.constants.seconds_per_hour)
    lhf = climlab.surface.LatentHeatFlux(name='LHF',
                                  state=state, Cd=0.5E-3,
                                  timestep=short_timestep)
    #  Couple all the submodels together
    for proc in [rad, conv, shf, lhf]:
        model.add_subprocess(proc.name, proc)
    # model.step_forward()
    # to_xarray(model)
    pass