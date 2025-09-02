import pickle
from io import BytesIO

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal


from ..datasets import load_robot
from ..sqrt import BiermanKalmanFilter, CholeskyKalmanFilter
from ..standard import KalmanFilter

KALMAN_FILTERS = [KalmanFilter, BiermanKalmanFilter, CholeskyKalmanFilter]


@pytest.fixture(params=KALMAN_FILTERS)
def kf_cls(request):
    return request.param


@pytest.fixture
def data():
    return load_robot()


class TestKalmanFilter:
    """All of the actual tests to check against an implementation of the usual
    Kalman Filter. Abstract so that sister implementations can reuse these
    tests.
    """

    def test_kalman_sampling(self, kf_cls, data):
        # Augmented: use differing sample sizes and matrix dimensions for edge cases
        # Test with sample size = 1 (minimal), and sample size = 10000 (large)
        # Test with zero transition offset, negative initial state mean
        # FIX: do not pass transition_offsets and transition_offsets of length 500 if sample size > 500
        # Instead, use scalars for offsets to allow arbitrary sampling length
        # The rest remains the same
        kf = kf_cls(
            data.transition_matrix * 2.0,  # scale to make data nontrivial
            data.observation_matrix + 0.1,  # add a small value to all entries
            data.transition_covariance * 5,  # increase variance
            data.observation_covariance * 5,  # increase observation variance
            np.zeros(data.transition_matrix.shape[0]),  # match transition offset shape
            data.observation_offset * -1.0,  # negative observation offset
            np.full_like(data.initial_state_mean, -1000.0),  # large negative initial state mean
            data.initial_state_covariance * 2.0,  # scale covariance up
        )

        # Test with edge case sample size: 1
        x1, z1 = kf.sample(1)
        assert x1.shape == (1, data.transition_matrix.shape[0])
        assert z1.shape == (1, data.observation_matrix.shape[0])

        # Test with very large sample size: 10000
        x2, z2 = kf.sample(10000)
        assert x2.shape == (10000, data.transition_matrix.shape[0])
        assert z2.shape == (10000, data.observation_matrix.shape[0])


    def test_kalman_filter_update(self, kf_cls, data):
        kf = kf_cls(
            data.transition_matrix,
            data.observation_matrix,
            data.transition_covariance,
            data.observation_covariance,
            data.transition_offsets,
            data.observation_offset,
            data.initial_state_mean,
            data.initial_state_covariance,
        )

        # use Kalman Filter
        (x_filt, V_filt) = kf.filter(X=data.observations)

        # use online Kalman Filter
        n_timesteps = data.observations.shape[0]
        n_dim_obs, n_dim_state = data.observation_matrix.shape
        kf2 = kf_cls(n_dim_state=n_dim_state, n_dim_obs=n_dim_obs)
        x_filt2 = np.zeros((n_timesteps, n_dim_state))
        V_filt2 = np.zeros((n_timesteps, n_dim_state, n_dim_state))
        for t in range(n_timesteps - 1):
            if t == 0:
                x_filt2[0] = data.initial_state_mean
                V_filt2[0] = data.initial_state_covariance
            (x_filt2[t + 1], V_filt2[t + 1]) = kf2.filter_update(
                x_filt2[t],
                V_filt2[t],
                observation=data.observations[t + 1],
                transition_matrix=data.transition_matrix,
                transition_offset=data.transition_offsets[t],
                transition_covariance=data.transition_covariance,
                observation_matrix=data.observation_matrix,
                observation_offset=data.observation_offset,
                observation_covariance=data.observation_covariance,
            )
        assert_array_almost_equal(x_filt, x_filt2)
        assert_array_almost_equal(V_filt, V_filt2)

    def test_kalman_filter(self, kf_cls, data):
        kf = kf_cls(
            data.transition_matrix,
            data.observation_matrix,
            data.transition_covariance,
            data.observation_covariance,
            data.transition_offsets,
            data.observation_offset,
            data.initial_state_mean,
            data.initial_state_covariance,
        )

        (x_filt, V_filt) = kf.filter(X=data.observations)
        assert_array_almost_equal(
            x_filt[:500], data.filtered_state_means[:500], decimal=7
        )
        assert_array_almost_equal(
            V_filt[:500], data.filtered_state_covariances[:500], decimal=7
        )

    def test_kalman_predict(self, kf_cls, data):
        kf = kf_cls(
            data.transition_matrix,
            data.observation_matrix,
            data.transition_covariance,
            data.observation_covariance,
            data.transition_offsets,
            data.observation_offset,
            data.initial_state_mean,
            data.initial_state_covariance,
        )

        x_smooth = kf.smooth(X=data.observations)[0]
        assert_array_almost_equal(
            x_smooth[:501], data.smoothed_state_means[:501], decimal=7
        )

    def test_kalman_fit(self, kf_cls, data):
        # check against MATLAB dataset
        kf = kf_cls(
            data.transition_matrix,
            data.observation_matrix,
            data.initial_transition_covariance,
            data.initial_observation_covariance,
            data.transition_offsets,
            data.observation_offset,
            data.initial_state_mean,
            data.initial_state_covariance,
            em_vars=["transition_covariance", "observation_covariance"],
        )

        loglikelihoods = np.zeros(5)
        for i in range(len(loglikelihoods)):
            loglikelihoods[i] = kf.loglikelihood(data.observations)
            kf.em(X=data.observations, n_iter=1)

        assert np.allclose(loglikelihoods, data.loglikelihoods[:5])

        # check that EM for all parameters is working
        kf.em_vars = "all"
        n_timesteps = 30
        for i in range(len(loglikelihoods)):
            kf.em(X=data.observations[0:n_timesteps], n_iter=1)
            loglikelihoods[i] = kf.loglikelihood(data.observations[0:n_timesteps])
        for i in range(len(loglikelihoods) - 1):
            assert (loglikelihoods[i] < loglikelihoods[i + 1]).all()

    def test_kalman_initialize_parameters(self, kf_cls):
        self.check_dims(5, 1, {"transition_matrices": np.eye(5)}, kf_cls)
        self.check_dims(1, 3, {"observation_offsets": np.zeros(3)}, kf_cls)
        self.check_dims(
            2,
            3,
            {"transition_covariance": np.eye(2), "observation_offsets": np.zeros(3)},
            kf_cls,
        )
        self.check_dims(3, 2, {"n_dim_state": 3, "n_dim_obs": 2}, kf_cls)
        self.check_dims(4, 1, {"initial_state_mean": np.zeros(4)}, kf_cls)

    def check_dims(self, n_dim_state, n_dim_obs, kwargs, kf_cls):
        kf = kf_cls(**kwargs)
        (
            transition_matrices,
            transition_offsets,
            transition_covariance,
            observation_matrices,
            observation_offsets,
            observation_covariance,
            initial_state_mean,
            initial_state_covariance,
        ) = kf._initialize_parameters()
        assert transition_matrices.shape == (n_dim_state, n_dim_state)
        assert transition_offsets.shape == (n_dim_state,)
        assert transition_covariance.shape == (n_dim_state, n_dim_state)
        assert observation_matrices.shape == (n_dim_obs, n_dim_state)
        assert observation_offsets.shape == (n_dim_obs,)
        assert observation_covariance.shape == (n_dim_obs, n_dim_obs)
        assert initial_state_mean.shape == (n_dim_state,)
        assert initial_state_covariance.shape == (n_dim_state, n_dim_state)

    def test_kalman_pickle(self, kf_cls, data):
        kf = kf_cls(
            data.transition_matrix,
            data.observation_matrix,
            data.transition_covariance,
            data.observation_covariance,
            data.transition_offsets,
            data.observation_offset,
            data.initial_state_mean,
            data.initial_state_covariance,
            em_vars="all",
        )

        # train and get log likelihood
        X = data.observations[0:10]
        kf = kf.em(X, n_iter=5)
        loglikelihood = kf.loglikelihood(X)

        # pickle Kalman Filter
        store = BytesIO()
        pickle.dump(kf, store)

        # check that parameters came out already
        np.testing.assert_almost_equal(loglikelihood, kf.loglikelihood(X))