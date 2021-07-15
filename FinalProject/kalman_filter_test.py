"""
    -*- coding: utf-8 -*-
    Time    : 2021-07-02 10:38 a.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: kalman_filter_test
    
    {Description}
    -------------
    
"""

from signal_analysis_tools.utility_import import *


class BearingStateTransitionModel(StateTransitionModel):
    def create_model(self):
        self.model = np.diag(np.ones(2, dtype=float))

    def set_dt(self, dt):
        self.model[0, 1] = dt


class BearingObservationCovarianceModel(ObservationCovarianceModel):
    def create_model(self):
        self.model = np.zeros((1, 1), dtype=float)

    def set_covariances(self, errors):
        # Errors assumed to be in stddev
        np.fill_diagonal(self.model, errors**2)


if __name__ == '__main__':

    truth_bearing = 90.0  # deg
    ambiguous_bearing = 270.0  # deg
    bearing_noise = 5.0  # deg
    dt = 1.0  # s
    num_records = 100

    truth_filter = KalmanFilter()
    truth_filter.state_model = np.zeros((2, 1), dtype=float)
    truth_filter.state_covariance = np.zeros((2, 2), dtype=float)
    truth_filter.state_transition_model = BearingStateTransitionModel()
    truth_filter.observation_model = np.zeros((1, 2), dtype=float)
    truth_filter.observation_model[0, 0] = 1
    truth_filter.observation_covariance_model = BearingObservationCovarianceModel()
    # truth_filter.process_noise_covariance = np.zeros(2, dtype=float)
    truth_filter.process_noise_covariance = np.array([1e-5, 1e-5])

    ambiguous_filter = KalmanFilter()
    ambiguous_filter.state_model = np.zeros((2, 1), dtype=float)
    ambiguous_filter.state_covariance = np.zeros((2, 2), dtype=float)
    ambiguous_filter.state_transition_model = BearingStateTransitionModel()
    ambiguous_filter.observation_model = np.zeros((1, 2), dtype=float)
    ambiguous_filter.observation_model[0, 0] = 1
    ambiguous_filter.observation_covariance_model = BearingObservationCovarianceModel()
    # ambiguous_filter.process_noise_covariance = np.zeros(2, dtype=float)
    ambiguous_filter.process_noise_covariance = np.array([1e-5, 1e-5])

    noise_estimates = np.random.randn(num_records) * bearing_noise

    truth_bearing_estimates = truth_bearing + noise_estimates
    ambiguous_bearing_estimates = ambiguous_bearing + noise_estimates * 2

    truth_filter.state_model[0, 0] = truth_bearing_estimates[0]
    truth_filter.state_model[1, 0] = 0.0
    np.fill_diagonal(truth_filter.state_covariance, np.array([bearing_noise, 1000.]))

    ambiguous_filter.state_model[0, 0] = ambiguous_bearing_estimates[0]
    ambiguous_filter.state_model[1, 0] = 0.0
    np.fill_diagonal(ambiguous_filter.state_covariance, np.array([bearing_noise*2, 1000.]))

    truth_filter_output = np.zeros((num_records, 2))
    ambiguous_filter_output = np.zeros((num_records, 2))

    truth_filter_output[0, 0] = truth_filter.state_model[0, 0]
    truth_filter_output[0, 1] = truth_filter.state_covariance[0, 0]
    ambiguous_filter_output[0, 0] = ambiguous_filter.state_model[0, 0]
    ambiguous_filter_output[0, 1] = ambiguous_filter.state_covariance[0, 0]

    for i in range(1, num_records):
        truth_filter.update(np.array([truth_bearing_estimates[i]]), np.array([bearing_noise]), dt)
        ambiguous_filter.update(np.array([ambiguous_bearing_estimates[i]]), np.array([bearing_noise]), dt)

        truth_filter_output[i, 0] = truth_filter.state_model[0, 0]
        truth_filter_output[i, 1] = truth_filter.state_covariance[0, 0]
        ambiguous_filter_output[i, 0] = ambiguous_filter.state_model[0, 0]
        ambiguous_filter_output[i, 1] = ambiguous_filter.state_covariance[0, 0]

    t = np.arange(0, num_records)

    fig, axes = plt.subplots(3)
    axes[0].plot(t, truth_bearing_estimates, marker='x', linestyle='')
    axes[0].plot(t, truth_filter_output[:, 0])
    axes[1].plot(t, ambiguous_bearing_estimates, marker='x', linestyle='')
    axes[1].plot(t, ambiguous_filter_output[:, 0])
    axes[2].plot(t, truth_filter_output[:, 1])
    axes[2].plot(t, ambiguous_filter_output[:, 1])

    plt.show()

    dummy = 1
