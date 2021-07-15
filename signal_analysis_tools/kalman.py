"""
    -*- coding: utf-8 -*-
    Time    : 2021-07-02 9:47 a.m.
    Author  : Kevin Dunphy
    E-mail  : kevin.dunphy1989@gmail.com
    FileName: kalman.py
    
    {Description}
    -------------
    
"""
from abc import abstractmethod

import numpy as np


def matmul(m1, m2):

    for m in [m1, m2]:
        m.shape = m.shape + (1,) if len(m.shape) < 2 else m.shape

    result = np.matmul(m1, m2)
    result.shape = (m1.shape[0], m2.shape[1])
    return result

def matcompat(m1, m2):
    for m in [m1, m2]:
        m.shape = m.shape + (1,) if len(m.shape) < 2 else m.shape

    return m1.shape[1] == m2.shape[0], np.zeros((m1.shape[0], m2.shape[1]))


class StateTransitionModel:
    def __init__(self):
        self.model = np.zeros(0, dtype=float)
        self.dt = 0.
        self.create_model()
        self.set_dt(0.)

    def __call__(self):
        return self.model

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def set_dt(self, dt):
        pass


class ObservationCovarianceModel:
    def __init__(self):
        self.model = np.zeros(0, dtype=float)
        self.create_model()

    def __call__(self):
        return self.model

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def set_covariances(self, errors):
        pass


class KalmanFilter:
    def __init__(self):
        self.state_model = np.zeros(0, dtype=float)
        self.state_covariance = np.zeros(0, dtype=float)
        self.state_transition_model = StateTransitionModel()
        self.observation_model = np.zeros(0, dtype=float)
        self.observation_covariance_model = ObservationCovarianceModel()
        self.process_noise_covariance = np.zeros(0, dtype=float)
        self.predicted_state = np.zeros(0, dtype=float)
        self.predicted_state_covariance = np.zeros(0, dtype=float)

    def valid(self):
        # Check state model and state transition model compatibility
        check = {'state transition model, state model': (self.state_transition_model(), self.state_model),
                 'state transition model, state covariance': (self.state_transition_model(), self.state_covariance)}

        errors = []
        for label, values in check.items():
            compatible, output = matcompat(*values)
            if not compatible:
                errors.append(label + ' incompatible')

        if len(errors) > 0:
            print('\n'.join(errors))
            return False

        self.predicted_state, self.predicted_state_covariance = self.predict(self.state_transition_model.dt)
        check = {'obs model, predicted state': (self.observation_model, self.predicted_state),
                 'obs model, predicted state covariance': (self.observation_model, self.predicted_state_covariance)}

        errors = []
        for label, values in check.items():
            compatible, output = matcompat(*values)
            if not compatible:
                errors.append(label + ' incompatible')

        if len(errors) > 0:
            print('\n'.join(errors))
            return False

        return True

    def predict(self, dt):
        self.state_transition_model.set_dt(dt)
        predictions = matmul(self.state_transition_model(), self.state_model)
        covariances = matmul(matmul(self.state_transition_model(), self.state_covariance),
                             np.transpose(self.state_transition_model())) + self.process_noise_covariance
        return predictions, covariances

    def update(self, observations, observation_errors, dt):
        for m in [observations, observation_errors]:
            m.shape = m.shape + (1,) if len(m.shape) < 2 else m.shape

        self.predicted_state, self.predicted_state_covariance = self.predict(dt)
        self.observation_covariance_model.set_covariances(observation_errors)

        residuals = observations - matmul(self.observation_model, self.predicted_state)
        residual_covariance = matmul(matmul(self.observation_model, self.predicted_state_covariance),
                                     np.transpose(self.observation_model)) + self.observation_covariance_model()

        kalman_gain = matmul(matmul(self.predicted_state_covariance, np.transpose(self.observation_model)),
                             np.linalg.inv(residual_covariance))
        self.state_model = self.predicted_state + matmul(kalman_gain, residuals)
        self.state_covariance = matmul(np.identity(len(self.state_model)) - matmul(kalman_gain, self.observation_model),
                                       self.predicted_state_covariance)

        post_fit_residuals = observations - matmul(self.observation_model, self.state_model)
        return post_fit_residuals
