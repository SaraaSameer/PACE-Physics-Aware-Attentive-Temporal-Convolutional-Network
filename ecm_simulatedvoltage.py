import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch
from scipy.optimize import curve_fit

class Thevenien_1RC:
    def first_order_ecm_model(self, t, R0, R1, C1, Uocv):
        tau = R1 * C1
        return Uocv - R0 * I - R1 * I * (1 - np.exp(-t / tau))

    def extract_1rc_features(self, time, voltage, current, Uocv=None):
        global I
        I = np.mean(current)  
        initial_guess = [0.01, 0.01, 1000]  # [R0, R1, C1]
        if Uocv is None:
            Uocv = voltage[0] + I * 0.05  
            popt, _ = curve_fit(
            lambda t, R0, R1, C1: self.first_order_ecm_model(t, R0, R1, C1, Uocv),
            time, voltage,
            p0=initial_guess,
            bounds=([0, 0, 0], [1, 1, 10000]),
            maxfev=10000  
        )
        R0, R1, C1 = popt
        return {"R0": R0, "R1": R1, "C1": C1, "Uocv": Uocv}


