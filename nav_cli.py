import sys
import json
import argparse
import math
import numpy as np
from datetime import datetime, timedelta


MU_SUN = 1.32712440018e11 
AU = 149597870.7

MU_PLANETS = {
    "mercury": 22032, "venus": 324859, "earth": 398600, 
    "mars": 42828, "jupiter": 126686534, "saturn": 37931187, 
    "uranus": 5793940, "neptune": 6836529, "pluto": 871, "ceres": 63
}
PLANET_ELEMENTS = {
    "mercury": [0.38709893, 0.20563069, 7.00487, 252.25084, 77.45645, 48.33167],
    "venus":   [0.72333199, 0.00677323, 3.39471, 181.97973, 131.53298, 76.68069],
    "earth":   [1.00000011, 0.01671022, 0.00005, 100.46435, 102.94719, 0.0],
    "mars":    [1.52366231, 0.09341233, 1.85061, -4.55343, 336.04084, 49.57854],
    "jupiter": [5.20336301, 0.04839266, 1.30530, 34.40438, 14.75385, 100.55615],
    "saturn":  [9.53707032, 0.05415060, 2.48446, 49.94432, 92.43194, 113.71504],
    "uranus":  [19.19126393, 0.04716771, 0.76986, 313.23218, 170.96424, 74.22988],
    "neptune": [30.06896348, 0.00858587, 1.76917, 304.88003, 44.97135, 131.72169],
    "pluto":   [39.48168677, 0.24880766, 17.14175, 238.92881, 224.06676, 110.30347],
    "ceres":   [2.767, 0.0758, 10.59, 153.23, 73.06, 80.30]
}

SHIPS = {
    "chevrolet_super_sonic":     {"DMass": 5000.0, "Fuel_Cap": 20000.0, "SI": 4.2, "Max_PLMass": 1000.0},
    "the_planet_hopper":         {"DMass": 10000.0, "Fuel_Cap": 100000.0, "SI": 6.7, "Max_PLMass": 4000.0},
    "moonivan":                  {"DMass": 25000.0, "Fuel_Cap": 400000.0, "SI": 9.1, "Max_PLMass": 10000.0},
    "blue_origin_delivery_ship": {"DMass": 69000.0, "Fuel_Cap": 800000.0, "SI": 15.2, "Max_PLMass": 50000.0},
    "yamaha_space_cycle":        {"DMass": 1000.0, "Fuel_Cap": 2500.0, "SI": 100.0, "Max_PLMass": 100.0},
    "ford_f-1500":               {"DMass": 10000.0, "Fuel_Cap": 100000.0, "SI": 18.67, "Max_PLMass": 8000.0},
    "beheamoth":                 {"DMass": 100000.0, "Fuel_Cap": 1500000.0, "SI": 11.1, "Max_PLMass": 100000.0}
}

# ==========================================
# 2. PHYSICS ENGINE
# ==========================================

class SolarSystemEngine:
    def __init__(self):
        self.elements = PLANET_ELEMENTS
        self.AU = AU

    def _kepler_equation(self, M, e):
        E = M
        for _ in range(15):
            E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        return E

    def get_state_vector(self, body_name, date_obj):
        """Returns r(km), v(km/s) using orbital elements and numerical diff"""
        if body_name not in self.elements:
            raise ValueError(f"Unknown planet: {body_name}")

        def get_pos_at_t(dt_obj):
            d0 = datetime(2000, 1, 1, 12, 0)
            diff_days = (dt_obj - d0).total_seconds() / (24 * 3600)
            a_au, e, i_deg, L_deg, lp_deg, ln_deg = self.elements[body_name]
            i, L, lp, ln = map(np.radians, [i_deg, L_deg, lp_deg, ln_deg])
            w = lp - ln
            M = L - lp
            n = 0.9856076686 / (a_au ** 1.5)
            M_curr = (M + np.radians(n * diff_days)) % (2 * np.pi)
            E = self._kepler_equation(M_curr, e)

            x_orb = a_au * (np.cos(E) - e)
            y_orb = a_au * np.sqrt(1 - e**2) * np.sin(E)

            x = (np.cos(ln)*np.cos(w) - np.sin(ln)*np.sin(w)*np.cos(i)) * x_orb + \
                (-np.cos(ln)*np.sin(w) - np.sin(ln)*np.cos(w)*np.cos(i)) * y_orb
            y = (np.sin(ln)*np.cos(w) + np.cos(ln)*np.sin(w)*np.cos(i)) * x_orb + \
                (-np.sin(ln)*np.