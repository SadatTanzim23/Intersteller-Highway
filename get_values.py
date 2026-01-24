# citation
# https://github.com/poliastro/poliastro/blob/main/src/poliastro/bodies.py

import numpy as np
import pandas as pd
from datetime import datetime

class SolarSystemEngine:
    def __init__(self):
        # J2000: a(AU), e, i(deg), L(deg), Argument of Periapsis(deg), Longitude of Ascending Node(deg)
        self.elements = {
            "Mercury": [0.38709893, 0.20563069, 7.00487, 252.25084, 77.45645, 48.33167],
            "Venus":   [0.72333199, 0.00677323, 3.39471, 181.97973, 131.53298, 76.68069],
            "Earth":   [1.00000011, 0.01671022, 0.00005, 100.46435, 102.94719, 0.0],
            "Mars":    [1.52366231, 0.09341233, 1.85061, -4.55343, 336.04084, 49.57854],
            "Jupiter": [5.20336301, 0.04839266, 1.30530, 34.40438, 14.75385, 100.55615],
            "Saturn":  [9.53707032, 0.05415060, 2.48446, 49.94432, 92.43194, 113.71504],
            "Uranus":  [19.19126393, 0.04716771, 0.76986, 313.23218, 170.96424, 74.22988],
            "Neptune": [30.06896348, 0.00858587, 1.76917, 304.88003, 44.97135, 131.72169],
            "Pluto":   [39.48168677, 0.24880766, 17.14175, 238.92881, 224.06676, 110.30347],
            "Ceres":   [2.767, 0.0758, 10.59, 153.23, 73.06, 80.30]
        }
        self.AU = 149597870.7

    def get_position(self, body_name, date_str):
        if body_name not in self.elements:
            return None
        
        d0 = datetime(2000, 1, 1, 12, 0)
        target_date = datetime.strptime(date_str, "%d%m%y") # Your CLI format
        diff = (target_date - d0).days / 36525.0
        
        a, e, i, L, lp, ln = self.elements[body_name]
        i, L, lp, ln = map(np.radians, [i, L, lp, ln])
        
        # Argument of perihelion (w) and Mean Anomaly (M)
        w = lp - ln
        M = L - lp
        
        # Solve Kepler's Equation for Eccentric Anomaly (E)
        E = M
        for _ in range(10): # Iterative solver
            E = E - (E - e*np.sin(E) - M) / (1 - e*np.cos(E))
            
        # Coordinates in orbital plane
        x_orb = a * (np.cos(E) - e)
        y_orb = a * np.sqrt(1 - e**2) * np.sin(E)
        
        # Rotate to Heliocentric Ecliptic coordinates
        # Source: [Source 183 - Orbital Mechanics for Engineering Students]
        x = (np.cos(ln)*np.cos(w) - np.sin(ln)*np.sin(w)*np.cos(i)) * x_orb + \
            (-np.cos(ln)*np.sin(w) - np.sin(ln)*np.cos(w)*np.cos(i)) * y_orb
        y = (np.sin(ln)*np.cos(w) + np.cos(ln)*np.sin(w)*np.cos(i)) * x_orb + \
            (-np.sin(ln)*np.sin(w) + np.cos(ln)*np.cos(w)*np.cos(i)) * y_orb
        z = (np.sin(w)*np.sin(i)) * x_orb + (np.cos(w)*np.sin(i)) * y_orb
        
        return x * self.AU, y * self.AU, z * self.AU

def main():
    engine = SolarSystemEngine()
    # Dummy test date for today
    test_date = "230126" 
    
    results = []
    for body in engine.elements.keys():
        pos = engine.get_position(body, test_date)
        results.append({
            "Body Name": body,
            "X (km)": f"{pos[0]:,.0f}",
            "Y (km)": f"{pos[1]:,.0f}",
            "Z (km)": f"{pos[2]:,.0f}"
        })
    
    df = pd.DataFrame(results)
    print("--- PLANETARY POSITIONS (HELIOCENTRIC) ---")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()