# Variables

# Planetary:
    # r1, r2: Absolute position vectors of planets at launch/arrival (3-axis)
    # pV1, pV2: Absolute velocity vectors of planets at launch/arrival (3-axis)
    # StdGravSun: Standard gravitational parameter of the Sun or Planet.

# Ship:
    # WMass: Mass with fuel
    # DMass: Mass without fuel
    # SI: Specific Impulse (velocity in km/s).
    # PLMass : Mass of payload

# Ship Trajectory:
    # sV1, sV2: Absolute ship velocity at launch and arrival
    # iV1, iV2: Injection and insertion velocities (relative to planet)
    # delV: Total change in velocity (measure of fuel expenditure)
    # PLMass : Mass of payload

# Orbit
    # t: Orbital period = 43200 seconds (specifically 12 hours for refueling).
    # Rfuel: Radius of the circular parking/refueling orbit.

import numpy as np
from datetime import datetime
from poliastro.iod import izzo
from get_values import SolarSystemEngine

# PLANETARY & SOLAR INITIALIZATION

StdGravSun = 1.32712440018e11 
StdGravPlanets = {
    "Mercury": 22032, "Venus": 324859, "Earth": 398600, 
    "Mars": 42828, "Jupiter": 126686534, "Saturn": 37931187, 
    "Uranus": 5793940, "Neptune": 6836529, "Pluto": 871, "Ceres": 63
}

r1 = np.array([0.0, 0.0, 0.0]) 
r2 = np.array([0.0, 0.0, 0.0])

pV1 = np.array([0.0, 0.0, 0.0])
pV2 = np.array([0.0, 0.0, 0.0])


# SHIP INITIALIZATION
import numpy as np

ships = {
    "Chevrolet Super Sonic": {
        "DMass": 5000.0, "Fuel_Cap": 20000.0, "SI": 4.2, 
        "Fuel_Type": "Leaded gasoline", "Max_PLMass": 1000.0, "PLMass": 0.0
    },
    "The Planet Hopper": {
        "DMass": 10000.0, "Fuel_Cap": 100000.0, "SI": 6.7, 
        "Fuel_Type": "Compressed air", "Max_PLMass": 4000.0, "PLMass": 0.0
    },
    "Moonivan": {
        "DMass": 25000.0, "Fuel_Cap": 400000.0, "SI": 9.1, 
        "Fuel_Type": "Biofuel", "Max_PLMass": 10000.0, "PLMass": 0.0
    },
    "Blue Origin Delivery Ship": {
        "DMass": 69000.0, "Fuel_Cap": 800000.0, "SI": 15.2, 
        "Fuel_Type": "Whale oil", "Max_PLMass": 50000.0, "PLMass": 0.0
    },
    "Yamaha Space Cycle": {
        "DMass": 1000.0, "Fuel_Cap": 2500.0, "SI": 100.0, 
        "Fuel_Type": "Antimatter", "Max_PLMass": 100.0, "PLMass": 0.0
    },
    "Ford F-1500": {
        "DMass": 10000.0, "Fuel_Cap": 100000.0, "SI": 18.67, 
        "Fuel_Type": "Space Diesel", "Max_PLMass": 8000.0, "PLMass": 0.0
    },
    "Beheamoth": {
        "DMass": 100000.0, "Fuel_Cap": 1500000.0, "SI": 11.1, 
        "Fuel_Type": "Nuclear propulsion", "Max_PLMass": 100000.0, "PLMass": 0.0
    }
}

for name, data in ships.items():
    # WMass = Dry Mass + Payload + Fuel Capacity 
    data["WMass"] = data["DMass"] + data["PLMass"] + data["Fuel_Cap"]


# SHIP TRAJECTORY INITIALIZATION
sV1 = np.array([0.0, 0.0, 0.0])
sV2 = np.array([0.0, 0.0, 0.0])

iV1 = np.array([0.0, 0.0, 0.0])
iV2 = np.array([0.0, 0.0, 0.0])

delV = 0.0

# ORBIT
T = 43200 #12 hours
Rfuel_data = {}

for planet, mu in StdGravPlanets.items():
    radius = (mu * (T / (2 * np.array(np.pi)))**2)**(1/3)
    velocity = np.sqrt(mu / radius)
    Rfuel_data[planet] = {
        "Rfuel": radius,
        "v_park": velocity
    }

# CITATION: https://github.com/poliastro/poliastro/blob/main/src/poliastro/iod/izzo.py
# Used to understand the izzo.lambert() function
def get_highway_velocities(r1, r2, TOF_days):
    mu_sun = 1.32712440018e11 
    seconds = TOF_days * 24 * 3600
    (sV1, sV2), = izzo.lambert(mu_sun, r1, r2, seconds)
    return sV1, sV2

# iV1: Injection velocity (Relative launch velocity)
# iV2: Insertion velocity (Relative arrival velocity)
def get_relative_velocities(sV1, pV1, sV2, pV2):
    iV1 = sV1 - pV1
    iV2 = sV2 - pV2
    return iV1, iV2

def calculate_total_delV(iV1, iV2, start_planet, end_planet):
    start_data = Rfuel_data[start_planet]
    end_data = Rfuel_data[end_planet]
    
    mu_start = StdGravPlanets[start_planet]
    mu_end = StdGravPlanets[end_planet]

    # launch force exertion
 
    v_inf_launch = np.linalg.norm(iV1)
    v_inf_arrival = np.linalg.norm(iV2)
    
    v_peri_launch = np.sqrt(v_inf_launch**2 + 2 * mu_start / start_data["Rfuel"])
    dv_launch = abs(v_peri_launch - start_data["v_park"])
    
    # arrival force exertion
    v_peri_arrival = np.sqrt(v_inf_arrival**2 + 2 * mu_end / end_data["Rfuel"])
    dv_arrival = abs(v_peri_arrival - end_data["v_park"])
    
    return dv_launch + dv_arrival

def get_required_fuel(delV_total, ship_name, current_payload):
    ship = ships[ship_name]
    mf = ship["DMass"] + current_payload
    
    m0 = mf * np.exp(delV_total / ship["SI"])
    
    fuel_needed = m0 - mf
    
    possible = fuel_needed <= ship["Fuel_Cap"]
    return fuel_needed, possible

def find_best_flight(start_p, end_p, ship_name, payload, date_str):
    engine = SolarSystemEngine()
    
    # 1. Get Positions & Velocities (Standardizing dates)
    # Start: Launch Date | End: Approx 200 days later for initial check
    r1, v_p1 = engine.get_position(start_p, date_str) 
    r2, v_p2 = engine.get_position(end_p, "010826") # Example arrival date
    
    # 2. Highway Velocity (sV1, sV2) [cite: 14, 15]
    # Assuming a 200-day flight for this rapid calculation
    sV1, sV2 = get_highway_velocities(r1, r2, 200)
    
    # 3. Relative Velocities (iV1, iV2) [cite: 21, 23]
    iV1, iV2 = get_relative_velocities(sV1, v_p1, sV2, v_p2)
    
    # 4. Total deltaV (Fuel Expenditure) [cite: 153, 319]
    total_dv = calculate_total_delV(iV1, iV2, start_p, end_p)
    
    # 5. Check Fuel Capacity [cite: 310, 324]
    fuel, is_possible = get_required_fuel(total_dv, ship_name, payload)
    
    return {
        "Flight impossible": str(not is_possible).lower(),
        "Fuel": fuel,
        "deltaV": total_dv
    }