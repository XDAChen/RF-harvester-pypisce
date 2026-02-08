#!/usr/bin/env python3
"""
================================================================================
Half-Wave RF Rectifier Simulation at 2.45 GHz
================================================================================
A simple single-diode half-wave rectifier for RF energy harvesting.

Circuit Topology:
                     D1
    RF_in o----+---->|----+----o Vout
               |          |
              C_in       C_out
               |          |
    GND   o----+----------+----o GND

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

from pathlib import Path
from utility import (
    calculate_esr_from_q,
    run_halfwave_simulation,
    plot_halfwave_results
)

# =============================================================================
# Simulation Parameters - EASY TO MODIFY
# =============================================================================

# Operating frequency
F_RF = 2.45e9           # RF frequency in Hz (2.45 GHz ISM band)
T_RF = 1 / F_RF         # RF period

# Input signal
V_RF_AMPLITUDE = 0.5    # RF input amplitude in Volts (adjust for your application)
R_SOURCE = 50           # Source impedance in Ohms

# Component values
C_IN = 10e-12           # Input coupling capacitor (10 pF)
C_OUT = 100e-12         # Output smoothing capacitor (100 pF)
R_LOAD = 10e3           # Load resistance (10 kOhm)

# Capacitor Q factor modeling (Q = 1/(2*pi*f*C*ESR))
CAP_Q_IN = 100          # Q factor for input capacitor
CAP_Q_OUT = 100         # Q factor for output capacitor

# Simulation timing
N_CYCLES = 50           # Number of RF cycles to simulate
T_STEP = T_RF / 40      # Time step (40 points per RF cycle)
T_STOP = N_CYCLES * T_RF

# Diode model configuration
DIODE_MODEL_FILE = "diode_models.lib"
DIODE_MODEL_NAME = "SMS7630"  # Options: SMS7630, HSMS2850, DEFAULT_SCHOTTKY


# =============================================================================
# Netlist Builder
# =============================================================================

def build_halfwave_netlist():
    """
    Build the half-wave rectifier SPICE netlist.
    
    Returns:
        Tuple of (netlist string, ESR values dict)
    """
    # Calculate ESR values
    esr_in = calculate_esr_from_q(C_IN, CAP_Q_IN, F_RF)
    esr_out = calculate_esr_from_q(C_OUT, CAP_Q_OUT, F_RF)
    
    print(f"[INFO] C_in ESR = {esr_in*1000:.3f} mΩ (Q={CAP_Q_IN} @ {F_RF/1e9:.2f} GHz)")
    print(f"[INFO] C_out ESR = {esr_out*1000:.3f} mΩ (Q={CAP_Q_OUT} @ {F_RF/1e9:.2f} GHz)")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / DIODE_MODEL_FILE
    
    netlist = f"""Half-Wave RF Rectifier @ {F_RF/1e9:.2f} GHz
* ============================================
* Circuit: Simple half-wave Schottky rectifier
* Frequency: {F_RF/1e9:.2f} GHz
* ============================================

* Include diode models
.include {model_path}

* RF Source with source impedance
Vrf rf_source 0 SIN(0 {V_RF_AMPLITUDE} {F_RF})
Rsource rf_source rf_in {R_SOURCE}

* Input coupling capacitor with ESR
Cin rf_in node_c_in {C_IN}
Resr_in node_c_in diode_anode {esr_in}

* Schottky diode
D1 diode_anode vout {DIODE_MODEL_NAME}

* Output smoothing capacitor with ESR
Cout vout node_c_out {C_OUT}
Resr_out node_c_out 0 {esr_out}

* Load resistor
Rload vout 0 {R_LOAD}

* Simulation control
.control
set filetype=ascii
set wr_vecnames
option noacct
tran {T_STEP} {T_STOP} uic
wrdata output.txt time v(rf_in) v(vout)
quit
.endc

.end
"""
    return netlist, {'esr_in': esr_in, 'esr_out': esr_out}


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("HALF-WAVE RF RECTIFIER SIMULATION")
    print(f"Frequency: {F_RF/1e9:.2f} GHz | Input: {V_RF_AMPLITUDE*1000:.0f} mV pk")
    print("="*60)
    
    # Build netlist
    netlist, esr_values = build_halfwave_netlist()
    
    # Print netlist for verification
    print("\n[INFO] Generated SPICE Netlist:")
    print("-" * 40)
    print(netlist)
    print("-" * 40)
    
    # Run simulation
    data = run_halfwave_simulation(netlist, T_STOP, N_CYCLES, T_STEP)
    
    # Plot results
    plot_halfwave_results(data, F_RF, V_RF_AMPLITUDE, R_LOAD)
