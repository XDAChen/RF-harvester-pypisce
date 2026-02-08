#!/usr/bin/env python3
"""
================================================================================
Dickson Charge Pump RF Rectifier Simulation at 2.45 GHz
================================================================================
A 2-stage Dickson charge pump (voltage multiplier) for RF energy harvesting.

Circuit Topology (2-stage Dickson):

    RF_in o---C1---+---D1--->---+---D2--->---+---o Vout
                   |            |            |
                  D0           C2           Cout
                   |            |            |
    GND   o--------+-----------+------------+---o GND

Theoretical output: ~N * Vpk (minus diode drops)

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

from pathlib import Path
from utility import (
    calculate_esr_from_q,
    run_dickson_simulation,
    plot_dickson_results
)

# =============================================================================
# Simulation Parameters - EASY TO MODIFY
# =============================================================================

# Operating frequency
F_RF = 2.45e9           # RF frequency in Hz (2.45 GHz ISM band)
T_RF = 1 / F_RF         # RF period (~408 ps)

# Input signal
V_RF_AMPLITUDE = 0.5    # RF input amplitude in Volts (adjust for your application)
R_SOURCE = 50           # Source impedance in Ohms

# Number of Dickson stages
N_STAGES = 2            # 2-stage Dickson (each stage adds ~Vpk to output)

# Component values
C_STAGE = 10e-12        # Stage coupling capacitors (10 pF each)
C_OUT = 100e-12         # Output smoothing capacitor (100 pF)
R_LOAD = 10e3           # Load resistance (10 kOhm)

# Capacitor Q factor modeling (Q = 1/(2*pi*f*C*ESR))
CAP_Q = 100             # Q factor for all capacitors

# Simulation timing
N_CYCLES = 100          # Number of RF cycles (Dickson needs more time to charge)
T_STEP = T_RF / 40      # Time step (40 points per RF cycle)
T_STOP = N_CYCLES * T_RF

# Diode model configuration
DIODE_MODEL_FILE = "diode_models.lib"
DIODE_MODEL_NAME = "SMS7630"  # Options: SMS7630, HSMS2850, DEFAULT_SCHOTTKY


# =============================================================================
# Netlist Builder
# =============================================================================

def build_dickson_netlist():
    """
    Build the N-stage Dickson charge pump rectifier SPICE netlist.
    
    Returns:
        Tuple of (netlist string, ESR values dict)
    """
    # Calculate ESR values
    esr_stage = calculate_esr_from_q(C_STAGE, CAP_Q, F_RF)
    esr_out = calculate_esr_from_q(C_OUT, CAP_Q, F_RF)
    
    print(f"[INFO] Stage capacitor ESR = {esr_stage*1000:.3f} mΩ (Q={CAP_Q})")
    print(f"[INFO] Output capacitor ESR = {esr_out*1000:.3f} mΩ (Q={CAP_Q})")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / DIODE_MODEL_FILE
    
    netlist = f"""{N_STAGES}-Stage Dickson Charge Pump RF Rectifier @ {F_RF/1e9:.2f} GHz
* ============================================
* Circuit: Dickson voltage multiplier
* Frequency: {F_RF/1e9:.2f} GHz
* Stages: {N_STAGES} (theoretical output: {N_STAGES}x Vpk)
* ============================================

* Include diode models
.include {model_path}

* RF Source with source impedance
Vrf rf_source 0 SIN(0 {V_RF_AMPLITUDE} {F_RF})
Rsource rf_source rf_in {R_SOURCE}

* === Stage 1 ===
* Coupling capacitor C1 (with ESR)
C1 rf_in node1_c {C_STAGE}
Resr1 node1_c node1 {esr_stage}

* Clamp diode D0 (clamps negative peak to ground)
D0 0 node1 {DIODE_MODEL_NAME}

* Series diode D1 (rectifies positive half)
D1 node1 node2 {DIODE_MODEL_NAME}

* === Stage 2 ===
* Hold capacitor C2 (with ESR) - stores charge between stages
C2 node2 node2_gnd {C_STAGE}
Resr2 node2_gnd 0 {esr_stage}

* Series diode D2 (adds second stage)
D2 node2 vout {DIODE_MODEL_NAME}

* === Output ===
* Output smoothing capacitor (with ESR)
Cout vout vout_esr {C_OUT}
Resr_out vout_esr 0 {esr_out}

* Load resistor
Rload vout 0 {R_LOAD}

* Simulation control
.control
set filetype=ascii
set wr_vecnames
option noacct
tran {T_STEP} {T_STOP} uic
wrdata output.txt time v(rf_in) v(node1) v(node2) v(vout)
quit
.endc

.end
"""
    return netlist, {'esr_stage': esr_stage, 'esr_out': esr_out}


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print(f"DICKSON CHARGE PUMP RF RECTIFIER SIMULATION ({N_STAGES} stages)")
    print(f"Frequency: {F_RF/1e9:.2f} GHz | Input: {V_RF_AMPLITUDE*1000:.0f} mV pk")
    print(f"Expected output: ~{N_STAGES * V_RF_AMPLITUDE*1000:.0f} mV (N × Vpk)")
    print("="*60)
    
    # Build netlist
    netlist, esr_values = build_dickson_netlist()
    
    # Print netlist for verification
    print("\n[INFO] Generated SPICE Netlist:")
    print("-" * 40)
    print(netlist)
    print("-" * 40)
    
    # Run simulation
    data = run_dickson_simulation(netlist, T_STOP, N_CYCLES, T_STEP)
    
    # Plot results
    plot_dickson_results(data, N_STAGES, F_RF, V_RF_AMPLITUDE, R_LOAD)
