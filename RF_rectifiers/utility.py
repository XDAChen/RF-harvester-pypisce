#!/usr/bin/env python3
"""
================================================================================
RF Rectifier Simulation Utilities
================================================================================
Shared utility functions for RF rectifier simulations.

This module provides:
    - ESR calculation from Q factor
    - ngspice transient simulation runner
    - Transient result plotting

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
import os
from pathlib import Path


# =============================================================================
# Component Modeling Functions
# =============================================================================

def calculate_esr_from_q(capacitance, q_factor, frequency):
    """
    Calculate ESR (Equivalent Series Resistance) from Q factor.
    
    Q = 1 / (2 * pi * f * C * ESR)
    Therefore: ESR = 1 / (2 * pi * f * C * Q)
    
    Args:
        capacitance: Capacitor value in Farads
        q_factor: Quality factor (dimensionless)
        frequency: Operating frequency in Hz
    
    Returns:
        ESR in Ohms
    """
    return 1 / (2 * np.pi * frequency * capacitance * q_factor)


# =============================================================================
# Simulation Functions
# =============================================================================

def run_transient_simulation(netlist, t_stop, n_cycles, t_step, num_signals=2):
    """
    Run transient simulation using ngspice subprocess.
    
    Args:
        netlist: SPICE netlist string
        t_stop: Simulation end time in seconds
        n_cycles: Number of RF cycles (for info display)
        t_step: Time step in seconds
        num_signals: Number of signals to parse (excluding time)
                    2 for halfwave (rf_in, vout)
                    4 for dickson (rf_in, node1, node2, vout)
    
    Returns:
        Dict with 'time' and signal arrays
    """
    print(f"\n[INFO] Running transient simulation...")
    print(f"       Duration: {t_stop*1e9:.2f} ns ({n_cycles} RF cycles)")
    print(f"       Time step: {t_step*1e12:.2f} ps")
    
    # Create temporary directory for simulation
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write netlist to file
        netlist_path = os.path.join(tmpdir, 'circuit.sp')
        output_path = os.path.join(tmpdir, 'output.txt')
        
        with open(netlist_path, 'w') as f:
            f.write(netlist)
        
        # Run ngspice
        print("[INFO] Running ngspice...")
        result = subprocess.run(
            ['ngspice', '-b', netlist_path],
            capture_output=True,
            text=True,
            cwd=tmpdir
        )
        
        if result.returncode != 0:
            print("[ERROR] ngspice failed:")
            print(result.stderr)
            print(result.stdout)
            raise RuntimeError("ngspice simulation failed")
        
        # Parse output file
        if not os.path.exists(output_path):
            print("[ERROR] Output file not created")
            print("ngspice stdout:", result.stdout)
            print("ngspice stderr:", result.stderr)
            raise RuntimeError("No simulation output")
        
        # Read and parse the data file
        # ngspice wrdata format: time repeated before each signal
        # For N signals: time time time sig1 time sig2 time sig3 ... time sigN
        # Total columns = N + (N+1) = 2N + 1 (but time repeats, so we have N+1 unique)
        
        # Expected columns: time + (N signals with time prefix each)
        # Example 2 signals: time time time v1 time v2 => 6 columns, v1=col3, v2=col5
        # Example 4 signals: time*5 v1 time v2 time v3 time v4 => 12 cols
        expected_cols = 2 + num_signals * 2  # First 2 time cols + (time, value) pairs
        
        data_lists = {f'signal_{i}': [] for i in range(num_signals)}
        data_lists['time'] = []
        
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= expected_cols:
                    try:
                        data_lists['time'].append(float(parts[0]))
                        # Signals are at indices: 3, 5, 7, 9, ... (every 2nd starting at 3)
                        for i in range(num_signals):
                            col_idx = 3 + i * 2
                            data_lists[f'signal_{i}'].append(float(parts[col_idx]))
                    except (ValueError, IndexError):
                        continue
        
        print(f"[INFO] Parsed {len(data_lists['time'])} data points")
        
        # Convert to numpy arrays
        result_dict = {'time': np.array(data_lists['time'])}
        for i in range(num_signals):
            result_dict[f'signal_{i}'] = np.array(data_lists[f'signal_{i}'])
        
        return result_dict


def run_halfwave_simulation(netlist, t_stop, n_cycles, t_step):
    """
    Run simulation for half-wave rectifier (2 signals: rf_in, vout).
    
    Returns:
        Dict with 'time', 'rf_in', 'vout'
    """
    data = run_transient_simulation(netlist, t_stop, n_cycles, t_step, num_signals=2)
    return {
        'time': data['time'],
        'rf_in': data['signal_0'],
        'vout': data['signal_1']
    }


def run_dickson_simulation(netlist, t_stop, n_cycles, t_step):
    """
    Run simulation for Dickson rectifier (4 signals: rf_in, node1, node2, vout).
    
    Returns:
        Dict with 'time', 'rf_in', 'node1', 'node2', 'vout'
    """
    print("       (Dickson needs more cycles to reach steady-state)")
    data = run_transient_simulation(netlist, t_stop, n_cycles, t_step, num_signals=4)
    return {
        'time': data['time'],
        'rf_in': data['signal_0'],
        'node1': data['signal_1'],
        'node2': data['signal_2'],
        'vout': data['signal_3']
    }


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_halfwave_results(data, f_rf, v_rf_amplitude, r_load, save_filename='halfwave_transient.png'):
    """
    Plot transient simulation results for half-wave rectifier.
    
    Args:
        data: Dict with 'time', 'rf_in', 'vout' arrays
        f_rf: RF frequency in Hz
        v_rf_amplitude: Input amplitude in Volts
        r_load: Load resistance in Ohms
        save_filename: Output plot filename
    """
    # Extract time and voltage arrays
    time = data['time'] * 1e9  # Convert to nanoseconds
    v_rf_in = data['rf_in']
    v_out = data['vout']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Half-Wave RF Rectifier Transient Analysis @ {f_rf/1e9:.2f} GHz', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Input RF Signal
    axes[0].plot(time, v_rf_in, 'b-', linewidth=0.8, label='RF Input')
    axes[0].set_ylabel('Voltage (V)', fontsize=11)
    axes[0].set_title('Input RF Signal (after source impedance)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 2: Output DC Voltage
    axes[1].plot(time, v_out, 'r-', linewidth=1.0, label='DC Output')
    axes[1].set_xlabel('Time (ns)', fontsize=11)
    axes[1].set_ylabel('Voltage (V)', fontsize=11)
    axes[1].set_title('Rectified Output Voltage', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Calculate steady-state metrics (use last 20% of simulation)
    ss_start = int(0.8 * len(v_out))
    v_dc_avg = np.mean(v_out[ss_start:])
    v_dc_ripple = np.max(v_out[ss_start:]) - np.min(v_out[ss_start:])
    
    axes[1].axhline(y=v_dc_avg, color='g', linestyle='--', linewidth=1.5, 
                    label=f'Avg DC = {v_dc_avg*1000:.2f} mV')
    axes[1].legend(loc='upper right')
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"SIMULATION RESULTS SUMMARY")
    print(f"="*60)
    print(f"  Steady-state DC output:  {v_dc_avg*1000:.2f} mV")
    print(f"  Output ripple (pk-pk):   {v_dc_ripple*1000:.2f} mV")
    print(f"  Load current:            {v_dc_avg/r_load*1e6:.2f} µA")
    print(f"  Output power:            {(v_dc_avg**2/r_load)*1e6:.2f} µW")
    print(f"="*60)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Plot saved to '{save_filename}'")
    plt.show()


def plot_dickson_results(data, n_stages, f_rf, v_rf_amplitude, r_load, 
                         save_filename='dickson_transient.png'):
    """
    Plot transient simulation results for Dickson charge pump.
    
    Args:
        data: Dict with 'time', 'rf_in', 'node1', 'node2', 'vout' arrays
        n_stages: Number of Dickson stages
        f_rf: RF frequency in Hz
        v_rf_amplitude: Input amplitude in Volts
        r_load: Load resistance in Ohms
        save_filename: Output plot filename
    """
    # Extract time and voltage arrays
    time = data['time'] * 1e9  # Convert to nanoseconds
    v_rf_in = data['rf_in']
    v_node1 = data['node1']
    v_node2 = data['node2']
    v_out = data['vout']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'{n_stages}-Stage Dickson Charge Pump Transient Analysis @ {f_rf/1e9:.2f} GHz', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Input RF Signal
    axes[0].plot(time, v_rf_in, 'b-', linewidth=0.8, label='RF Input')
    axes[0].set_ylabel('Voltage (V)', fontsize=11)
    axes[0].set_title('Input RF Signal (after source impedance)', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 2: Intermediate nodes
    axes[1].plot(time, v_node1, 'orange', linewidth=0.8, label='Node 1 (after clamp)')
    axes[1].plot(time, v_node2, 'purple', linewidth=0.8, label='Node 2 (stage 2 input)')
    axes[1].set_ylabel('Voltage (V)', fontsize=11)
    axes[1].set_title('Intermediate Stage Voltages', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    
    # Plot 3: Output DC Voltage
    axes[2].plot(time, v_out, 'r-', linewidth=1.0, label='DC Output')
    axes[2].set_xlabel('Time (ns)', fontsize=11)
    axes[2].set_ylabel('Voltage (V)', fontsize=11)
    axes[2].set_title('Rectified Output Voltage (Dickson Output)', fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    # Calculate steady-state metrics (use last 20% of simulation)
    ss_start = int(0.8 * len(v_out))
    v_dc_avg = np.mean(v_out[ss_start:])
    v_dc_ripple = np.max(v_out[ss_start:]) - np.min(v_out[ss_start:])
    
    # Theoretical max for N-stage Dickson: N * Vpk (minus N diode drops)
    v_theoretical = n_stages * v_rf_amplitude
    
    axes[2].axhline(y=v_dc_avg, color='g', linestyle='--', linewidth=1.5)
    axes[2].axhline(y=v_theoretical, color='gray', linestyle=':', linewidth=1.5)
    axes[2].legend([
        'DC Output',
        f'Avg DC = {v_dc_avg*1000:.1f} mV',
        f'Theoretical max = {v_theoretical*1000:.1f} mV'
    ], loc='upper right')
    
    # Print summary
    print(f"\n" + "="*60)
    print(f"SIMULATION RESULTS SUMMARY ({n_stages}-Stage Dickson)")
    print(f"="*60)
    print(f"  RF Input amplitude:      {v_rf_amplitude*1000:.1f} mV pk")
    print(f"  Theoretical max output:  {v_theoretical*1000:.1f} mV (N×Vpk)")
    print(f"  Actual DC output:        {v_dc_avg*1000:.2f} mV")
    print(f"  Efficiency:              {(v_dc_avg/v_theoretical)*100:.1f}% of theoretical")
    print(f"  Output ripple (pk-pk):   {v_dc_ripple*1000:.2f} mV")
    print(f"  Load current:            {v_dc_avg/r_load*1e6:.2f} µA")
    print(f"  Output power:            {(v_dc_avg**2/r_load)*1e6:.2f} µW")
    print(f"="*60)
    
    # Voltage multiplication factor
    if v_rf_amplitude > 0:
        mult_factor = v_dc_avg / v_rf_amplitude
        print(f"  Voltage multiplication:  {mult_factor:.2f}x")
        print(f"="*60)
    
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"\n[INFO] Plot saved to '{save_filename}'")
    plt.show()
