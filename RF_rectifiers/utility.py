#!/usr/bin/env python3
"""
================================================================================
RF Rectifier Simulation Utilities (PySpice-based)
================================================================================
Shared utility functions for RF rectifier simulations using PySpice.

Provides:
    - PySpice simulator wrapper
    - Harmonic analysis (FFT-based)
    - Frequency sweep analysis
    - Sensitivity / Monte Carlo analysis

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# =============================================================================
# Publication Plot Style
# =============================================================================

PUB_STYLE = {
    'font.family': 'serif',
    'font.size': 24,
    'axes.labelsize': 28,
    'axes.labelweight': 'bold',
    'axes.titlesize': 28,
    'axes.linewidth': 2,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'legend.fontsize': 20,
    'legend.frameon': True,
    'legend.edgecolor': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'grid.linewidth': 1,
}

# Line styles for black & white plots
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
MARKER_STYLES = ['o', 's', '^', 'D', 'v', 'p']
HATCH_STYLES = ['', '///', '\\\\', 'xxx', '...', '+++']

def apply_pub_style():
    """Apply publication style to matplotlib."""
    plt.rcParams.update(PUB_STYLE)

def get_save_path(filename):
    """Get save path in temp_image folder."""
    script_dir = Path(__file__).parent.absolute()
    img_dir = script_dir / 'temp_image'
    img_dir.mkdir(exist_ok=True)
    return img_dir / filename


# PySpice imports
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Spice.Parser import SpiceParser
from PySpice.Probe.Plot import plot
from PySpice.Unit import *


# =============================================================================
# PySpice Simulator Wrapper
# =============================================================================

class RFSimulator:
    """
    PySpice-based simulator for RF circuits.
    Wraps ngspice shared library for efficient repeated simulations.
    """
    
    def __init__(self):
        self._ngspice = None
    
    def _get_simulator(self):
        """Lazy initialization of ngspice shared instance."""
        if self._ngspice is None:
            self._ngspice = NgSpiceShared.new_instance()
        return self._ngspice
    
    def run_transient(self, circuit, t_step, t_stop, nodes):
        """
        Run transient analysis using PySpice.
        
        Args:
            circuit: PySpice Circuit object
            t_step: Time step in seconds
            t_stop: Stop time in seconds
            nodes: List of node names to capture
        
        Returns:
            Dict with 'time' and node voltage arrays
        """
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=t_step, end_time=t_stop)
        
        result = {'time': np.array(analysis.time)}
        for node in nodes:
            result[node] = np.array(analysis[node])
        
        return result
    
    def run_ac(self, circuit, start_freq, stop_freq, n_points, nodes, variation='dec'):
        """
        Run AC analysis using PySpice.
        
        Args:
            circuit: PySpice Circuit object
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            n_points: Number of points (per decade if variation='dec')
            nodes: List of node names to capture
            variation: 'dec' (decade), 'oct' (octave), or 'lin' (linear)
        
        Returns:
            Dict with 'frequency' and node voltage arrays (complex)
        """
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=start_freq, stop_frequency=stop_freq,
                                number_of_points=n_points, variation=variation)
        
        result = {'frequency': np.array(analysis.frequency)}
        for node in nodes:
            result[node] = np.array(analysis[node])
        
        return result
    
    def run_from_netlist(self, netlist_str, t_step, t_stop, nodes):
        """
        Run transient from raw SPICE netlist string.
        
        Args:
            netlist_str: SPICE netlist as string
            t_step: Time step
            t_stop: Stop time
            nodes: List of node names
        
        Returns:
            Dict with simulation results
        """
        parser = SpiceParser(source=netlist_str)
        circuit = parser.build_circuit()
        return self.run_transient(circuit, t_step, t_stop, nodes)


# Global simulator instance
_simulator = RFSimulator()


def get_simulator():
    """Get the global simulator instance."""
    return _simulator


# =============================================================================
# Component Utilities
# =============================================================================

def calculate_esr_from_q(capacitance, q_factor, frequency):
    """
    Calculate ESR from Q factor.
    
    ESR = 1 / (2 * pi * f * C * Q)
    
    Args:
        capacitance: Capacitance in Farads
        q_factor: Quality factor
        frequency: Frequency in Hz
    
    Returns:
        ESR in Ohms
    """
    return 1.0 / (2 * np.pi * frequency * capacitance * q_factor)


# =============================================================================
# Harmonic Analysis
# =============================================================================

def harmonic_analysis(time, signal, fundamental_freq, n_harmonics=10):
    """
    FFT-based harmonic analysis.
    
    Args:
        time: Time array (seconds)
        signal: Voltage/current array
        fundamental_freq: Fundamental frequency (Hz)
        n_harmonics: Number of harmonics to extract
    
    Returns:
        Dict with dc, harmonics list, thd, spectrum data
    """
    # Use steady-state (last 50%)
    n = len(signal)
    ss_idx = n // 2
    sig = signal[ss_idx:]
    t = time[ss_idx:]
    
    dt = np.mean(np.diff(t))
    n_fft = len(sig)
    
    # FFT
    fft = np.fft.fft(sig)
    freqs = np.fft.fftfreq(n_fft, dt)
    mag = np.abs(fft) / n_fft
    mag[1:] *= 2  # Single-sided correction
    
    # DC component
    dc = mag[0]
    
    # Extract harmonics
    harmonics = []
    for h in range(1, n_harmonics + 1):
        target = h * fundamental_freq
        idx = np.argmin(np.abs(freqs[:n_fft//2] - target))
        harmonics.append({
            'n': h,
            'freq': freqs[idx],
            'mag': mag[idx],
            'phase_deg': np.angle(fft[idx], deg=True)
        })
    
    # THD calculation
    fund = harmonics[0]['mag'] if harmonics else 1e-12
    thd = np.sqrt(sum(h['mag']**2 for h in harmonics[1:])) / fund * 100 if fund > 0 else 0
    
    return {
        'dc': dc,
        'fundamental': harmonics[0] if harmonics else None,
        'harmonics': harmonics,
        'thd': thd,
        'freqs': freqs[:n_fft//2],
        'spectrum': mag[:n_fft//2]
    }


def plot_harmonics(result, save_prefix='harmonics'):
    """
    Plot harmonic analysis results as individual plots (publication style).
    
    Args:
        result: Output from harmonic_analysis()
        save_prefix: Prefix for output filenames
    
    Returns:
        List of figure objects
    """
    apply_pub_style()
    figs = []
    
    # Plot 1: Spectrum
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    fmax = result['fundamental']['freq'] * 12 if result['fundamental'] else 1e9
    mask = result['freqs'] <= fmax
    ax1.semilogy(result['freqs'][mask] / 1e6, result['spectrum'][mask], 'k-', lw=2)
    ax1.set_xlabel('Frequency (MHz)')
    ax1.set_ylabel('Magnitude (V)')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.locator_params(axis='x', nbins=6)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_spectrum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    figs.append(fig1)
    plt.show()
    
    # Plot 2: Harmonic bars
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    h_n = [h['n'] for h in result['harmonics']]
    h_mag = [h['mag'] * 1000 for h in result['harmonics']]
    ax2.bar(h_n, h_mag, color='white', edgecolor='black', linewidth=2, hatch='///')
    ax2.set_xlabel('Harmonic Number')
    ax2.set_ylabel('Magnitude (mV)')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    # Add THD annotation
    ax2.text(0.95, 0.95, f"THD = {result['thd']:.2f}%", transform=ax2.transAxes,
             fontsize=20, fontweight='bold', ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_harmonics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    figs.append(fig2)
    plt.show()
    
    return figs


# =============================================================================
# Frequency Sweep Analysis
# =============================================================================

def frequency_sweep(circuit_builder, freq_list, t_cycles, metric_fn, 
                    nodes=['vout'], verbose=True):
    """
    Sweep frequency and extract metrics at each point.
    
    Args:
        circuit_builder: Function(freq_hz) -> PySpice Circuit
        freq_list: List of frequencies (Hz)
        t_cycles: Number of RF cycles per simulation
        metric_fn: Function(sim_data, freq) -> dict of metrics
        nodes: Nodes to capture
        verbose: Print progress
    
    Returns:
        Dict with 'freq' array and metric arrays
    """
    results = {'freq': np.array(freq_list)}
    all_metrics = []
    
    for i, f in enumerate(freq_list):
        if verbose:
            print(f"[SWEEP] {i+1}/{len(freq_list)}: {f/1e9:.3f} GHz", end=' ')
        
        try:
            circuit = circuit_builder(f)
            t_period = 1.0 / f
            t_step = t_period / 40
            t_stop = t_cycles * t_period
            
            sim = get_simulator()
            data = sim.run_transient(circuit, t_step, t_stop, nodes)
            metrics = metric_fn(data, f)
            all_metrics.append(metrics)
            
            if verbose:
                print(f"-> OK")
        except Exception as e:
            if verbose:
                print(f"-> FAIL: {e}")
            all_metrics.append(None)
    
    # Aggregate metrics
    if all_metrics and all_metrics[0]:
        for key in all_metrics[0]:
            results[key] = np.array([m[key] if m else np.nan for m in all_metrics])
    
    return results


def plot_freq_sweep(results, y_keys, y_labels=None, title='Frequency Sweep', 
                    save_path=None):
    """
    Plot frequency sweep results.
    
    Args:
        results: Output from frequency_sweep()
        y_keys: List of metric keys to plot
        y_labels: Optional axis labels
        title: Plot title
        save_path: Optional save path
    
    Returns:
        Figure object
    """
    if y_labels is None:
        y_labels = y_keys
    
    n = len(y_keys)
    fig, axes = plt.subplots(n, 1, figsize=(9, 3*n), sharex=True)
    if n == 1:
        axes = [axes]
    
    freq_ghz = results['freq'] / 1e9
    
    for ax, key, label in zip(axes, y_keys, y_labels):
        ax.plot(freq_ghz, results[key], 'o-', lw=1.5, ms=5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Frequency (GHz)')
    fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# =============================================================================
# Sensitivity / Monte Carlo Analysis
# =============================================================================

def sensitivity_sweep(circuit_builder, param_name, param_values, base_params,
                      sim_params, metric_fn, nodes=['vout'], verbose=True):
    """
    Sweep one parameter and measure sensitivity.
    
    Args:
        circuit_builder: Function(params_dict) -> PySpice Circuit
        param_name: Name of parameter to sweep
        param_values: List of values for the parameter
        base_params: Dict of baseline parameter values
        sim_params: Dict with 'freq', 'n_cycles' for transient setup
        metric_fn: Function(sim_data) -> dict of metrics
        nodes: Nodes to capture
        verbose: Print progress
    
    Returns:
        Dict with 'values' and metric arrays
    """
    results = {'param': param_name, 'values': np.array(param_values)}
    all_metrics = []
    
    freq = sim_params['freq']
    t_period = 1.0 / freq
    t_step = t_period / 40
    t_stop = sim_params['n_cycles'] * t_period
    
    for i, val in enumerate(param_values):
        if verbose:
            print(f"[SENS] {param_name}={val:.4g} ({i+1}/{len(param_values)})", end=' ')
        
        try:
            params = base_params.copy()
            params[param_name] = val
            circuit = circuit_builder(params)
            
            sim = get_simulator()
            data = sim.run_transient(circuit, t_step, t_stop, nodes)
            metrics = metric_fn(data)
            all_metrics.append(metrics)
            
            if verbose:
                print("-> OK")
        except Exception as e:
            if verbose:
                print(f"-> FAIL: {e}")
            all_metrics.append(None)
    
    # Aggregate
    if all_metrics and all_metrics[0]:
        for key in all_metrics[0]:
            results[key] = np.array([m[key] if m else np.nan for m in all_metrics])
    
    return results


def monte_carlo(circuit_builder, param_specs, n_runs, sim_params, metric_fn,
                nodes=['vout'], verbose=True):
    """
    Monte Carlo analysis with random parameter variations.
    
    Args:
        circuit_builder: Function(params_dict) -> PySpice Circuit
        param_specs: Dict of {param: (nominal, tolerance_pct)}
                    e.g. {'C_in': (10e-12, 10)} for 10pF ±10%
        n_runs: Number of MC iterations
        sim_params: Dict with 'freq', 'n_cycles'
        metric_fn: Function(sim_data) -> dict of metrics
        nodes: Nodes to capture
        verbose: Print progress
    
    Returns:
        Dict with samples, metrics, and statistics
    """
    results = {'params': list(param_specs.keys()), 'samples': [], 'metrics': []}
    
    freq = sim_params['freq']
    t_period = 1.0 / freq
    t_step = t_period / 40
    t_stop = sim_params['n_cycles'] * t_period
    
    for i in range(n_runs):
        if verbose:
            print(f"[MC] Run {i+1}/{n_runs}", end=' ')
        
        # Random parameter values
        params = {}
        for name, (nom, tol) in param_specs.items():
            delta = np.random.uniform(-tol/100, tol/100)
            params[name] = nom * (1 + delta)
        
        results['samples'].append(params.copy())
        
        try:
            circuit = circuit_builder(params)
            sim = get_simulator()
            data = sim.run_transient(circuit, t_step, t_stop, nodes)
            metrics = metric_fn(data)
            results['metrics'].append(metrics)
            if verbose:
                print("-> OK")
        except Exception as e:
            if verbose:
                print(f"-> FAIL: {e}")
            results['metrics'].append(None)
    
    # Statistics
    valid = [m for m in results['metrics'] if m is not None]
    if valid:
        results['stats'] = {}
        for key in valid[0]:
            vals = [m[key] for m in valid]
            results['stats'][key] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals)
            }
    
    return results


def plot_sensitivity(results, metric_key, ylabel=None, title=None, save_path=None):
    """
    Plot sensitivity sweep results.
    
    Args:
        results: Output from sensitivity_sweep()
        metric_key: Metric to plot
        ylabel: Y-axis label
        title: Plot title
        save_path: Optional save path
    
    Returns:
        Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(results['values'], results[metric_key], 'o-', lw=1.5, ms=6)
    ax.set_xlabel(results['param'])
    ax.set_ylabel(ylabel or metric_key)
    ax.set_title(title or f"Sensitivity: {results['param']}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


def plot_monte_carlo(results, metric_key, xlabel=None, title='Monte Carlo', 
                     save_path=None):
    """
    Plot Monte Carlo histogram.
    
    Args:
        results: Output from monte_carlo()
        metric_key: Metric to plot
        xlabel: X-axis label
        title: Plot title
        save_path: Optional save path
    
    Returns:
        Figure object
    """
    vals = [m[metric_key] for m in results['metrics'] if m is not None]
    stats = results['stats'][metric_key]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(vals, bins=20, color='steelblue', edgecolor='k', alpha=0.7)
    ax.axvline(stats['mean'], color='red', ls='--', lw=2, label=f"μ={stats['mean']:.3g}")
    ax.axvline(stats['mean'] - stats['std'], color='orange', ls=':', lw=1.5)
    ax.axvline(stats['mean'] + stats['std'], color='orange', ls=':', lw=1.5, 
               label=f"σ={stats['std']:.3g}")
    
    ax.set_xlabel(xlabel or metric_key)
    ax.set_ylabel('Count')
    cv = stats['std'] / stats['mean'] * 100 if stats['mean'] != 0 else 0
    ax.set_title(f"{title} (N={len(vals)}, CV={cv:.1f}%)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# =============================================================================
# Metric Extraction Helpers
# =============================================================================

def extract_dc_metrics(data, r_load, v_out_key='vout', v_in_key=None):
    """
    Extract DC rectifier metrics from transient data.
    
    Args:
        data: Simulation result dict
        r_load: Load resistance (Ohms)
        v_out_key: Key for output voltage
        v_in_key: Optional key for input voltage
    
    Returns:
        Dict with v_dc, ripple, power metrics
    """
    v_out = np.array(data[v_out_key])
    
    # Steady-state: last 20%
    ss = int(0.8 * len(v_out))
    v_dc = np.mean(v_out[ss:])
    ripple = np.max(v_out[ss:]) - np.min(v_out[ss:])
    p_out = v_dc**2 / r_load
    
    result = {
        'v_dc': v_dc,
        'ripple': ripple,
        'i_load': v_dc / r_load,
        'p_out': p_out
    }
    
    if v_in_key and v_in_key in data:
        v_in = np.array(data[v_in_key])
        result['v_in_pk'] = np.max(np.abs(v_in[ss:]))
    
    return result


# =============================================================================
# Legacy Simulation Functions (ngspice subprocess fallback)
# =============================================================================

import subprocess
import tempfile
import os

def run_transient_ngspice(netlist, t_stop, n_cycles, t_step, num_signals=2):
    """
    Run transient simulation using ngspice subprocess (fallback).
    
    Args:
        netlist: SPICE netlist string
        t_stop: End time (seconds)
        n_cycles: Number of RF cycles
        t_step: Time step (seconds)
        num_signals: Number of signals to parse
    
    Returns:
        Dict with 'time' and signal arrays
    """
    print(f"[INFO] Transient: {t_stop*1e9:.2f} ns, {n_cycles} cycles")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        netlist_path = os.path.join(tmpdir, 'circuit.sp')
        output_path = os.path.join(tmpdir, 'output.txt')
        
        with open(netlist_path, 'w') as f:
            f.write(netlist)
        
        result = subprocess.run(['ngspice', '-b', netlist_path],
                                capture_output=True, text=True, cwd=tmpdir)
        
        if result.returncode != 0:
            raise RuntimeError(f"ngspice failed: {result.stderr}")
        
        if not os.path.exists(output_path):
            raise RuntimeError("No output file")
        
        # Parse wrdata format
        expected_cols = 2 + num_signals * 2
        data = {f'signal_{i}': [] for i in range(num_signals)}
        data['time'] = []
        
        with open(output_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= expected_cols:
                    try:
                        data['time'].append(float(parts[0]))
                        for i in range(num_signals):
                            data[f'signal_{i}'].append(float(parts[3 + i*2]))
                    except (ValueError, IndexError):
                        continue
        
        return {k: np.array(v) for k, v in data.items()}


def run_halfwave_simulation(netlist, t_stop, n_cycles, t_step):
    """Run halfwave rectifier simulation (2 signals)."""
    data = run_transient_ngspice(netlist, t_stop, n_cycles, t_step, num_signals=2)
    return {'time': data['time'], 'rf_in': data['signal_0'], 'vout': data['signal_1']}


def run_dickson_simulation(netlist, t_stop, n_cycles, t_step):
    """Run Dickson simulation (4 signals)."""
    data = run_transient_ngspice(netlist, t_stop, n_cycles, t_step, num_signals=4)
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

def plot_halfwave_results(data, f_rf, v_rf_amplitude, r_load, 
                          save_prefix='halfwave_transient'):
    """Plot halfwave rectifier transient results as individual plots."""
    apply_pub_style()
    time = data['time'] * 1e9
    v_rf = data['rf_in']
    v_out = data['vout']
    
    # Steady-state metrics
    ss = int(0.8 * len(v_out))
    v_dc = np.mean(v_out[ss:])
    ripple = np.max(v_out[ss:]) - np.min(v_out[ss:])
    
    print(f"\n{'='*50}")
    print(f"DC Output:    {v_dc*1e3:.2f} mV")
    print(f"Ripple:       {ripple*1e3:.2f} mV pk-pk")
    print(f"Load Current: {v_dc/r_load*1e6:.2f} uA")
    print(f"Power:        {v_dc**2/r_load*1e6:.2f} uW")
    print(f"{'='*50}")
    
    # Plot 1: Input signal
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(time, v_rf, 'k-', lw=2)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Input Voltage (V)')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.locator_params(axis='x', nbins=6)
    ax1.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_input.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 2: Output signal
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(time, v_out, 'k-', lw=2, label='Output')
    ax2.axhline(v_dc, color='k', ls='--', lw=2, label=f'DC = {v_dc*1e3:.2f} mV')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Output Voltage (V)')
    ax2.legend(loc='lower right')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.locator_params(axis='x', nbins=6)
    ax2.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_output.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()


def plot_dickson_results(data, n_stages, f_rf, v_rf_amplitude, r_load,
                         save_prefix='dickson_transient'):
    """Plot Dickson charge pump transient results as individual plots."""
    apply_pub_style()
    time = data['time'] * 1e9
    v_rf = data['rf_in']
    v_n1 = data['node1']
    v_n2 = data['node2']
    v_out = data['vout']
    
    ss = int(0.8 * len(v_out))
    v_dc = np.mean(v_out[ss:])
    v_theo = n_stages * v_rf_amplitude
    
    print(f"\n{'='*50}")
    print(f"Theoretical Max: {v_theo*1e3:.1f} mV")
    print(f"Actual DC:       {v_dc*1e3:.2f} mV")
    print(f"Efficiency:      {v_dc/v_theo*100:.1f}%")
    print(f"{'='*50}")
    
    # Plot 1: Input
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(time, v_rf, 'k-', lw=2)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Input Voltage (V)')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.locator_params(axis='x', nbins=6)
    ax1.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_input.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 2: Intermediate nodes
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(time, v_n1, 'k-', lw=2, label='Node 1')
    ax2.plot(time, v_n2, 'k--', lw=2, label='Node 2')
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend(loc='lower right')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.locator_params(axis='x', nbins=6)
    ax2.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_intermediate.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 3: Output
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.plot(time, v_out, 'k-', lw=2, label='Output')
    ax3.axhline(v_dc, color='k', ls='--', lw=2, label=f'DC = {v_dc*1e3:.1f} mV')
    ax3.axhline(v_theo, color='k', ls=':', lw=2, label=f'Theo = {v_theo*1e3:.1f} mV')
    ax3.set_xlabel('Time (ns)')
    ax3.set_ylabel('Output Voltage (V)')
    ax3.legend(loc='lower right')
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    ax3.locator_params(axis='x', nbins=6)
    ax3.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_output.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
