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


# =============================================================================
# WiFi Signal Generation (OFDM-like Multi-tone)
# =============================================================================

def generate_wifi_pwl_source(v_amplitude, center_freq=2.45e9, bw=20e6, n_tones=13,
                             t_stop=100e-9, t_step=None):
    """
    Generate a pseudo-WiFi OFDM signal as PWL (Piecewise Linear) data for SPICE.
    
    802.11n WiFi at 2.4 GHz characteristics:
    - 20 MHz channel bandwidth
    - 64 subcarriers (52 data + 4 pilot + 8 null)
    - Subcarrier spacing: 312.5 kHz
    - High PAPR (Peak-to-Average Power Ratio) ~10-12 dB
    
    This function creates a multi-tone approximation with:
    - n_tones spread uniformly across bw MHz
    - Random initial phases for realistic PAPR
    - Total power scaled to match single-tone equivalent
    
    Args:
        v_amplitude: Equivalent single-tone amplitude (V peak)
        center_freq: Center frequency in Hz (default 2.45 GHz)
        bw: Bandwidth in Hz (default 20 MHz for WiFi)
        n_tones: Number of subcarriers to simulate (default 13)
        t_stop: Simulation duration in seconds
        t_step: Time step (default: 1/40 of highest frequency period)
    
    Returns:
        tuple: (time_array, voltage_array) for PWL source
    """
    if t_step is None:
        f_max = center_freq + bw/2
        t_step = 1.0 / (40 * f_max)
    
    t = np.arange(0, t_stop, t_step)
    
    # Generate subcarrier frequencies (spread across bandwidth)
    freq_offsets = np.linspace(-bw/2, bw/2, n_tones)
    freqs = center_freq + freq_offsets
    
    # Random phases for each subcarrier (creates realistic PAPR)
    np.random.seed(42)  # Reproducible
    phases = np.random.uniform(0, 2*np.pi, n_tones)
    
    # Each tone amplitude: scale so total power = single-tone power
    # P_total = n_tones * (A_tone^2 / 2) = A_single^2 / 2
    # A_tone = A_single / sqrt(n_tones)
    a_tone = v_amplitude / np.sqrt(n_tones)
    
    # Sum all tones
    v = np.zeros_like(t)
    for i, (f, phi) in enumerate(zip(freqs, phases)):
        v += a_tone * np.sin(2 * np.pi * f * t + phi)
    
    return t, v


def get_wifi_spice_source(v_amplitude, center_freq=2.45e9, bw=20e6, n_tones=13):
    """
    Generate SPICE voltage source string for WiFi-like multi-tone signal.
    
    Uses multiple SIN sources in series to create OFDM-like spectrum.
    
    Args:
        v_amplitude: Equivalent single-tone amplitude (V peak)
        center_freq: Center frequency in Hz
        bw: Bandwidth in Hz
        n_tones: Number of subcarriers
    
    Returns:
        str: SPICE netlist fragment with multi-tone source
    """
    freq_offsets = np.linspace(-bw/2, bw/2, n_tones)
    freqs = center_freq + freq_offsets
    
    np.random.seed(42)
    phases = np.random.uniform(0, 2*np.pi, n_tones)
    phase_degrees = np.degrees(phases)
    
    a_tone = v_amplitude / np.sqrt(n_tones)
    
    lines = [f"* WiFi-like OFDM source: {n_tones} tones, {bw/1e6:.0f} MHz BW"]
    
    # Create series of voltage sources
    for i, (f, phi_deg) in enumerate(zip(freqs, phase_degrees)):
        node_p = f"wifi_n{i}" if i < n_tones - 1 else "rf_source"
        node_n = f"wifi_n{i+1}" if i < n_tones - 1 else "0"
        if i == 0:
            node_n = "0"
            node_p = f"wifi_n{i}"
        lines.append(f"Vwifi{i} {node_p} {node_n} SIN(0 {a_tone} {f} 0 0 {phi_deg})")
    
    # Connect first node to rf_source
    lines.append(f"Vwifi_link rf_source wifi_n0 0")
    
    return "\n".join(lines)


# =============================================================================
# WiFi Signal Generation using CommPy (Realistic 802.11 OFDM)
# =============================================================================

def generate_wifi_commpy(v_amplitude, center_freq=2.45e9, n_symbols=10,
                          t_stop=100e-9, mcs=0, channel_bw=20):
    """
    Generate realistic WiFi 802.11 OFDM signal using scikit-commpy.
    
    Uses CommPy's wifi80211 module for standards-compliant 802.11 waveforms:
    - Proper OFDM subcarrier structure (52 data + 4 pilot + 8 null for HT)
    - Correct subcarrier spacing (312.5 kHz)
    - Realistic PAPR characteristics
    - Cyclic prefix support
    
    Args:
        v_amplitude: Target RMS amplitude (V)
        center_freq: Center frequency in Hz (default 2.45 GHz)
        n_symbols: Number of OFDM symbols to generate
        t_stop: Desired simulation duration in seconds
        mcs: Modulation and Coding Scheme (0-7 for HT)
        channel_bw: Channel bandwidth in MHz (20 or 40)
    
    Returns:
        tuple: (time_array, voltage_array) for PWL source
        dict: Signal info (papr_db, n_subcarriers, etc.)
    """
    try:
        from commpy.modulation import QAMModem, ofdm_tx
        import numpy as np
    except ImportError:
        print("[WARNING] scikit-commpy not installed. Using multi-tone approximation.")
        t, v = generate_wifi_pwl_source(v_amplitude, center_freq)
        return t, v, {'papr_db': 10, 'method': 'multi-tone', 'n_subcarriers': 13}
    
    # 802.11n HT parameters
    n_fft = 64
    n_data_subcarriers = 52  # 48 data + 4 pilot
    cp_length = 16  # Cyclic prefix
    subcarrier_spacing = 312.5e3  # Hz
    symbol_duration = 1 / subcarrier_spacing  # 3.2 us
    total_symbol_duration = symbol_duration + (cp_length / n_fft) * symbol_duration  # ~4 us
    
    # MCS to modulation mapping (simplified)
    mcs_modulation = {
        0: 2,   # BPSK
        1: 4,   # QPSK
        2: 4,   # QPSK
        3: 16,  # 16-QAM
        4: 16,  # 16-QAM
        5: 64,  # 64-QAM
        6: 64,  # 64-QAM
        7: 64,  # 64-QAM
    }
    m_order = mcs_modulation.get(mcs, 4)  # Default QPSK
    
    # Generate random bits and modulate
    bits_per_symbol = int(np.log2(m_order))
    n_bits = n_symbols * n_data_subcarriers * bits_per_symbol
    np.random.seed(42)  # Reproducible
    bits = np.random.randint(0, 2, n_bits)
    
    # QAM modulation
    modem = QAMModem(m_order)
    symbols = modem.modulate(bits)
    
    # Reshape to OFDM symbols
    symbols = symbols.reshape(n_symbols, -1)
    
    # Create OFDM spectrum (place data on subcarriers)
    ofdm_symbols = np.zeros((n_symbols, n_fft), dtype=complex)
    
    # 802.11 subcarrier allocation: -26 to -1, +1 to +26
    data_indices = list(range(-26, 0)) + list(range(1, 27))
    data_indices = [(i + n_fft) % n_fft for i in data_indices]
    
    for i in range(n_symbols):
        ofdm_symbols[i, data_indices] = symbols[i, :len(data_indices)]
    
    # IFFT to get time-domain signal
    time_symbols = np.fft.ifft(ofdm_symbols, axis=1)
    
    # Add cyclic prefix
    cp = time_symbols[:, -cp_length:]
    time_with_cp = np.concatenate([cp, time_symbols], axis=1)
    
    # Flatten to continuous waveform
    baseband = time_with_cp.flatten()
    
    # Scale to desired amplitude
    rms_current = np.sqrt(np.mean(np.abs(baseband)**2))
    baseband = baseband / rms_current * v_amplitude
    
    # Calculate PAPR
    peak = np.max(np.abs(baseband))
    rms = np.sqrt(np.mean(np.abs(baseband)**2))
    papr_db = 20 * np.log10(peak / rms)
    
    # Upconvert to RF
    sample_rate = n_fft * subcarrier_spacing
    n_samples = len(baseband)
    t_baseband = np.arange(n_samples) / sample_rate
    
    # Upsample for RF frequency (need ~40x carrier frequency)
    upsample_factor = max(1, int(40 * center_freq / sample_rate))
    t_rf = np.linspace(0, t_baseband[-1], len(baseband) * upsample_factor)
    
    # Interpolate baseband
    from scipy.interpolate import interp1d
    interp_real = interp1d(t_baseband, np.real(baseband), kind='linear', fill_value='extrapolate')
    interp_imag = interp1d(t_baseband, np.imag(baseband), kind='linear', fill_value='extrapolate')
    
    baseband_up = interp_real(t_rf) + 1j * interp_imag(t_rf)
    
    # RF carrier
    carrier = np.exp(1j * 2 * np.pi * center_freq * t_rf)
    rf_signal = np.real(baseband_up * carrier)
    
    # Trim to desired duration
    if t_stop is not None and t_rf[-1] > t_stop:
        mask = t_rf <= t_stop
        t_rf = t_rf[mask]
        rf_signal = rf_signal[mask]
    
    info = {
        'papr_db': papr_db,
        'method': 'commpy_ofdm',
        'n_subcarriers': n_data_subcarriers,
        'n_fft': n_fft,
        'mcs': mcs,
        'modulation': m_order,
        'symbol_duration_us': total_symbol_duration * 1e6,
        'sample_rate_mhz': sample_rate / 1e6,
    }
    
    return t_rf, rf_signal, info


def plot_wifi_spectrum_commpy(v_amplitude, center_freq=2.45e9, n_symbols=10,
                               mcs=0, save_prefix='wifi_spectrum'):
    """
    Plot WiFi signal spectrum using CommPy-generated OFDM (publication style).
    
    Args:
        v_amplitude: Target RMS amplitude (V)
        center_freq: Center frequency in Hz
        n_symbols: Number of OFDM symbols
        mcs: Modulation and Coding Scheme
        save_prefix: Output filename prefix
    
    Returns:
        Figure object, signal info dict
    """
    apply_pub_style()
    
    # Generate signal
    t, v, info = generate_wifi_commpy(v_amplitude, center_freq, n_symbols,
                                       t_stop=2e-6, mcs=mcs)
    
    # FFT analysis
    n_fft = len(v)
    dt = t[1] - t[0] if len(t) > 1 else 1e-12
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(n_fft, dt)
    mag = np.abs(fft_v) / n_fft * 2
    
    # Convert to dB (relative to carrier)
    mag_db = 20 * np.log10(mag / np.max(mag) + 1e-10)
    
    # Plot around center frequency
    bw = 40e6  # Show ±40 MHz
    mask = (freqs > center_freq - bw) & (freqs < center_freq + bw)
    freq_offset_mhz = (freqs[mask] - center_freq) / 1e6
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Spectrum plot (dB)
    ax1.plot(freq_offset_mhz, mag_db[mask], 'k-', lw=0.8, alpha=0.8)
    ax1.set_xlabel('Frequency Offset from 2.45 GHz (MHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_xlim([-30, 30])
    ax1.set_ylim([-60, 5])
    
    # Add WiFi channel bandwidth shading
    ax1.axvspan(-10, 10, alpha=0.15, color='blue', label='20 MHz Channel')
    ax1.axvline(-10, color='blue', ls='--', lw=1.5, alpha=0.7)
    ax1.axvline(10, color='blue', ls='--', lw=1.5, alpha=0.7)
    ax1.legend(loc='upper right', fontsize=16)
    
    # Add info box
    info_text = (f"CommPy OFDM WiFi\n"
                 f"MCS: {info['mcs']} ({info['modulation']}-QAM)\n"
                 f"PAPR: {info['papr_db']:.1f} dB\n"
                 f"{info['n_subcarriers']} subcarriers")
    ax1.text(0.05, 0.05, info_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='bottom', bbox=dict(boxstyle='round', 
             facecolor='white', edgecolor='black', alpha=0.9))
    
    # Time-domain plot (first ~100 ns)
    t_ns = t * 1e9
    mask_time = t_ns < 100
    ax2.plot(t_ns[mask_time], v[mask_time] * 1000, 'k-', lw=0.8)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Amplitude (mV)')
    ax2.set_xlim([0, 100])
    
    # Calculate and show peak
    peak_v = np.max(np.abs(v)) * 1000
    rms_v = np.sqrt(np.mean(v**2)) * 1000
    ax2.axhline(peak_v, color='red', ls=':', lw=1.5, alpha=0.7)
    ax2.axhline(-peak_v, color='red', ls=':', lw=1.5, alpha=0.7)
    ax2.axhline(rms_v, color='green', ls='--', lw=1.5, alpha=0.7)
    ax2.axhline(-rms_v, color='green', ls='--', lw=1.5, alpha=0.7)
    
    ax2.text(0.95, 0.95, f"Peak: {peak_v:.1f} mV\nRMS: {rms_v:.1f} mV",
             transform=ax2.transAxes, fontsize=14, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.locator_params(axis='x', nbins=7)
        ax.locator_params(axis='y', nbins=5)
    
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    print(f"[INFO] CommPy OFDM: PAPR={info['papr_db']:.1f} dB, {info['n_subcarriers']} subcarriers")
    plt.show()
    return fig, info


def plot_wifi_spectrum(v_amplitude, center_freq=2.45e9, bw=20e6, n_tones=13,
                       save_prefix='wifi_spectrum'):
    """
    Plot the WiFi-like signal spectrum (publication style).
    
    Args:
        v_amplitude: Equivalent single-tone amplitude
        center_freq: Center frequency
        bw: Bandwidth
        n_tones: Number of tones
        save_prefix: Output filename prefix
    
    Returns:
        Figure object
    """
    apply_pub_style()
    
    # Generate time-domain signal
    t, v = generate_wifi_pwl_source(v_amplitude, center_freq, bw, n_tones, 
                                     t_stop=1e-6, t_step=1e-12)
    
    # FFT
    n_fft = len(v)
    dt = t[1] - t[0]
    fft_v = np.fft.fft(v)
    freqs = np.fft.fftfreq(n_fft, dt)
    mag = np.abs(fft_v) / n_fft * 2  # Single-sided
    
    # Plot only positive frequencies around center
    mask = (freqs > center_freq - bw) & (freqs < center_freq + bw)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot((freqs[mask] - center_freq) / 1e6, mag[mask] * 1000, 'k-', lw=1.5)
    ax.set_xlabel('Frequency Offset from 2.45 GHz (MHz)')
    ax.set_ylabel('Amplitude (mV)')
    ax.set_xlim([-bw/1e6, bw/1e6])
    
    # Add channel bandwidth annotation
    ax.axvline(-bw/2/1e6, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax.axvline(bw/2/1e6, color='gray', ls='--', lw=1.5, alpha=0.7)
    ax.text(0.95, 0.95, f"WiFi 20 MHz Channel\n{n_tones} subcarriers",
            transform=ax.transAxes, fontsize=16, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.locator_params(axis='x', nbins=7)
    ax.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    return fig


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
    """Plot halfwave rectifier transient results - input and output superimposed."""
    apply_pub_style()
    time = data['time'] * 1e9  # Convert to ns
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
    
    # Combined plot: Input and Output superimposed
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Input signal (left y-axis)
    color_in = 'black'
    ax1.plot(time, v_rf, color=color_in, ls='-', lw=2, label='RF Input')
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('RF Input (V)', color=color_in)
    ax1.tick_params(axis='y', labelcolor=color_in)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    
    # Output signal (right y-axis)
    ax2 = ax1.twinx()
    color_out = 'black'
    ax2.plot(time, v_out, color=color_out, ls='--', lw=2, label='DC Output')
    ax2.axhline(v_dc, color=color_out, ls=':', lw=2, label=f'DC = {v_dc*1e3:.2f} mV')
    ax2.set_ylabel('DC Output (V)', color=color_out)
    ax2.tick_params(axis='y', labelcolor=color_out)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax1.locator_params(axis='x', nbins=6)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_waveforms.png')
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


# =============================================================================
# Frequency Sweep Plotting (Publication Style)
# =============================================================================

def plot_freq_sweep_dc(freq_ghz, v_dc_mv, save_prefix='halfwave_freq_sweep'):
    """
    Plot DC output vs frequency (publication style).
    
    Args:
        freq_ghz: Frequency array in GHz
        v_dc_mv: DC output array in mV
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freq_ghz, v_dc_mv, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('DC Output (mV)')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_dc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    return fig


def plot_freq_sweep_power(freq_ghz, p_out_uw, save_prefix='halfwave_freq_sweep'):
    """
    Plot output power vs frequency (publication style).
    
    Args:
        freq_ghz: Frequency array in GHz
        p_out_uw: Output power array in uW
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(freq_ghz, p_out_uw, 'k--', lw=2, marker='s', ms=10, mfc='white', mew=2)
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Output Power (uW)')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.locator_params(axis='x', nbins=6)
    ax.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_power.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    return fig


# =============================================================================
# Sensitivity Analysis Plotting (Publication Style)
# =============================================================================

def plot_sens_input_amplitude(v_amp_mv, v_dc_mv, save_prefix='halfwave_sens'):
    """
    Plot DC output vs input amplitude (publication style).
    
    Args:
        v_amp_mv: Input amplitude array in mV
        v_dc_mv: DC output array in mV
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(v_amp_mv, v_dc_mv, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2)
    ax.set_xlabel('Input Amplitude (mV)')
    ax.set_ylabel('DC Output (mV)')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_input_amplitude.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    return fig


def plot_sens_combined(v_amp_mv, v_dc_amp_mv, c_out_pf, v_dc_cout_mv, 
                       save_prefix='halfwave_sens'):
    """
    Combined sensitivity plot: Input amplitude (top x-axis) and Cout (bottom x-axis)
    both vs DC output (shared y-axis). Publication style.
    
    This shows two different sensitivities in one plot:
    - How DC output varies with input RF amplitude
    - How DC output varies with output capacitor size
    
    Args:
        v_amp_mv: Input amplitude array in mV
        v_dc_amp_mv: DC output array for amplitude sweep in mV
        c_out_pf: Output capacitor array in pF
        v_dc_cout_mv: DC output array for capacitor sweep in mV
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Bottom x-axis: Output capacitor (log scale)
    color1 = 'black'
    line1, = ax1.semilogx(c_out_pf, v_dc_cout_mv, color=color1, ls='--', lw=2, 
                          marker='s', ms=10, mfc='white', mew=2, label='vs C_out')
    ax1.set_xlabel('Output Capacitor (pF)', color=color1)
    ax1.set_ylabel('DC Output (mV)')
    ax1.tick_params(axis='x', labelcolor=color1)
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    
    # Top x-axis: Input amplitude (linear scale)
    ax2 = ax1.twiny()
    color2 = 'black'
    line2, = ax2.plot(v_amp_mv, v_dc_amp_mv, color=color2, ls='-', lw=2,
                      marker='o', ms=10, mfc='white', mew=2, label='vs V_in')
    ax2.set_xlabel('Input Amplitude (mV)', color=color2)
    ax2.tick_params(axis='x', labelcolor=color2)
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    
    # Combined legend
    ax1.legend([line2, line1], ['vs Input Amplitude (top axis)', 'vs Output Capacitor (bottom axis)'],
               loc='lower right')
    
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_combined.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    return fig


def plot_sens_load_resistance(r_load_kohm, v_dc_mv, p_out_uw, save_prefix='halfwave_sens'):
    """
    Plot DC output and power vs load resistance (publication style, dual y-axis).
    
    Args:
        r_load_kohm: Load resistance array in kOhm
        v_dc_mv: DC output array in mV
        p_out_uw: Output power array in uW
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(r_load_kohm, v_dc_mv, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2, label='DC Voltage')
    ax.set_xlabel('Load Resistance (kOhm)')
    ax.set_ylabel('DC Output (mV)')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    
    ax_right = ax.twinx()
    ax_right.semilogx(r_load_kohm, p_out_uw, 'k--', lw=2, marker='s', ms=10, mfc='white', mew=2, label='Power')
    ax_right.set_ylabel('Power (uW)')
    for spine in ax_right.spines.values():
        spine.set_linewidth(2)
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_right.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_load_resistance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    return fig


def plot_sens_capacitor_dc(c_out_pf, v_dc_mv, save_prefix='halfwave_sens'):
    """
    Plot DC output vs output capacitor (publication style).
    
    Args:
        c_out_pf: Output capacitor array in pF
        v_dc_mv: DC output array in mV
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(c_out_pf, v_dc_mv, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2)
    ax.set_xlabel('Output Capacitor (pF)')
    ax.set_ylabel('DC Output (mV)')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_cout_dc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    return fig


def plot_sens_capacitor_ripple(c_out_pf, ripple_mv, save_prefix='halfwave_sens'):
    """
    Plot ripple vs output capacitor (publication style).
    
    Args:
        c_out_pf: Output capacitor array in pF
        ripple_mv: Ripple array in mV pk-pk
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(c_out_pf, ripple_mv, 'k--', lw=2, marker='s', ms=10, mfc='white', mew=2)
    ax.set_xlabel('Output Capacitor (pF)')
    ax.set_ylabel('Ripple (mV pk-pk)')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}_cout_ripple.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    return fig


# =============================================================================
# Monte Carlo Plotting (Publication Style)
# =============================================================================

def plot_mc_histogram(values, mean, std, xlabel, hatch='///', save_name='mc_hist.png'):
    """
    Plot Monte Carlo histogram (publication style).
    
    Args:
        values: Array of metric values
        mean: Mean value
        std: Standard deviation
        xlabel: X-axis label
        hatch: Hatch pattern for bars
        save_name: Output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(values, bins=15, color='white', edgecolor='black', linewidth=2, hatch=hatch)
    ax.axvline(mean, color='black', ls='--', lw=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.locator_params(axis='x', nbins=5)
    ax.text(0.95, 0.95, f"mean = {mean:.2f}\nstd = {std:.2f}",
            transform=ax.transAxes, fontsize=18, fontweight='bold', ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    plt.tight_layout()
    save_path = get_save_path(save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    return fig


def plot_mc_dc(v_dc_mv, stats, save_prefix='halfwave_mc'):
    """
    Plot Monte Carlo DC output histogram (publication style).
    
    Args:
        v_dc_mv: Array of DC output values in mV
        stats: Dict with 'mean' and 'std' keys
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    return plot_mc_histogram(
        v_dc_mv, 
        stats['mean'], stats['std'],
        'DC Output (mV)', 
        hatch='///',
        save_name=f'{save_prefix}_dc.png'
    )


def plot_mc_ripple(ripple_mv, stats, save_prefix='halfwave_mc'):
    """
    Plot Monte Carlo ripple histogram (publication style).
    
    Args:
        ripple_mv: Array of ripple values in mV
        stats: Dict with 'mean' and 'std' keys
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    return plot_mc_histogram(
        ripple_mv,
        stats['mean'], stats['std'],
        'Ripple (mV)',
        hatch='xxx',
        save_name=f'{save_prefix}_ripple.png'
    )


def plot_mc_power(p_out_uw, stats, save_prefix='halfwave_mc'):
    """
    Plot Monte Carlo power histogram (publication style).
    
    Args:
        p_out_uw: Array of power values in uW
        stats: Dict with 'mean' and 'std' keys
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    return plot_mc_histogram(
        p_out_uw,
        stats['mean'], stats['std'],
        'Power (uW)',
        hatch='...',
        save_name=f'{save_prefix}_power.png'
    )


# =============================================================================
# Frequency Stability Plotting (Publication Style)
# =============================================================================

def plot_freq_stability(freq_mhz_offset, v_dc_normalized, bw_3db_mhz=None, 
                        wifi_bw_mhz=20, save_prefix='halfwave_freq_stability'):
    """
    Plot frequency stability / narrowband response (publication style).
    Shows rectifier response across frequency drift, with WiFi channel bandwidth indicated.
    
    Args:
        freq_mhz_offset: Frequency offset from center in MHz (e.g., -20 to +20)
        v_dc_normalized: Normalized DC output (1.0 at center frequency)
        bw_3db_mhz: Optional -3dB bandwidth annotation in MHz
        wifi_bw_mhz: WiFi channel bandwidth to shade (default 20 MHz)
        save_prefix: Prefix for output filename
    
    Returns:
        Figure object
    """
    apply_pub_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Shade WiFi channel bandwidth region
    ax.axvspan(-wifi_bw_mhz/2, wifi_bw_mhz/2, alpha=0.15, color='gray',
               label=f'WiFi {wifi_bw_mhz} MHz channel')
    
    # Plot normalized response
    ax.plot(freq_mhz_offset, v_dc_normalized, 'k-', lw=2, marker='o', ms=8, mfc='white', mew=2,
            label='DC Response')
    
    # Add -3dB reference line
    ax.axhline(1/np.sqrt(2), color='black', ls='--', lw=2, label='-3 dB')
    ax.axhline(1.0, color='black', ls=':', lw=1.5, alpha=0.5)
    
    # Mark ±10 MHz drift boundaries (WiFi channel edge drift)
    ax.axvline(-10, color='gray', ls=':', lw=1.5, alpha=0.7)
    ax.axvline(10, color='gray', ls=':', lw=1.5, alpha=0.7)
    
    ax.set_xlabel('Center Frequency Offset (MHz)')
    ax.set_ylabel('Normalized DC Output')
    ax.set_ylim([0, 1.15])
    
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.locator_params(axis='x', nbins=9)
    ax.locator_params(axis='y', nbins=5)
    
    # Add bandwidth annotation if provided
    if bw_3db_mhz is not None:
        ax.text(0.95, 0.85, f"BW$_{{-3dB}}$ = {bw_3db_mhz:.1f} MHz",
                transform=ax.transAxes, fontsize=16, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    ax.legend(loc='lower left', fontsize=16)
    plt.tight_layout()
    save_path = get_save_path(f'{save_prefix}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    return fig
