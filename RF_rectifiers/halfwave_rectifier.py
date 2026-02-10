#!/usr/bin/env python3
"""
================================================================================
Half-Wave RF Rectifier - Full Analysis Suite at 2.45 GHz
================================================================================
RF energy harvesting rectifier with:
    1. Transient simulation
    2. Harmonic analysis
    3. Frequency sweep
    4. Sensitivity / Monte Carlo analysis

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

import numpy as np
from pathlib import Path

from utility import (
    calculate_esr_from_q,
    run_halfwave_simulation,
    plot_halfwave_results,
    harmonic_analysis,
    plot_harmonics,
    extract_dc_metrics,
    apply_pub_style,
    get_save_path,
    LINE_STYLES,
    MARKER_STYLES,
    HATCH_STYLES
)


# =============================================================================
# Circuit Parameters - 2.45 GHz RF Energy Harvesting
# =============================================================================

F_RF = 2.45e9           # 2.45 GHz ISM band
T_RF = 1 / F_RF

V_RF_AMPLITUDE = 0.3    # 300 mV peak (typical RF harvesting level)
R_SOURCE = 50           # 50 #note- not sure what is the impedance looking from the rectifier to the matching, but for rectifier only, i set it to 50 ohms

C_IN = 100e-12           # 100 pF input coupling
C_OUT = 100e-12         # 100 pF output smoothing
R_LOAD = 10e3           # 10 kOhm load

CAP_Q = 30             # 30 is a more realistic number for commericial RF capacitors at GHz frequencies

N_CYCLES = 100          # Enough cycles for steady-state
T_STEP = T_RF / 40
T_STOP = N_CYCLES * T_RF

DIODE_MODEL_FILE = "diode_models.lib"
DIODE_MODEL_NAME = "SMS7630"


# =============================================================================
# Netlist Builder
# =============================================================================

def build_netlist(freq=F_RF, v_amp=V_RF_AMPLITUDE, c_in=C_IN, c_out=C_OUT, 
                  r_load=R_LOAD, n_cycles=N_CYCLES):
    """Build SPICE netlist with given parameters."""
    t_rf = 1.0 / freq
    t_step = t_rf / 40
    t_stop = n_cycles * t_rf
    
    esr_in = calculate_esr_from_q(c_in, CAP_Q, freq)
    esr_out = calculate_esr_from_q(c_out, CAP_Q, freq)
    
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / DIODE_MODEL_FILE
    


    ##############Note from Chen: this circuit is the replicate simulation of the RF rectifier in figure 2 of the paper. netlist is checked by Chen
    netlist = f"""Half-Wave RF Rectifier 
.include {model_path}

Vrf rf_source 0 SIN(0 {v_amp} {freq})
Rsource rf_source rf_in {R_SOURCE}

Cin rf_in node_c_in {c_in}
Resr_in node_c_in diode_anode {esr_in}

D2 0 diode_anode {DIODE_MODEL_NAME}

D1 diode_anode vout {DIODE_MODEL_NAME}

Cout vout node_c_out {c_out}
Resr_out node_c_out 0 {esr_out}

Rload vout 0 {r_load}

.control
set filetype=ascii
set wr_vecnames
option noacct
tran {t_step} {t_stop} uic
wrdata output.txt time v(rf_in) v(vout)
quit
.endc

.end
"""
    return netlist, t_stop, n_cycles, t_step


# =============================================================================
# Analysis 1: Transient + Harmonic Analysis
# =============================================================================

def run_transient_and_harmonic():
    """Run transient simulation and harmonic analysis."""
    print("\n" + "="*60)
    print("ANALYSIS 1: TRANSIENT + HARMONIC ANALYSIS")
    print("="*60)
    
    netlist, t_stop, n_cycles, t_step = build_netlist()
    
    # Transient
    data = run_halfwave_simulation(netlist, t_stop, n_cycles, t_step)
    plot_halfwave_results(data, F_RF, V_RF_AMPLITUDE, R_LOAD, 
                          save_prefix='halfwave_transient')
    
    # Harmonic analysis on output
    harm = harmonic_analysis(data['time'], data['vout'], F_RF, n_harmonics=8)
    
    print(f"\nHarmonic Analysis Results:")
    print(f"  DC Component:  {harm['dc']*1000:.3f} mV")
    print(f"  Fundamental:   {harm['fundamental']['mag']*1000:.3f} mV @ {harm['fundamental']['freq']/1e9:.3f} GHz")
    print(f"  THD:           {harm['thd']:.2f}%")
    print("\n  Harmonic breakdown:")
    for h in harm['harmonics'][:5]:
        print(f"    H{h['n']}: {h['mag']*1000:.4f} mV")
    
    plot_harmonics(harm, save_prefix='halfwave_harmonics')
    
    return data, harm


# =============================================================================
# Analysis 2: Frequency Sweep
# =============================================================================

def run_frequency_sweep():
    """Sweep frequency across common RF harvesting bands."""
    print("\n" + "="*60)
    print("ANALYSIS 2: FREQUENCY SWEEP (0.9 - 5.8 GHz)")
    print("="*60)
    
    import matplotlib.pyplot as plt
    apply_pub_style()
    
    freq_list = [
        0.915e9,   # 915 MHz ISM
        1.8e9,     # GSM 1800
        2.1e9,     # UMTS
        2.45e9,    # WiFi/BT 2.4 GHz
        3.5e9,     # 5G mid-band
        5.8e9      # WiFi 5 GHz
    ]
    
    results = {'freq': [], 'v_dc': [], 'ripple': [], 'p_out': []}
    
    for f in freq_list:
        print(f"  Simulating {f/1e9:.2f} GHz...", end=' ')
        try:
            netlist, t_stop, n_cyc, t_step = build_netlist(freq=f, n_cycles=100)
            data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
            metrics = extract_dc_metrics(data, R_LOAD, v_out_key='vout', v_in_key='rf_in')
            
            results['freq'].append(f)
            results['v_dc'].append(metrics['v_dc'])
            results['ripple'].append(metrics['ripple'])
            results['p_out'].append(metrics['p_out'])
            print(f"V_dc = {metrics['v_dc']*1000:.2f} mV")
        except Exception as e:
            print(f"FAILED: {e}")
    
    for k in results:
        results[k] = np.array(results[k])
    
    # Individual plots - publication style
    freq_ghz = results['freq'] / 1e9
    
    # Plot 1: DC output vs frequency
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(freq_ghz, results['v_dc'] * 1000, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2)
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('DC Output (mV)')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.locator_params(axis='x', nbins=6)
    ax1.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path('halfwave_freq_sweep_dc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 2: Power output vs frequency
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.plot(freq_ghz, results['p_out'] * 1e6, 'k--', lw=2, marker='s', ms=10, mfc='white', mew=2)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Output Power (uW)')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.locator_params(axis='x', nbins=6)
    ax2.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path('halfwave_freq_sweep_power.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    return results


# =============================================================================
# Analysis 3: Sensitivity Analysis
# =============================================================================

def run_sensitivity_analysis():
    """Sensitivity analysis on component variations."""
    print("\n" + "="*60)
    print("ANALYSIS 3: SENSITIVITY ANALYSIS")
    print("="*60)
    
    import matplotlib.pyplot as plt
    apply_pub_style()
    
    # 3a: Sweep input amplitude
    print("\n3a) Sweeping input amplitude...")
    v_amps = np.linspace(0.1, 0.8, 8)
    sens_v = {'values': v_amps, 'v_dc': [], 'p_out': []}
    
    for v in v_amps:
        netlist, t_stop, n_cyc, t_step = build_netlist(v_amp=v)
        data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
        m = extract_dc_metrics(data, R_LOAD)
        sens_v['v_dc'].append(m['v_dc'])
        sens_v['p_out'].append(m['p_out'])
    
    sens_v['v_dc'] = np.array(sens_v['v_dc'])
    sens_v['p_out'] = np.array(sens_v['p_out'])
    
    # 3b: Sweep load resistance
    print("3b) Sweeping load resistance...")
    r_loads = np.array([1e3, 2e3, 5e3, 10e3, 20e3, 50e3, 100e3])
    sens_r = {'values': r_loads, 'v_dc': [], 'p_out': []}
    
    for r in r_loads:
        netlist, t_stop, n_cyc, t_step = build_netlist(r_load=r)
        data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
        m = extract_dc_metrics(data, r)
        sens_r['v_dc'].append(m['v_dc'])
        sens_r['p_out'].append(m['p_out'])
    
    sens_r['v_dc'] = np.array(sens_r['v_dc'])
    sens_r['p_out'] = np.array(sens_r['p_out'])
    
    # 3c: Sweep output capacitor
    print("3c) Sweeping output capacitor...")
    c_outs = np.array([10e-12, 22e-12, 47e-12, 100e-12, 220e-12, 470e-12])
    sens_c = {'values': c_outs, 'v_dc': [], 'ripple': []}
    
    for c in c_outs:
        netlist, t_stop, n_cyc, t_step = build_netlist(c_out=c)
        data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
        m = extract_dc_metrics(data, R_LOAD)
        sens_c['v_dc'].append(m['v_dc'])
        sens_c['ripple'].append(m['ripple'])
    
    sens_c['v_dc'] = np.array(sens_c['v_dc'])
    sens_c['ripple'] = np.array(sens_c['ripple'])
    
    # Individual plots - publication style
    
    # Plot 1: Input amplitude vs DC output
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(sens_v['values'] * 1000, sens_v['v_dc'] * 1000, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2)
    ax1.set_xlabel('Input Amplitude (mV)')
    ax1.set_ylabel('DC Output (mV)')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.locator_params(axis='x', nbins=5)
    ax1.locator_params(axis='y', nbins=5)
    plt.tight_layout()
    save_path = get_save_path('halfwave_sens_input_amplitude.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 2: Load resistance (voltage and power)
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.semilogx(sens_r['values'] / 1e3, sens_r['v_dc'] * 1000, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2, label='DC Voltage')
    ax2.set_xlabel('Load Resistance (kOhm)')
    ax2.set_ylabel('DC Output (mV)')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2_right = ax2.twinx()
    ax2_right.semilogx(sens_r['values'] / 1e3, sens_r['p_out'] * 1e6, 'k--', lw=2, marker='s', ms=10, mfc='white', mew=2, label='Power')
    ax2_right.set_ylabel('Power (uW)')
    for spine in ax2_right.spines.values():
        spine.set_linewidth(2)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_right.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.tight_layout()
    save_path = get_save_path('halfwave_sens_load_resistance.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 3: Output capacitor vs DC
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.semilogx(sens_c['values'] * 1e12, sens_c['v_dc'] * 1000, 'k-', lw=2, marker='o', ms=10, mfc='white', mew=2)
    ax3.set_xlabel('Output Capacitor (pF)')
    ax3.set_ylabel('DC Output (mV)')
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    save_path = get_save_path('halfwave_sens_cout_dc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 4: Output capacitor vs ripple
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    ax4.semilogx(sens_c['values'] * 1e12, sens_c['ripple'] * 1000, 'k--', lw=2, marker='s', ms=10, mfc='white', mew=2)
    ax4.set_xlabel('Output Capacitor (pF)')
    ax4.set_ylabel('Ripple (mV pk-pk)')
    for spine in ax4.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    save_path = get_save_path('halfwave_sens_cout_ripple.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    return sens_v, sens_r, sens_c


def run_monte_carlo(n_runs=50):
    """Monte Carlo with component tolerances."""
    print("\n" + "="*60)
    print("ANALYSIS 3b: MONTE CARLO (Component Tolerances)")
    print("="*60)
    
    import matplotlib.pyplot as plt
    apply_pub_style()
    
    results = {'v_dc': [], 'ripple': [], 'p_out': []}
    
    c_in_nom, c_in_tol = C_IN, 10
    c_out_nom, c_out_tol = C_OUT, 10
    r_load_nom, r_load_tol = R_LOAD, 5
    v_amp_nom, v_amp_tol = V_RF_AMPLITUDE, 3
    
    print(f"Running {n_runs} Monte Carlo iterations...")
    print(f"  C_in:   {c_in_nom*1e12:.0f} pF +/-{c_in_tol}%")
    print(f"  C_out:  {c_out_nom*1e12:.0f} pF +/-{c_out_tol}%")
    print(f"  R_load: {r_load_nom/1e3:.0f} kOhm +/-{r_load_tol}%")
    print(f"  V_amp:  {v_amp_nom*1e3:.0f} mV +/-{v_amp_tol}%")
    
    for i in range(n_runs):
        c_in = c_in_nom * (1 + np.random.uniform(-c_in_tol, c_in_tol) / 100)
        c_out = c_out_nom * (1 + np.random.uniform(-c_out_tol, c_out_tol) / 100)
        r_load = r_load_nom * (1 + np.random.uniform(-r_load_tol, r_load_tol) / 100)
        v_amp = v_amp_nom * (1 + np.random.uniform(-v_amp_tol, v_amp_tol) / 100)
        
        try:
            netlist, t_stop, n_cyc, t_step = build_netlist(
                v_amp=v_amp, c_in=c_in, c_out=c_out, r_load=r_load
            )
            data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
            m = extract_dc_metrics(data, r_load)
            
            results['v_dc'].append(m['v_dc'])
            results['ripple'].append(m['ripple'])
            results['p_out'].append(m['p_out'])
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i+1}/{n_runs}")
        except Exception as e:
            print(f"  Run {i+1} failed: {e}")
    
    for k in results:
        results[k] = np.array(results[k])
    
    stats = {}
    for k in ['v_dc', 'ripple', 'p_out']:
        stats[k] = {
            'mean': np.mean(results[k]),
            'std': np.std(results[k]),
            'min': np.min(results[k]),
            'max': np.max(results[k])
        }
    
    print(f"\nMonte Carlo Results (N={len(results['v_dc'])})")
    print(f"  V_dc:   {stats['v_dc']['mean']*1000:.2f} +/- {stats['v_dc']['std']*1000:.2f} mV")
    print(f"  Ripple: {stats['ripple']['mean']*1000:.2f} +/- {stats['ripple']['std']*1000:.2f} mV")
    print(f"  Power:  {stats['p_out']['mean']*1e6:.2f} +/- {stats['p_out']['std']*1e6:.2f} uW")
    
    # Individual histograms - publication style
    
    # Plot 1: V_dc histogram
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.hist(results['v_dc'] * 1000, bins=15, color='white', edgecolor='black', linewidth=2, hatch='///')
    ax1.axvline(stats['v_dc']['mean'] * 1000, color='black', ls='--', lw=3)
    ax1.set_xlabel('DC Output (mV)')
    ax1.set_ylabel('Count')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    ax1.locator_params(axis='x', nbins=5)
    ax1.text(0.95, 0.95, f"mean = {stats['v_dc']['mean']*1000:.2f} mV\nstd = {stats['v_dc']['std']*1000:.2f} mV",
             transform=ax1.transAxes, fontsize=18, fontweight='bold', ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    plt.tight_layout()
    save_path = get_save_path('halfwave_mc_dc.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 2: Ripple histogram
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.hist(results['ripple'] * 1000, bins=15, color='white', edgecolor='black', linewidth=2, hatch='xxx')
    ax2.axvline(stats['ripple']['mean'] * 1000, color='black', ls='--', lw=3)
    ax2.set_xlabel('Ripple (mV)')
    ax2.set_ylabel('Count')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    ax2.locator_params(axis='x', nbins=5)
    ax2.text(0.95, 0.95, f"mean = {stats['ripple']['mean']*1000:.2f} mV\nstd = {stats['ripple']['std']*1000:.2f} mV",
             transform=ax2.transAxes, fontsize=18, fontweight='bold', ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    plt.tight_layout()
    save_path = get_save_path('halfwave_mc_ripple.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    # Plot 3: Power histogram
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.hist(results['p_out'] * 1e6, bins=15, color='white', edgecolor='black', linewidth=2, hatch='...')
    ax3.axvline(stats['p_out']['mean'] * 1e6, color='black', ls='--', lw=3)
    ax3.set_xlabel('Power (uW)')
    ax3.set_ylabel('Count')
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    ax3.locator_params(axis='x', nbins=5)
    ax3.text(0.95, 0.95, f"mean = {stats['p_out']['mean']*1e6:.2f} uW\nstd = {stats['p_out']['std']*1e6:.2f} uW",
             transform=ax3.transAxes, fontsize=18, fontweight='bold', ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    plt.tight_layout()
    save_path = get_save_path('halfwave_mc_power.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved '{save_path}'")
    plt.show()
    
    return results, stats


# =============================================================================

if __name__ == "__main__":
    print("="*60)
    print("HALF-WAVE RF RECTIFIER - FULL ANALYSIS SUITE")
    print(f"Target: {F_RF/1e9:.2f} GHz | Input: {V_RF_AMPLITUDE*1000:.0f} mV pk")
    print("="*60)
    
    # 1. Transient + Harmonic
    trans_data, harm_data = run_transient_and_harmonic()
    
    # 2. Frequency Sweep
    freq_data = run_frequency_sweep()
    
    # 3. Sensitivity Analysis
    sens_data = run_sensitivity_analysis()
    
    # 4. Monte Carlo
    mc_data, mc_stats = run_monte_carlo(n_runs=50)
    
    print("\n" + "="*60)
    print("ALL ANALYSES COMPLETE")
    print("="*60)
    print("Generated individual plots in temp_image/:")
    print("  Transient:")
    print("    - halfwave_transient_input.png")
    print("    - halfwave_transient_output.png")
    print("  Harmonics:")
    print("    - halfwave_harmonics_spectrum.png")
    print("    - halfwave_harmonics_harmonics.png")
    print("  Frequency Sweep:")
    print("    - halfwave_freq_sweep_dc.png")
    print("    - halfwave_freq_sweep_power.png")
    print("  Sensitivity:")
    print("    - halfwave_sens_input_amplitude.png")
    print("    - halfwave_sens_load_resistance.png")
    print("    - halfwave_sens_cout_dc.png")
    print("    - halfwave_sens_cout_ripple.png")
    print("  Monte Carlo:")
    print("    - halfwave_mc_dc.png")
    print("    - halfwave_mc_ripple.png")
    print("    - halfwave_mc_power.png")
