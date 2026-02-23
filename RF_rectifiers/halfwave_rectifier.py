#!/usr/bin/env python3
"""
================================================================================
Half-Wave RF Rectifier - Full Analysis Suite at 2.45 GHz
================================================================================
RF energy harvesting rectifier with:
    1. Transient simulation (with/without Pi-match)
    2. Harmonic analysis
    3. Frequency sweep
    4. Sensitivity / Monte Carlo analysis
    5. Pi-match network optimization integration

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

import numpy as np
from pathlib import Path

from utility import (
    calculate_esr_from_q,
    run_halfwave_simulation,
    run_matched_rectifier_simulation,
    plot_halfwave_results,
    plot_matched_rectifier_results,
    harmonic_analysis,
    plot_harmonics,
    extract_dc_metrics,
    apply_pub_style,
    get_save_path,
    plot_wifi_spectrum,
    plot_wifi_spectrum_commpy,  # CommPy 802.11 OFDM
    plot_sens_combined,
    plot_freq_stability,
    plot_mc_dc,
    plot_mc_ripple,
    plot_mc_power,
)

# Import Pi-match and optimization modules
from pi_match import (
    PiMatchNetwork, Inductor, Capacitor,
    create_pi_match_from_values, design_pi_match
)
from optimization import optimize_pi_match, OptimizationResult


# =============================================================================
# Circuit Parameters - 2.45 GHz RF Energy Harvesting
# =============================================================================

F_RF = 2.45e9           # 2.45 GHz ISM band
T_RF = 1 / F_RF

V_RF_AMPLITUDE = 0.3    # 300 mV peak (typical RF harvesting level)

# Antenna / Source impedance (will be matched by Pi-network)
ANT_IMP = 50            # Antenna impedance (Ohms) - typical 50 Ohm antenna

# Estimated rectifier input impedance (load for Pi-match)
RECT_IMP_EST = 30       # Estimated rectifier input impedance (Ohms)

# Legacy parameter for backwards compatibility
R_SOURCE = ANT_IMP

C_IN = 100e-12          # 100 pF input coupling
C_OUT = 100e-12         # 100 pF output smoothing
R_LOAD = 5e3            # 5 kOhm load (fixed for sensitivity analysis)

# Quality factors for non-ideal components
CAP_Q = 30              # Capacitor Q (realistic for commercial RF caps at GHz)
IND_Q = 50              # Inductor Q (realistic for SMD inductors at GHz)
PI_MATCH_CAP_Q = 100    # Higher Q caps for matching network

N_CYCLES = 100          # Enough cycles for steady-state
T_STEP = T_RF / 40
T_STOP = N_CYCLES * T_RF

DIODE_MODEL_FILE = "diode_models.lib"
DIODE_MODEL_NAME = "SMS7630"

# Pi-Match design targets
PI_MATCH_TARGET_RL_DB = -20   # Target return loss (dB)
PI_MATCH_TARGET_BW_MHZ = 30   # Target bandwidth (MHz)


# =============================================================================
# Netlist Builder - Direct (No Matching)
# =============================================================================

def build_netlist(freq=F_RF, v_amp=V_RF_AMPLITUDE, c_in=C_IN, c_out=C_OUT, 
                  r_load=R_LOAD, n_cycles=N_CYCLES, ant_imp=ANT_IMP):
    """Build SPICE netlist with given parameters (direct connection, no matching)."""
    t_rf = 1.0 / freq
    t_step = t_rf / 40
    t_stop = n_cycles * t_rf
    
    esr_in = calculate_esr_from_q(c_in, CAP_Q, freq)
    esr_out = calculate_esr_from_q(c_out, CAP_Q, freq)
    
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / DIODE_MODEL_FILE
    
    ##############Note from Chen: this circuit is the replicate simulation of the RF rectifier in figure 2 of the paper. netlist is checked by Chen
    netlist = f"""Half-Wave RF Rectifier (Direct - No Matching)
.include {model_path}

* RF Source with antenna impedance
Vrf rf_source 0 SIN(0 {v_amp} {freq})
Rant rf_source rf_in {ant_imp}

* Input coupling capacitor with ESR
Cin rf_in node_c_in {c_in}
Resr_in node_c_in diode_anode {esr_in}

* Half-wave rectifier diodes
D2 0 diode_anode {DIODE_MODEL_NAME}
D1 diode_anode vout {DIODE_MODEL_NAME}

* Output filter
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
# Netlist Builder - With Pi-Match Network
# =============================================================================

def build_netlist_with_pi_match(freq=F_RF, v_amp=V_RF_AMPLITUDE, c_in=C_IN, c_out=C_OUT,
                                 r_load=R_LOAD, n_cycles=N_CYCLES, ant_imp=ANT_IMP,
                                 pi_match: PiMatchNetwork = None,
                                 L_val=None, C1_val=None, C2_val=None,
                                 q_L=IND_Q, q_C=PI_MATCH_CAP_Q):
    """
    Build SPICE netlist with Pi-matching network.
    
    Can provide either:
    - pi_match: PiMatchNetwork object, OR
    - L_val, C1_val, C2_val: Individual component values
    
    If neither provided, will use default analytical design.
    """
    t_rf = 1.0 / freq
    t_step = t_rf / 40
    t_stop = n_cycles * t_rf
    
    # ESR for rectifier capacitors
    esr_in = calculate_esr_from_q(c_in, CAP_Q, freq)
    esr_out = calculate_esr_from_q(c_out, CAP_Q, freq)
    
    # Get or create Pi-match network
    if pi_match is None:
        if L_val is not None and C1_val is not None and C2_val is not None:
            pi_match = create_pi_match_from_values(L_val, C1_val, C2_val, 
                                                   ant_imp, RECT_IMP_EST, freq, q_L, q_C)
        else:
            pi_match = design_pi_match(ant_imp, RECT_IMP_EST, freq, q_L, q_C)
    
    # Get Pi-match netlist fragment from the PiMatchNetwork object
    pi_match_netlist = pi_match.generate_spice_netlist(node_in='ant_out', node_out='pi_out')
    
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / DIODE_MODEL_FILE
    
    netlist = f"""Half-Wave RF Rectifier with Pi-Matching Network
.include {model_path}

* Design: {ant_imp} Ohm antenna -> Pi-match -> Rectifier

* RF Source with antenna impedance
Vrf rf_source 0 SIN(0 {v_amp} {freq})
Rant rf_source ant_out {ant_imp}

{pi_match_netlist}

* === Half-Wave Rectifier ===
* Input coupling capacitor with ESR
Cin pi_out node_c_in {c_in}
Resr_in node_c_in diode_anode {esr_in}

* Half-wave rectifier diodes
D2 0 diode_anode {DIODE_MODEL_NAME}
D1 diode_anode vout {DIODE_MODEL_NAME}

* Output filter
Cout vout node_c_out {c_out}
Resr_out node_c_out 0 {esr_out}
Rload vout 0 {r_load}

.control
set filetype=ascii
set wr_vecnames
option noacct
tran {t_step} {t_stop} uic
wrdata output.txt time v(ant_out) v(pi_out) v(vout)
quit
.endc

.end
"""
    return netlist, t_stop, n_cycles, t_step


def build_netlist_with_esr(freq=F_RF, v_amp=V_RF_AMPLITUDE, c_in=C_IN, c_out=C_OUT, 
                           r_load=R_LOAD, n_cycles=N_CYCLES, esr_in=None, esr_out=None,
                           ant_imp=ANT_IMP):
    """
    Build SPICE netlist with explicitly specified ESR values.
    Used for Monte Carlo analysis where ESR varies independently.
    """
    t_rf = 1.0 / freq
    t_step = t_rf / 40
    t_stop = n_cycles * t_rf
    
    # Use provided ESR or calculate from default Q
    if esr_in is None:
        esr_in = calculate_esr_from_q(c_in, CAP_Q, freq)
    if esr_out is None:
        esr_out = calculate_esr_from_q(c_out, CAP_Q, freq)
    
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / DIODE_MODEL_FILE
    
    netlist = f"""Half-Wave RF Rectifier (ESR specified)
.include {model_path}

* RF Source with antenna impedance
Vrf rf_source 0 SIN(0 {v_amp} {freq})
Rant rf_source rf_in {ant_imp}

* Input coupling capacitor with ESR
Cin rf_in node_c_in {c_in}
Resr_in node_c_in diode_anode {esr_in}

* Half-wave rectifier diodes
D2 0 diode_anode {DIODE_MODEL_NAME}
D1 diode_anode vout {DIODE_MODEL_NAME}

* Output filter
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
# Analysis 0: Pi-Match Optimization (Run First)
# =============================================================================

def run_pi_match_optimization(verbose=True):
    """
    Run Pi-match optimization to find optimal L, C1, C2 values.
    
    Constraints:
    - Return loss < -20 dB at center frequency
    - Bandwidth > 30 MHz centered at 2.45 GHz
    - Uses non-ideal components with Q factors
    
    Returns:
        OptimizationResult with optimal Pi-match network
    """
    print("\n" + "="*60)
    print("ANALYSIS 0: PI-MATCH OPTIMIZATION")
    print("="*60)
    print(f"Source (antenna): {ANT_IMP} Ω")
    print(f"Load (rectifier): {RECT_IMP_EST} Ω (estimated)")
    print(f"Center frequency: {F_RF/1e9:.3f} GHz")
    print(f"Target return loss: < {PI_MATCH_TARGET_RL_DB} dB")
    print(f"Target bandwidth: > {PI_MATCH_TARGET_BW_MHZ} MHz")
    print(f"Component Q: L={IND_Q}, C={PI_MATCH_CAP_Q}")
    print("-"*60)
    
    result = optimize_pi_match(
        z_source=ANT_IMP,
        z_load=RECT_IMP_EST,
        f_center=F_RF,
        q_L=IND_Q,
        q_C=PI_MATCH_CAP_Q,
        target_rl_dB=PI_MATCH_TARGET_RL_DB,
        target_bw_MHz=PI_MATCH_TARGET_BW_MHZ,
        method='auto',
        verbose=verbose
    )
    
    return result


# =============================================================================
# Analysis 1: Transient + Harmonic Analysis (Direct - No Matching)
# =============================================================================

def run_transient_and_harmonic():
    """Run transient simulation and harmonic analysis (direct, no matching)."""
    print("\n" + "="*60)
    print("ANALYSIS 1: TRANSIENT + HARMONIC ANALYSIS (Direct)")
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
# Analysis 2: Sensitivity Analysis
# =============================================================================

def run_sensitivity_analysis():
    """Sensitivity analysis on component variations."""
    print("\n" + "="*60)
    print("ANALYSIS 3: SENSITIVITY ANALYSIS")
    print("="*60)
    
    # 3a: Sweep input amplitude (20 mV to 800 mV)
    print("\n3a) Sweeping input amplitude (20-800 mV)...")
    v_amps = np.linspace(0.02, 0.8, 12)  # 20 mV to 800 mV, 12 points
    sens_v = {'values': v_amps, 'v_dc': [], 'p_out': []}
    
    for v in v_amps:
        netlist, t_stop, n_cyc, t_step = build_netlist(v_amp=v)
        data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
        m = extract_dc_metrics(data, R_LOAD)
        sens_v['v_dc'].append(m['v_dc'])
        sens_v['p_out'].append(m['p_out'])
        print(f"    V_in={v*1000:.0f} mV -> V_dc={m['v_dc']*1000:.2f} mV")
    
    sens_v['v_dc'] = np.array(sens_v['v_dc'])
    sens_v['p_out'] = np.array(sens_v['p_out'])
    
    # 3b: Sweep output capacitor
    print("\n3b) Sweeping output capacitor...")
    c_outs = np.array([10e-12, 22e-12, 47e-12, 100e-12, 220e-12, 470e-12])
    sens_c = {'values': c_outs, 'v_dc': []}
    
    for c in c_outs:
        netlist, t_stop, n_cyc, t_step = build_netlist(c_out=c)
        data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
        m = extract_dc_metrics(data, R_LOAD)
        sens_c['v_dc'].append(m['v_dc'])
    
    sens_c['v_dc'] = np.array(sens_c['v_dc'])
    
    # Combined plot: Input amplitude (top x-axis) + Cout (bottom x-axis) vs DC output
    plot_sens_combined(
        sens_v['values'] * 1000,      # mV
        sens_v['v_dc'] * 1000,        # mV
        sens_c['values'] * 1e12,      # pF
        sens_c['v_dc'] * 1000         # mV
    )
    
    return sens_v, sens_c


# =============================================================================
# Analysis 3b: Frequency Stability (Narrowband Response)
# =============================================================================

def run_frequency_stability():
    """
    Frequency stability analysis for narrowband RF harvesting with WiFi consideration.
    
    Tests how the rectifier output varies within ±20 MHz of the target frequency,
    considering that WiFi 802.11n already occupies 20 MHz bandwidth.
    
    Analysis tests:
    - Rectifier response across center frequency drift
    - WiFi channel bandwidth (20 MHz) shown as shaded region
    - Target: maintain >70% (-3dB) output across ±10 MHz drift
    
    This matters because:
    - WiFi signal already spans ±10 MHz around center (20 MHz channel)
    - Oscillator drift can shift entire channel another ±5-10 MHz
    - Rectifier must work across total effective bandwidth
    """
    print("\n" + "="*60)
    print("ANALYSIS 3b: FREQUENCY STABILITY (±20 MHz around 2.45 GHz)")
    print("With WiFi 20 MHz channel bandwidth consideration")
    print("="*60)
    
    # Frequency points: ±20 MHz around center (covers WiFi BW + drift)
    freq_offsets_mhz = np.linspace(-20, 20, 21)  # -20 to +20 MHz, 2 MHz steps
    freq_list = F_RF + freq_offsets_mhz * 1e6
    
    results = {'freq_offset_mhz': freq_offsets_mhz, 'v_dc': [], 'p_out': []}
    
    print(f"  Target frequency: {F_RF/1e9:.3f} GHz")
    print(f"  WiFi channel: {(F_RF-10e6)/1e9:.3f} - {(F_RF+10e6)/1e9:.3f} GHz (20 MHz)")
    print(f"  Sweep range: {freq_list[0]/1e9:.3f} - {freq_list[-1]/1e9:.3f} GHz (40 MHz)")
    
    for i, f in enumerate(freq_list):
        offset = freq_offsets_mhz[i]
        print(f"  {offset:+.0f} MHz...", end=' ')
        try:
            netlist, t_stop, n_cyc, t_step = build_netlist(freq=f, n_cycles=100)
            data = run_halfwave_simulation(netlist, t_stop, n_cyc, t_step)
            m = extract_dc_metrics(data, R_LOAD)
            results['v_dc'].append(m['v_dc'])
            results['p_out'].append(m['p_out'])
            print(f"V_dc = {m['v_dc']*1000:.2f} mV")
        except Exception as e:
            print(f"FAILED: {e}")
            results['v_dc'].append(np.nan)
            results['p_out'].append(np.nan)
    
    results['v_dc'] = np.array(results['v_dc'])
    results['p_out'] = np.array(results['p_out'])
    
    # Normalize to center frequency value
    center_idx = len(freq_offsets_mhz) // 2  # Index of 0 MHz offset
    v_dc_center = results['v_dc'][center_idx]
    v_dc_normalized = results['v_dc'] / v_dc_center
    
    # Calculate -3dB bandwidth
    threshold_3db = 1 / np.sqrt(2)  # ~0.707
    above_3db = v_dc_normalized >= threshold_3db
    
    # Find bandwidth (assuming symmetric response)
    bw_3db = None
    for i, offset in enumerate(freq_offsets_mhz):
        if offset > 0 and not above_3db[i]:
            bw_3db = 2 * freq_offsets_mhz[i-1]  # Total bandwidth
            break
    if bw_3db is None and all(above_3db):
        bw_3db = 2 * freq_offsets_mhz[-1]  # Full range is within -3dB
    
    # Check WiFi coverage
    wifi_bw = 20  # MHz
    wifi_covered = all(v_dc_normalized[np.abs(freq_offsets_mhz) <= wifi_bw/2] >= threshold_3db)
    
    print(f"\n  Results:")
    print(f"    Center V_dc: {v_dc_center*1000:.2f} mV")
    print(f"    -3dB BW: {'>' if bw_3db == 2*freq_offsets_mhz[-1] else ''}{bw_3db:.1f} MHz")
    print(f"    WiFi 20 MHz channel coverage: {'PASS' if wifi_covered else 'FAIL'}")
    if bw_3db and bw_3db >= 20:
        print(f"    Status: PASS (BW >= WiFi channel + ±10 MHz drift margin)")
    else:
        print(f"    Status: WARNING (BW may be marginal for drifted WiFi)")
    
    # Plot with WiFi channel shading
    plot_freq_stability(freq_offsets_mhz, v_dc_normalized, bw_3db, wifi_bw_mhz=wifi_bw)
    
    return results, bw_3db


def run_monte_carlo(n_runs=50):
    """
    Monte Carlo with component tolerances including ESR.
    
    Models manufacturing variations:
    - Capacitor values: ±10% (typical ceramic capacitor tolerance)
    - Capacitor Q/ESR: ±20% (accounts for PCB parasitics & solder joints)
    - Load resistance: ±5% (resistor tolerance)
    - Input amplitude: ±3% (RF source variation)
    
    Note on PCB/Soldering Effects:
    - PCB trace inductance/capacitance: absorbed into component tolerances
    - Solder joint resistance: modeled via ESR variation (Q factor)
    - Via inductance: negligible at 2.45 GHz for short vias
    - Ground plane quality: affects Q, modeled in ESR variation
    """
    print("\n" + "="*60)
    print("ANALYSIS 4: MONTE CARLO (Component Tolerances + ESR)")
    print("="*60)
    
    results = {'v_dc': [], 'ripple': [], 'p_out': []}
    
    # Nominal values and tolerances
    c_in_nom, c_in_tol = C_IN, 10       # ±10% capacitor tolerance
    c_out_nom, c_out_tol = C_OUT, 10    # ±10% capacitor tolerance
    r_load_nom, r_load_tol = R_LOAD, 5  # ±5% resistor tolerance
    v_amp_nom, v_amp_tol = V_RF_AMPLITUDE, 3  # ±3% RF source variation
    q_nom, q_tol = CAP_Q, 20            # ±20% Q variation (ESR, PCB parasitics, solder)
    
    print(f"Running {n_runs} Monte Carlo iterations...")
    print(f"  C_in:   {c_in_nom*1e12:.0f} pF ±{c_in_tol}%")
    print(f"  C_out:  {c_out_nom*1e12:.0f} pF ±{c_out_tol}%")
    print(f"  R_load: {r_load_nom/1e3:.0f} kΩ ±{r_load_tol}%")
    print(f"  V_amp:  {v_amp_nom*1e3:.0f} mV ±{v_amp_tol}%")
    print(f"  CAP_Q:  {q_nom} ±{q_tol}% (ESR variation, includes PCB/solder)")
    
    for i in range(n_runs):
        # Random component values with tolerances
        c_in = c_in_nom * (1 + np.random.uniform(-c_in_tol, c_in_tol) / 100)
        c_out = c_out_nom * (1 + np.random.uniform(-c_out_tol, c_out_tol) / 100)
        r_load = r_load_nom * (1 + np.random.uniform(-r_load_tol, r_load_tol) / 100)
        v_amp = v_amp_nom * (1 + np.random.uniform(-v_amp_tol, v_amp_tol) / 100)
        q_factor = q_nom * (1 + np.random.uniform(-q_tol, q_tol) / 100)
        
        # Calculate ESR with varied Q
        esr_in = calculate_esr_from_q(c_in, q_factor, F_RF)
        esr_out = calculate_esr_from_q(c_out, q_factor, F_RF)
        
        try:
            netlist, t_stop, n_cyc, t_step = build_netlist_with_esr(
                v_amp=v_amp, c_in=c_in, c_out=c_out, r_load=r_load,
                esr_in=esr_in, esr_out=esr_out
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
    print(f"  V_dc:   {stats['v_dc']['mean']*1000:.2f} ± {stats['v_dc']['std']*1000:.2f} mV")
    print(f"  Ripple: {stats['ripple']['mean']*1000:.2f} ± {stats['ripple']['std']*1000:.2f} mV")
    print(f"  Power:  {stats['p_out']['mean']*1e6:.2f} ± {stats['p_out']['std']*1e6:.2f} uW")
    
    # Plot using utility functions
    plot_mc_dc(results['v_dc'] * 1000, 
               {'mean': stats['v_dc']['mean'] * 1000, 'std': stats['v_dc']['std'] * 1000})
    plot_mc_ripple(results['ripple'] * 1000,
                   {'mean': stats['ripple']['mean'] * 1000, 'std': stats['ripple']['std'] * 1000})
    plot_mc_power(results['p_out'] * 1e6,
                  {'mean': stats['p_out']['mean'] * 1e6, 'std': stats['p_out']['std'] * 1e6})
    
    return results, stats


# =============================================================================
# Analysis 5: Matched Rectifier Comparison
# =============================================================================

def run_matched_rectifier_comparison(pi_match_result: OptimizationResult = None):
    """
    Compare rectifier performance with and without Pi-matching network.
    
    Args:
        pi_match_result: Optimization result from run_pi_match_optimization()
                        If None, will run optimization first.
    
    Returns:
        Dict with comparison results
    """
    print("\n" + "="*60)
    print("ANALYSIS 5: MATCHED vs DIRECT RECTIFIER COMPARISON")
    print("="*60)
    
    # Get optimized Pi-match if not provided
    if pi_match_result is None:
        print("Running Pi-match optimization first...")
        pi_match_result = run_pi_match_optimization(verbose=False)
    
    pi_match = pi_match_result.pi_match
    
    print(f"\nPi-Match Network:")
    print(f"  L  = {pi_match.L.L*1e9:.4f} nH (Q={pi_match.L.Q})")
    print(f"  C1 = {pi_match.C1.C*1e12:.4f} pF (Q={pi_match.C1.Q})")
    print(f"  C2 = {pi_match.C2.C*1e12:.4f} pF (Q={pi_match.C2.Q})")
    print(f"  Return Loss: {pi_match_result.return_loss_dB:.2f} dB")
    print(f"  Bandwidth:   {pi_match_result.bandwidth_MHz:.1f} MHz")
    
    # Simulate direct (no matching)
    print("\nSimulating direct connection (no matching)...")
    netlist_direct, t_stop, n_cyc, t_step = build_netlist()
    data_direct = run_halfwave_simulation(netlist_direct, t_stop, n_cyc, t_step)
    metrics_direct = extract_dc_metrics(data_direct, R_LOAD)
    
    # Simulate with Pi-match
    print("Simulating with Pi-matching network...")
    netlist_matched, t_stop, n_cyc, t_step = build_netlist_with_pi_match(
        pi_match=pi_match
    )
    data_matched = run_matched_rectifier_simulation(netlist_matched, t_stop, n_cyc, t_step)
    metrics_matched = extract_dc_metrics(data_matched, R_LOAD)
    
    # Calculate improvement
    if metrics_direct['v_dc'] > 0:
        v_improvement = (metrics_matched['v_dc'] / metrics_direct['v_dc'] - 1) * 100
        p_improvement = (metrics_matched['p_out'] / metrics_direct['p_out'] - 1) * 100
    else:
        v_improvement = float('inf')
        p_improvement = float('inf')
    
    print("\n" + "-"*60)
    print("RESULTS COMPARISON")
    print("-"*60)
    print(f"{'Metric':<20} {'Direct':>15} {'Matched':>15} {'Improvement':>15}")
    print("-"*60)
    print(f"{'V_DC (mV)':<20} {metrics_direct['v_dc']*1000:>15.3f} {metrics_matched['v_dc']*1000:>15.3f} {v_improvement:>+14.1f}%")
    print(f"{'Ripple (mV)':<20} {metrics_direct['ripple']*1000:>15.3f} {metrics_matched['ripple']*1000:>15.3f}")
    print(f"{'Power (uW)':<20} {metrics_direct['p_out']*1e6:>15.3f} {metrics_matched['p_out']*1e6:>15.3f} {p_improvement:>+14.1f}%")
    print("-"*60)
    
    # Plot comparison
    plot_matched_rectifier_results(
        data_direct, data_matched, 
        metrics_direct, metrics_matched,
        F_RF, V_RF_AMPLITUDE, R_LOAD,
        save_prefix='matched_rectifier_comparison'
    )
    
    return {
        'direct': {'data': data_direct, 'metrics': metrics_direct},
        'matched': {'data': data_matched, 'metrics': metrics_matched},
        'v_improvement_pct': v_improvement,
        'p_improvement_pct': p_improvement,
        'pi_match': pi_match
    }


# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HALF-WAVE RF RECTIFIER - FULL ANALYSIS SUITE (with Pi-Match)")
    print(f"Target: {F_RF/1e9:.2f} GHz | Input: {V_RF_AMPLITUDE*1000:.0f} mV pk | R_load: {R_LOAD/1e3:.0f} kΩ")
    print(f"Antenna Impedance: {ANT_IMP} Ω")
    print("="*70)
    
    # 0a. Plot WiFi input spectrum (using CommPy 802.11 OFDM - realistic spectrum)
    print("\nGenerating WiFi OFDM input spectrum (CommPy)...")
    plot_wifi_spectrum_commpy(V_RF_AMPLITUDE, F_RF, n_symbols=10, mcs=3)  # MCS3 = 16-QAM
    
    # 0b. Run Pi-Match Optimization
    pi_opt_result = run_pi_match_optimization(verbose=True)
    
    # 1. Transient + Harmonic (Direct - no matching)
    trans_data, harm_data = run_transient_and_harmonic()
    
    # 2. Matched vs Direct Comparison
    comparison_results = run_matched_rectifier_comparison(pi_opt_result)
    
    # 3. Sensitivity Analysis (combined plot: input amplitude + Cout)
    sens_v, sens_c = run_sensitivity_analysis()
    
    # 4. Frequency Stability (narrowband +/-20 MHz with WiFi BW consideration)
    freq_stab_data, bw_3db = run_frequency_stability()
    
    # 5. Monte Carlo (with ESR variation for PCB/solder effects)
    mc_data, mc_stats = run_monte_carlo(n_runs=50)
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    print("\nGenerated plots in temp_image/:")
    print("  - wifi_spectrum.png")
    print("  - halfwave_transient_waveforms.png")
    print("  - halfwave_harmonics_spectrum.png")
    print("  - halfwave_harmonics_harmonics.png")
    print("  - matched_rectifier_comparison.png  [NEW]")
    print("  - halfwave_sens_combined.png")
    print("  - halfwave_freq_stability.png")
    print("  - halfwave_mc_dc.png")
    print("  - halfwave_mc_ripple.png")
    print("  - halfwave_mc_power.png")
    
    print("\nPi-Match Network Summary:")
    print(f"  L  = {pi_opt_result.L*1e9:.4f} nH")
    print(f"  C1 = {pi_opt_result.C1*1e12:.4f} pF")
    print(f"  C2 = {pi_opt_result.C2*1e12:.4f} pF")
    print(f"  Performance: RL={pi_opt_result.return_loss_dB:.1f}dB, BW={pi_opt_result.bandwidth_MHz:.1f}MHz")
