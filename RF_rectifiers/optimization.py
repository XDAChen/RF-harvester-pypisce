#!/usr/bin/env python3
"""
================================================================================
Optimization Utilities for RF Energy Harvesting
================================================================================
Modern optimization algorithms for component value optimization.

Features:
    - Pi-match network optimization (L, C1, C2 values)
    - Multi-objective optimization (return loss, bandwidth, insertion loss)
    - Support for non-ideal component constraints
    - Multiple optimization backends (scipy, optuna)

Constraints:
    - Return loss < -20 dB (|S11| < 0.1)
    - Bandwidth >= 30 MHz centered at 2.45 GHz
    - Non-ideal component Q factors

Author: RF Energy Harvesting Team
Date: 2026
================================================================================
"""

import numpy as np
from typing import Tuple, Dict, Callable, Optional, List
from dataclasses import dataclass, field
import warnings
from pathlib import Path

# Scipy optimization
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.optimize import NonlinearConstraint, Bounds

# Import local pi_match module
from pi_match import (
    PiMatchNetwork, Inductor, Capacitor,
    create_pi_match_from_values, design_pi_match
)

# Import utility functions for SPICE simulation
from utility import (
    run_transient_ngspice,
    extract_dc_metrics,
    calculate_esr_from_q
)


# =============================================================================
# Optimization Result Container
# =============================================================================

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    success: bool
    L: float              # Optimal inductance (H)
    C1: float             # Optimal input capacitor (F)
    C2: float             # Optimal output capacitor (F)
    return_loss_dB: float # Return loss at center frequency
    bandwidth_MHz: float  # Achieved bandwidth
    insertion_loss_dB: float  # Insertion loss at center
    n_iterations: int     # Number of iterations
    n_function_evals: int # Number of function evaluations
    message: str          # Optimization message
    pi_match: PiMatchNetwork  # Resulting network object
    
    def summary(self) -> str:
        """Return formatted summary string."""
        return (
            f"Optimization Result:\n"
            f"  Success: {self.success}\n"
            f"  L  = {self.L*1e9:.4f} nH\n"
            f"  C1 = {self.C1*1e12:.4f} pF\n"
            f"  C2 = {self.C2*1e12:.4f} pF\n"
            f"  Return Loss: {self.return_loss_dB:.2f} dB\n"
            f"  Bandwidth:   {self.bandwidth_MHz:.1f} MHz\n"
            f"  Insertion Loss: {self.insertion_loss_dB:.2f} dB\n"
            f"  Iterations: {self.n_iterations}, Evals: {self.n_function_evals}\n"
            f"  Message: {self.message}"
        )


# =============================================================================
# Power Conversion Utilities
# =============================================================================

def dBm_to_watts(p_dBm: float) -> float:
    """Convert power from dBm to Watts."""
    return 10 ** ((p_dBm - 30) / 10)

def watts_to_dBm(p_watts: float) -> float:
    """Convert power from Watts to dBm."""
    return 10 * np.log10(max(p_watts, 1e-15)) + 30

def dBm_to_vpeak(p_dBm: float, r_source: float = 50.0) -> float:
    """
    Convert available power (dBm) to peak voltage amplitude.
    
    For a sinusoidal source with series resistance R_source:
    P_avail = V_peak^2 / (8 * R_source)
    
    Args:
        p_dBm: Available power in dBm
        r_source: Source resistance (Ohms), default 50
    
    Returns:
        Peak voltage amplitude (V)
    """
    p_watts = dBm_to_watts(p_dBm)
    return np.sqrt(8 * r_source * p_watts)

def vpeak_to_dBm(v_peak: float, r_source: float = 50.0) -> float:
    """
    Convert peak voltage amplitude to available power (dBm).
    
    Args:
        v_peak: Peak voltage amplitude (V)
        r_source: Source resistance (Ohms), default 50
    
    Returns:
        Available power in dBm
    """
    p_watts = v_peak**2 / (8 * r_source)
    return watts_to_dBm(p_watts)


@dataclass
class LoadPullResult:
    """
    Container for load-pull optimization results.
    
    This optimization maximizes DC output power by sweeping Pi-match
    component values and running full SPICE simulations with diode model.
    """
    success: bool
    L: float                    # Optimal inductance (H)
    C1: float                   # Optimal input capacitor (F)
    C2: float                   # Optimal output capacitor (F)
    max_power_mW: float         # Maximum DC output power (mW)
    v_dc_at_max: float          # DC voltage at max power (V)
    ripple_at_max: float        # Ripple at max power (V)
    efficiency_percent: float   # Power conversion efficiency (%)
    n_simulations: int          # Number of SPICE simulations run
    input_power_dBm: float = 0  # Input power (dBm) this was optimized for
    all_results: List[Dict] = field(default_factory=list)  # All sweep results
    pi_match: PiMatchNetwork = None  # Optimal network object
    
    def summary(self) -> str:
        """Return formatted summary string."""
        p_out_dBm = watts_to_dBm(self.max_power_mW / 1000) if self.max_power_mW > 0 else -99
        return (
            f"Load-Pull Optimization Result:\n"
            f"  Success: {self.success}\n"
            f"  Input Power: {self.input_power_dBm:.1f} dBm\n"
            f"  Optimal Pi-Match:\n"
            f"    L  = {self.L*1e9:.4f} nH\n"
            f"    C1 = {self.C1*1e12:.4f} pF\n"
            f"    C2 = {self.C2*1e12:.4f} pF\n"
            f"  Performance at Max Power:\n"
            f"    P_out = {self.max_power_mW*1000:.2f} µW ({p_out_dBm:.2f} dBm)\n"
            f"    V_DC  = {self.v_dc_at_max*1000:.2f} mV\n"
            f"    Ripple = {self.ripple_at_max*1000:.2f} mV\n"
            f"    Efficiency = {self.efficiency_percent:.2f}%\n"
            f"  Simulations: {self.n_simulations}"
        )


@dataclass
class LoadPullSweepResult:
    """
    Container for full load-pull characterization across multiple input power levels.
    
    This is the proper load-pull characterization that shows how the optimal
    matching component values vary with input power level.
    """
    success: bool
    power_levels_dBm: List[float]       # Input power levels swept (dBm)
    results_per_power: List[LoadPullResult]  # LoadPullResult for each power
    optimal_L_vs_power: List[float]     # Optimal L at each power
    optimal_C1_vs_power: List[float]    # Optimal C1 at each power
    optimal_C2_vs_power: List[float]    # Optimal C2 at each power
    max_power_vs_input: List[float]     # Max P_out (W) at each input level
    efficiency_vs_input: List[float]    # Efficiency (%) at each input level
    total_simulations: int
    
    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "="*70,
            "LOAD-PULL SWEEP CHARACTERIZATION",
            "="*70,
            f"  Power levels: {len(self.power_levels_dBm)}",
            f"  Range: {min(self.power_levels_dBm):.0f} to {max(self.power_levels_dBm):.0f} dBm",
            f"  Total simulations: {self.total_simulations}",
            "-"*70,
            f"{'P_in (dBm)':<12} {'L (nH)':<10} {'C1 (pF)':<10} {'C2 (pF)':<10} {'P_out (µW)':<12} {'Eff (%)':<10}",
            "-"*70,
        ]
        for i, p in enumerate(self.power_levels_dBm):
            L = self.optimal_L_vs_power[i] * 1e9
            C1 = self.optimal_C1_vs_power[i] * 1e12
            C2 = self.optimal_C2_vs_power[i] * 1e12
            P_out = self.max_power_vs_input[i] * 1e6  # W to µW
            eff = self.efficiency_vs_input[i]
            lines.append(f"{p:<12.0f} {L:<10.3f} {C1:<10.3f} {C2:<10.3f} {P_out:<12.3f} {eff:<10.2f}")
        lines.append("="*70)
        return "\n".join(lines)


# =============================================================================
# Objective Functions for Pi-Match Optimization
# =============================================================================

def pi_match_objective(x: np.ndarray, 
                       z_source: float, z_load: float, f_center: float,
                       q_L: float, q_C: float,
                       target_rl_dB: float = -20.0,
                       target_bw_MHz: float = 30.0,
                       weight_rl: float = 1.0,
                       weight_bw: float = 1.0,
                       weight_il: float = 0.5) -> float:
    """
    Objective function for Pi-match optimization.
    
    Minimizes a weighted combination of:
    - Deviation from target return loss
    - Deviation from target bandwidth (penalize if below target)
    - Insertion loss (minimize)
    
    Args:
        x: [L, C1, C2] in Henries and Farads
        z_source: Source impedance
        z_load: Load impedance
        f_center: Center frequency
        q_L: Inductor Q factor
        q_C: Capacitor Q factor
        target_rl_dB: Target return loss (dB, negative)
        target_bw_MHz: Target bandwidth (MHz)
        weight_rl: Weight for return loss term
        weight_bw: Weight for bandwidth term
        weight_il: Weight for insertion loss term
    
    Returns:
        Scalar objective value (lower is better)
    """
    L, C1, C2 = x
    
    # Sanity checks
    if L <= 0 or C1 <= 0 or C2 <= 0:
        return 1e10
    
    try:
        # Create network
        pi = create_pi_match_from_values(L, C1, C2, z_source, z_load, 
                                          f_center, q_L, q_C)
        
        # Return loss at center
        rl_dB = pi.return_loss_db(f_center)
        
        # Insertion loss at center
        il_dB = pi.insertion_loss_db(f_center)
        
        # Bandwidth analysis
        bw_result = pi.analyze_bandwidth(
            f_start=f_center - 50e6,
            f_stop=f_center + 50e6,
            return_loss_threshold=-10.0  # -10 dB for bandwidth calc
        )
        bw_MHz = bw_result['bandwidth_hz'] / 1e6
        
        # Objective components
        
        # 1. Return loss: penalize if worse than target
        # We want rl_dB < target_rl_dB (more negative)
        if rl_dB > target_rl_dB:
            # Not meeting spec - heavy penalty
            rl_penalty = (rl_dB - target_rl_dB)**2 * 10
        else:
            # Meeting spec - small reward for being better
            rl_penalty = (rl_dB - target_rl_dB) * 0.1
        
        # 2. Bandwidth: penalize if below target
        if bw_MHz < target_bw_MHz:
            # Not meeting spec
            bw_penalty = (target_bw_MHz - bw_MHz)**2
        else:
            # Meeting spec - no penalty
            bw_penalty = 0
        
        # 3. Insertion loss: always minimize (less negative = less loss)
        # il_dB is negative, closer to 0 is better
        il_penalty = -il_dB  # Make positive, smaller is better
        
        # Total objective
        obj = (weight_rl * rl_penalty + 
               weight_bw * bw_penalty + 
               weight_il * il_penalty)
        
        return obj
        
    except Exception as e:
        return 1e10


def pi_match_objective_with_constraints(x: np.ndarray,
                                         z_source: float, z_load: float, 
                                         f_center: float,
                                         q_L: float, q_C: float) -> Tuple[float, float, float]:
    """
    Return individual metrics for constraint-based optimization.
    
    Returns:
        (return_loss_dB, bandwidth_MHz, insertion_loss_dB)
    """
    L, C1, C2 = x
    
    if L <= 0 or C1 <= 0 or C2 <= 0:
        return (0, 0, -100)  # Invalid
    
    try:
        pi = create_pi_match_from_values(L, C1, C2, z_source, z_load,
                                          f_center, q_L, q_C)
        
        rl_dB = pi.return_loss_db(f_center)
        il_dB = pi.insertion_loss_db(f_center)
        
        bw_result = pi.analyze_bandwidth(
            f_start=f_center - 50e6,
            f_stop=f_center + 50e6,
            return_loss_threshold=-10.0
        )
        bw_MHz = bw_result['bandwidth_hz'] / 1e6
        
        return (rl_dB, bw_MHz, il_dB)
        
    except:
        return (0, 0, -100)


# =============================================================================
# Optimization Algorithms
# =============================================================================

def optimize_pi_match_scipy(z_source: float = 50.0,
                            z_load: float = 30.0,
                            f_center: float = 2.45e9,
                            q_L: float = 50.0,
                            q_C: float = 100.0,
                            target_rl_dB: float = -20.0,
                            target_bw_MHz: float = 30.0,
                            method: str = 'differential_evolution',
                            verbose: bool = True) -> OptimizationResult:
    """
    Optimize Pi-match network using scipy optimization.
    
    Args:
        z_source: Source impedance (antenna), Ohms
        z_load: Load impedance (rectifier), Ohms
        f_center: Center frequency, Hz
        q_L: Inductor Q factor
        q_C: Capacitor Q factor
        target_rl_dB: Target return loss (must be achieved)
        target_bw_MHz: Target bandwidth (must be achieved)
        method: 'differential_evolution', 'dual_annealing', or 'SLSQP'
        verbose: Print progress
    
    Returns:
        OptimizationResult with optimal component values
    """
    if verbose:
        print("="*60)
        print(f"Pi-Match Optimization (scipy.{method})")
        print("="*60)
        print(f"  Source Z: {z_source} Ω")
        print(f"  Load Z:   {z_load} Ω")
        print(f"  Center:   {f_center/1e9:.3f} GHz")
        print(f"  Target RL: < {target_rl_dB} dB")
        print(f"  Target BW: > {target_bw_MHz} MHz")
        print(f"  Component Q: L={q_L}, C={q_C}")
        print("-"*60)
    
    # Get initial guess from analytical design
    init_pi = design_pi_match(z_source, z_load, f_center, q_L, q_C)
    x0 = np.array([init_pi.L.L, init_pi.C1.C, init_pi.C2.C])
    
    if verbose:
        print(f"Initial guess from analytical design:")
        print(f"  L  = {x0[0]*1e9:.4f} nH")
        print(f"  C1 = {x0[1]*1e12:.4f} pF")
        print(f"  C2 = {x0[2]*1e12:.4f} pF")
    
    # Bounds: reasonable range for 2.45 GHz components
    # L: 0.1 nH to 50 nH
    # C: 0.1 pF to 100 pF
    bounds = Bounds(
        lb=[0.1e-9, 0.1e-12, 0.1e-12],
        ub=[50e-9, 100e-12, 100e-12]
    )
    
    # Objective function wrapper
    def obj_func(x):
        return pi_match_objective(
            x, z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz,
            weight_rl=2.0, weight_bw=1.5, weight_il=0.5
        )
    
    # Run optimization
    n_evals = [0]  # Mutable counter
    
    def obj_with_count(x):
        n_evals[0] += 1
        return obj_func(x)
    
    if method == 'differential_evolution':
        # Global optimization - good for finding global minimum
        result = differential_evolution(
            obj_with_count,
            bounds=[(0.1e-9, 50e-9), (0.1e-12, 100e-12), (0.1e-12, 100e-12)],
            x0=x0,
            maxiter=500,
            tol=1e-8,
            seed=42,
            workers=1,  # Set to -1 for parallel
            updating='deferred',
            polish=True,  # Local refinement at end
            disp=verbose
        )
        n_iter = result.nit
        
    elif method == 'dual_annealing':
        # Simulated annealing variant - good for complex landscapes
        result = dual_annealing(
            obj_with_count,
            bounds=[(0.1e-9, 50e-9), (0.1e-12, 100e-12), (0.1e-12, 100e-12)],
            x0=x0,
            maxiter=1000,
            seed=42,
            no_local_search=False
        )
        n_iter = result.nit
        
    elif method == 'SLSQP':
        # Local gradient-based - fast but may find local minimum
        result = minimize(
            obj_with_count,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'disp': verbose}
        )
        n_iter = result.nit if hasattr(result, 'nit') else 0
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Extract optimal values
    L_opt, C1_opt, C2_opt = result.x
    
    # Create final network and evaluate
    pi_opt = create_pi_match_from_values(
        L_opt, C1_opt, C2_opt,
        z_source, z_load, f_center, q_L, q_C
    )
    
    rl_final = pi_opt.return_loss_db(f_center)
    il_final = pi_opt.insertion_loss_db(f_center)
    bw_result = pi_opt.analyze_bandwidth(
        f_start=f_center - 50e6,
        f_stop=f_center + 50e6,
        return_loss_threshold=-10.0
    )
    bw_final = bw_result['bandwidth_hz'] / 1e6
    
    # Check success
    success = (rl_final < target_rl_dB) and (bw_final >= target_bw_MHz * 0.9)
    
    opt_result = OptimizationResult(
        success=success,
        L=L_opt,
        C1=C1_opt,
        C2=C2_opt,
        return_loss_dB=rl_final,
        bandwidth_MHz=bw_final,
        insertion_loss_dB=il_final,
        n_iterations=n_iter,
        n_function_evals=n_evals[0],
        message=result.message if hasattr(result, 'message') else str(result.success),
        pi_match=pi_opt
    )
    
    if verbose:
        print("\n" + opt_result.summary())
    
    return opt_result


def optimize_pi_match_grid_search(z_source: float = 50.0,
                                   z_load: float = 30.0,
                                   f_center: float = 2.45e9,
                                   q_L: float = 50.0,
                                   q_C: float = 100.0,
                                   target_rl_dB: float = -20.0,
                                   target_bw_MHz: float = 30.0,
                                   n_points: int = 20,
                                   verbose: bool = True) -> OptimizationResult:
    """
    Coarse grid search followed by local refinement.
    
    Good for understanding the design space and finding robust solutions.
    """
    if verbose:
        print("="*60)
        print("Pi-Match Optimization (Grid Search + Refinement)")
        print("="*60)
    
    # Get analytical starting point for range estimation
    init_pi = design_pi_match(z_source, z_load, f_center, q_L, q_C)
    
    # Search range: 0.2x to 5x of analytical values
    L_range = np.linspace(init_pi.L.L * 0.2, init_pi.L.L * 5, n_points)
    C1_range = np.linspace(init_pi.C1.C * 0.2, init_pi.C1.C * 5, n_points)
    C2_range = np.linspace(init_pi.C2.C * 0.2, init_pi.C2.C * 5, n_points)
    
    best_obj = float('inf')
    best_x = None
    n_evals = 0
    
    if verbose:
        print(f"Grid search: {n_points}^3 = {n_points**3} points...")
    
    for L in L_range:
        for C1 in C1_range:
            for C2 in C2_range:
                x = np.array([L, C1, C2])
                obj = pi_match_objective(
                    x, z_source, z_load, f_center, q_L, q_C,
                    target_rl_dB, target_bw_MHz
                )
                n_evals += 1
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()
    
    if verbose:
        print(f"Grid search complete. Best objective: {best_obj:.4f}")
        print("Refining with local optimizer...")
    
    # Local refinement
    def obj_func(x):
        return pi_match_objective(
            x, z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz
        )
    
    result = minimize(
        obj_func,
        x0=best_x,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-6}
    )
    n_evals += result.nfev
    
    L_opt, C1_opt, C2_opt = result.x
    
    # Create final network
    pi_opt = create_pi_match_from_values(
        L_opt, C1_opt, C2_opt,
        z_source, z_load, f_center, q_L, q_C
    )
    
    rl_final = pi_opt.return_loss_db(f_center)
    il_final = pi_opt.insertion_loss_db(f_center)
    bw_result = pi_opt.analyze_bandwidth(return_loss_threshold=-10.0)
    bw_final = bw_result['bandwidth_hz'] / 1e6
    
    success = (rl_final < target_rl_dB) and (bw_final >= target_bw_MHz * 0.9)
    
    opt_result = OptimizationResult(
        success=success,
        L=L_opt,
        C1=C1_opt,
        C2=C2_opt,
        return_loss_dB=rl_final,
        bandwidth_MHz=bw_final,
        insertion_loss_dB=il_final,
        n_iterations=n_points**3,
        n_function_evals=n_evals,
        message="Grid search + Nelder-Mead refinement",
        pi_match=pi_opt
    )
    
    if verbose:
        print("\n" + opt_result.summary())
    
    return opt_result


# =============================================================================
# Optuna-based Multi-Objective Optimization (Optional Advanced Method)
# =============================================================================

def optimize_pi_match_optuna(z_source: float = 50.0,
                              z_load: float = 30.0,
                              f_center: float = 2.45e9,
                              q_L: float = 50.0,
                              q_C: float = 100.0,
                              target_rl_dB: float = -20.0,
                              target_bw_MHz: float = 30.0,
                              n_trials: int = 200,
                              verbose: bool = True) -> OptimizationResult:
    """
    Optimize using Optuna framework with TPE sampler.
    
    Optuna provides:
    - Tree-structured Parzen Estimator (TPE) for efficient search
    - Automatic pruning of bad trials
    - Easy parallelization
    - Built-in visualization
    
    Install with: pip install optuna
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("[WARNING] Optuna not installed. Install with: pip install optuna")
        print("         Falling back to scipy differential_evolution")
        return optimize_pi_match_scipy(
            z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz, 
            method='differential_evolution', verbose=verbose
        )
    
    if verbose:
        print("="*60)
        print("Pi-Match Optimization (Optuna TPE)")
        print("="*60)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Get initial guess for bounds
    init_pi = design_pi_match(z_source, z_load, f_center, q_L, q_C)
    
    def objective(trial):
        # Suggest values with log-uniform distribution
        L = trial.suggest_float('L', 0.1e-9, 50e-9, log=True)
        C1 = trial.suggest_float('C1', 0.1e-12, 100e-12, log=True)
        C2 = trial.suggest_float('C2', 0.1e-12, 100e-12, log=True)
        
        x = np.array([L, C1, C2])
        return pi_match_objective(
            x, z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz,
            weight_rl=2.0, weight_bw=1.5, weight_il=0.5
        )
    
    # Create study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    # Add initial point
    study.enqueue_trial({
        'L': init_pi.L.L,
        'C1': init_pi.C1.C,
        'C2': init_pi.C2.C
    })
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    
    # Extract best
    best = study.best_params
    L_opt, C1_opt, C2_opt = best['L'], best['C1'], best['C2']
    
    # Create final network
    pi_opt = create_pi_match_from_values(
        L_opt, C1_opt, C2_opt,
        z_source, z_load, f_center, q_L, q_C
    )
    
    rl_final = pi_opt.return_loss_db(f_center)
    il_final = pi_opt.insertion_loss_db(f_center)
    bw_result = pi_opt.analyze_bandwidth(return_loss_threshold=-10.0)
    bw_final = bw_result['bandwidth_hz'] / 1e6
    
    success = (rl_final < target_rl_dB) and (bw_final >= target_bw_MHz * 0.9)
    
    opt_result = OptimizationResult(
        success=success,
        L=L_opt,
        C1=C1_opt,
        C2=C2_opt,
        return_loss_dB=rl_final,
        bandwidth_MHz=bw_final,
        insertion_loss_dB=il_final,
        n_iterations=len(study.trials),
        n_function_evals=len(study.trials),
        message=f"Optuna TPE: best value = {study.best_value:.4f}",
        pi_match=pi_opt
    )
    
    if verbose:
        print("\n" + opt_result.summary())
    
    return opt_result


# =============================================================================
# High-Level Optimization Interface
# =============================================================================

def optimize_pi_match(z_source: float = 50.0,
                      z_load: float = 30.0,
                      f_center: float = 2.45e9,
                      q_L: float = 50.0,
                      q_C: float = 100.0,
                      target_rl_dB: float = -20.0,
                      target_bw_MHz: float = 30.0,
                      method: str = 'auto',
                      verbose: bool = True) -> OptimizationResult:
    """
    High-level interface for Pi-match optimization.
    
    Automatically selects the best method based on availability.
    
    Args:
        z_source: Source impedance (antenna), Ohms
        z_load: Load impedance (rectifier), Ohms  
        f_center: Center frequency, Hz
        q_L: Inductor Q factor (non-ideal)
        q_C: Capacitor Q factor (non-ideal)
        target_rl_dB: Target return loss (< this value)
        target_bw_MHz: Target bandwidth (> this value)
        method: 'auto', 'optuna', 'differential_evolution', 
                'dual_annealing', 'grid', 'SLSQP'
        verbose: Print progress
    
    Returns:
        OptimizationResult with optimal component values
    """
    if method == 'auto':
        # Try optuna first, fall back to scipy
        try:
            import optuna
            method = 'optuna'
        except ImportError:
            method = 'differential_evolution'
    
    if method == 'optuna':
        return optimize_pi_match_optuna(
            z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz, verbose=verbose
        )
    elif method == 'grid':
        return optimize_pi_match_grid_search(
            z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz, verbose=verbose
        )
    else:
        return optimize_pi_match_scipy(
            z_source, z_load, f_center, q_L, q_C,
            target_rl_dB, target_bw_MHz, method=method, verbose=verbose
        )


# =============================================================================
# Load-Pull Optimization: Maximize DC Output Power
# =============================================================================

def _build_netlist_for_load_pull(L: float, C1: float, C2: float,
                                  freq: float, v_amp: float, 
                                  c_in: float, c_out: float, r_load: float,
                                  n_cycles: int, ant_imp: float,
                                  q_L: float, q_C: float, cap_q: float,
                                  diode_model_name: str, model_path: Path) -> Tuple[str, float, int, float]:
    """
    Build SPICE netlist for load-pull optimization.
    
    Internal function used by optimize_pi_match_load_pull.
    """
    t_rf = 1.0 / freq
    t_step = t_rf / 40
    t_stop = n_cycles * t_rf
    
    # ESR for rectifier capacitors
    esr_in = calculate_esr_from_q(c_in, cap_q, freq)
    esr_out = calculate_esr_from_q(c_out, cap_q, freq)
    
    # Create Pi-match network
    pi_match = create_pi_match_from_values(L, C1, C2, ant_imp, ant_imp, freq, q_L, q_C)
    pi_match_netlist = pi_match.generate_spice_netlist(node_in='ant_out', node_out='pi_out')
    
    netlist = f"""Half-Wave RF Rectifier - Load Pull Optimization
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
D2 0 diode_anode {diode_model_name}
D1 diode_anode vout {diode_model_name}

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


def _run_single_load_pull_sim(L: float, C1: float, C2: float,
                               freq: float, v_amp: float,
                               c_in: float, c_out: float, r_load: float,
                               n_cycles: int, ant_imp: float,
                               q_L: float, q_C: float, cap_q: float,
                               diode_model_name: str, model_path: Path) -> Dict:
    """
    Run a single SPICE simulation and extract DC metrics.
    
    Returns:
        Dict with L, C1, C2, v_dc, ripple, p_out, or None if sim failed
    """
    try:
        netlist, t_stop, n_cyc, t_step = _build_netlist_for_load_pull(
            L, C1, C2, freq, v_amp, c_in, c_out, r_load,
            n_cycles, ant_imp, q_L, q_C, cap_q, diode_model_name, model_path
        )
        
        # Run simulation using ngspice
        data = run_transient_ngspice(netlist, t_stop, n_cyc, t_step, num_signals=3)
        sim_data = {
            'time': data['time'],
            'ant_out': data['signal_0'],
            'pi_out': data['signal_1'],
            'vout': data['signal_2']
        }
        
        # Extract DC metrics
        metrics = extract_dc_metrics(sim_data, r_load)
        
        # Calculate AVAILABLE input power from source
        # For a sinusoidal source with peak amplitude v_amp and source impedance ant_imp:
        # P_avail = V_peak^2 / (8 * R_source)
        # This is the maximum power that can be delivered to a matched load
        p_in_avail = v_amp**2 / (8 * ant_imp)
        
        # Efficiency = P_out / P_available
        efficiency = (metrics['p_out'] / p_in_avail) * 100 if p_in_avail > 0 else 0
        
        return {
            'L': L,
            'C1': C1,
            'C2': C2,
            'v_dc': metrics['v_dc'],
            'ripple': metrics['ripple'],
            'p_out': metrics['p_out'],
            'p_in': p_in_avail,
            'efficiency': efficiency,
            'success': True
        }
    except Exception as e:
        return {
            'L': L, 'C1': C1, 'C2': C2,
            'v_dc': 0, 'ripple': 0, 'p_out': 0, 'p_in': 0, 'efficiency': 0,
            'success': False, 'error': str(e)
        }


def estimate_effective_impedance(L_opt: float, C1_opt: float, C2_opt: float,
                                  freq: float, z_source: float,
                                  q_L: float, q_C: float) -> complex:
    """
    Estimate the effective input impedance seen by the source through the Pi-match.
    
    This computes what impedance transformation the optimal Pi-match is providing,
    which tells us what the "effective" load impedance is at the maximum power point.
    
    At maximum power transfer, the source sees Z_eff ≈ Z_source* (conjugate match).
    We can back-calculate what Z_load the Pi-match was transforming.
    """
    try:
        # Create the optimal Pi-match network
        # Note: We use z_source as both source and nominal load since we don't
        # know the "true" load - the optimization found it empirically
        pi = create_pi_match_from_values(L_opt, C1_opt, C2_opt, 
                                          z_source, z_source, freq, q_L, q_C)
        
        # Get the input impedance at center frequency
        # This is Z_in seen looking into the Pi-match
        z_in = pi.input_impedance(freq)
        
        # The effective load impedance that achieves max power is approximately
        # the conjugate of what the Pi-match presents to the source
        # Z_eff = Z_in* at resonance for max power transfer
        return z_in
        
    except Exception:
        return complex(z_source, 0)


def optimize_pi_match_load_pull(
    freq: float = 2.437e9,
    v_amp: float = 0.3,
    r_load: float = 5000.0,
    c_in: float = 100e-12,
    c_out: float = 100e-12,
    ant_imp: float = 50.0,
    q_L: float = 50.0,
    q_C: float = 100.0,
    cap_q: float = 30.0,
    n_cycles: int = 100,
    diode_model_name: str = "SMS7630",
    diode_model_file: str = "diode_models.lib",
    L_range: Tuple[float, float] = None,
    C1_range: Tuple[float, float] = None,
    C2_range: Tuple[float, float] = None,
    n_points: int = 10,
    refine: bool = True,
    verbose: bool = True
) -> LoadPullResult:
    """
    Load-Pull Optimization: Find Pi-match values that MAXIMIZE DC output power.
    
    This is the proper approach for nonlinear rectifier optimization:
    1. Sweep Pi-match component values (L, C1, C2)
    2. For each combination, run full SPICE simulation with diode model
    3. Measure actual DC output power
    4. Find the combination that maximizes P_out
    5. At max power point, compute/estimate the effective impedance
    
    Unlike traditional S-parameter optimization (which assumes linear load),
    this method directly optimizes the real objective: rectified DC power.
    
    Args:
        freq: Operating frequency (Hz) - default 2.437 GHz (WiFi Ch 6)
        v_amp: RF input amplitude (V)
        r_load: DC load resistance (Ohms)
        c_in: Input coupling capacitor (F)
        c_out: Output filter capacitor (F)
        ant_imp: Antenna/source impedance (Ohms)
        q_L: Inductor Q factor
        q_C: Pi-match capacitor Q factor
        cap_q: Rectifier capacitor Q factor
        n_cycles: Number of RF cycles for simulation
        diode_model_name: SPICE diode model name
        diode_model_file: Path to diode model library
        L_range: (L_min, L_max) in H, or None for auto
        C1_range: (C1_min, C1_max) in F, or None for auto
        C2_range: (C2_min, C2_max) in F, or None for auto
        n_points: Number of points per dimension for grid search
        refine: If True, refine with local optimization after grid search
        verbose: Print progress
    
    Returns:
        LoadPullResult with optimal Pi-match values and performance metrics
    """
    if verbose:
        print("\n" + "="*70)
        print("LOAD-PULL OPTIMIZATION: Maximize DC Output Power")
        print("="*70)
        print(f"  Frequency:      {freq/1e9:.3f} GHz")
        print(f"  V_RF amplitude: {v_amp*1000:.1f} mV")
        print(f"  R_load:         {r_load/1000:.1f} kΩ")
        print(f"  Antenna Z:      {ant_imp} Ω")
        print(f"  Diode model:    {diode_model_name}")
        print(f"  Grid points:    {n_points}³ = {n_points**3} simulations")
        print("-"*70)
    
    # Get diode model path
    script_dir = Path(__file__).parent.absolute()
    model_path = script_dir / diode_model_file
    
    # Auto-determine search ranges if not provided
    # Start from analytical design as center point
    init_pi = design_pi_match(ant_imp, ant_imp, freq, q_L, q_C)
    
    if L_range is None:
        L_center = init_pi.L.L
        L_range = (L_center * 0.1, L_center * 10)
    if C1_range is None:
        C1_center = init_pi.C1.C
        C1_range = (C1_center * 0.1, C1_center * 10)
    if C2_range is None:
        C2_center = init_pi.C2.C
        C2_range = (C2_center * 0.1, C2_center * 10)
    
    if verbose:
        print(f"  L  range: {L_range[0]*1e9:.2f} - {L_range[1]*1e9:.2f} nH")
        print(f"  C1 range: {C1_range[0]*1e12:.2f} - {C1_range[1]*1e12:.2f} pF")
        print(f"  C2 range: {C2_range[0]*1e12:.2f} - {C2_range[1]*1e12:.2f} pF")
        print("-"*70)
    
    # Create parameter grid
    L_vals = np.linspace(L_range[0], L_range[1], n_points)
    C1_vals = np.linspace(C1_range[0], C1_range[1], n_points)
    C2_vals = np.linspace(C2_range[0], C2_range[1], n_points)
    
    all_results = []
    best_result = None
    best_power = -np.inf
    n_sims = 0
    n_failed = 0
    
    total_sims = n_points ** 3
    
    if verbose:
        print(f"Running {total_sims} SPICE simulations...")
    
    for i, L in enumerate(L_vals):
        for j, C1 in enumerate(C1_vals):
            for k, C2 in enumerate(C2_vals):
                n_sims += 1
                
                # Suppress detailed ngspice output during sweep
                result = _run_single_load_pull_sim(
                    L, C1, C2, freq, v_amp, c_in, c_out, r_load,
                    n_cycles, ant_imp, q_L, q_C, cap_q,
                    diode_model_name, model_path
                )
                
                all_results.append(result)
                
                if result['success'] and result['p_out'] > best_power:
                    best_power = result['p_out']
                    best_result = result.copy()
                
                if not result['success']:
                    n_failed += 1
                
                # Progress indicator
                if verbose and n_sims % max(1, total_sims // 10) == 0:
                    print(f"  Progress: {n_sims}/{total_sims} ({100*n_sims/total_sims:.0f}%), "
                          f"Best P_out: {best_power*1e6:.2f} µW")
    
    if verbose:
        print(f"Grid search complete. Failed sims: {n_failed}/{total_sims}")
        if best_result:
            print(f"Best power from grid: {best_result['p_out']*1e6:.2f} µW")
    
    # Refine with local optimization if requested
    if refine and best_result:
        if verbose:
            print("\nRefining with Nelder-Mead local optimization...")
        
        def neg_power_objective(x):
            """Negative power for minimization."""
            L, C1, C2 = x
            if L <= 0 or C1 <= 0 or C2 <= 0:
                return 1e10
            result = _run_single_load_pull_sim(
                L, C1, C2, freq, v_amp, c_in, c_out, r_load,
                n_cycles, ant_imp, q_L, q_C, cap_q,
                diode_model_name, model_path
            )
            if result['success']:
                all_results.append(result)
                return -result['p_out']  # Negative for minimization
            return 1e10
        
        x0 = np.array([best_result['L'], best_result['C1'], best_result['C2']])
        
        opt_result = minimize(
            neg_power_objective,
            x0=x0,
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        n_sims += opt_result.nfev
        
        # Check if refinement improved
        final_result = _run_single_load_pull_sim(
            opt_result.x[0], opt_result.x[1], opt_result.x[2],
            freq, v_amp, c_in, c_out, r_load,
            n_cycles, ant_imp, q_L, q_C, cap_q,
            diode_model_name, model_path
        )
        n_sims += 1
        
        if final_result['success'] and final_result['p_out'] > best_result['p_out']:
            best_result = final_result
            if verbose:
                print(f"Refinement improved power to: {best_result['p_out']*1e6:.2f} µW")
        else:
            if verbose:
                print("Refinement did not improve - keeping grid search result")
    
    # Build result
    if best_result:
        # Create optimal Pi-match network object
        pi_opt = create_pi_match_from_values(
            best_result['L'], best_result['C1'], best_result['C2'],
            ant_imp, ant_imp, freq, q_L, q_C
        )
        
        # Convert V_amp to dBm for storage
        p_in_dBm = vpeak_to_dBm(v_amp, ant_imp)
        
        result = LoadPullResult(
            success=True,
            L=best_result['L'],
            C1=best_result['C1'],
            C2=best_result['C2'],
            max_power_mW=best_result['p_out'] * 1000,
            v_dc_at_max=best_result['v_dc'],
            ripple_at_max=best_result['ripple'],
            efficiency_percent=best_result['efficiency'],
            n_simulations=n_sims,
            input_power_dBm=p_in_dBm,
            all_results=all_results,
            pi_match=pi_opt
        )
    else:
        result = LoadPullResult(
            success=False,
            L=0, C1=0, C2=0,
            max_power_mW=0,
            v_dc_at_max=0,
            ripple_at_max=0,
            efficiency_percent=0,
            n_simulations=n_sims,
            input_power_dBm=vpeak_to_dBm(v_amp, ant_imp),
            all_results=all_results,
            pi_match=None
        )
    
    if verbose:
        print("\n" + result.summary())
    
    return result


def plot_load_pull_results(result: LoadPullResult,
                           save_path: str = None,
                           show: bool = True):
    """
    Plot load-pull optimization results (publication style).
    
    Shows how DC output power varies with Pi-match component values.
    """
    import matplotlib.pyplot as plt
    
    # Apply publication style
    try:
        from utility import apply_pub_style, get_save_path, PUB_STYLE
        apply_pub_style()
    except ImportError:
        pass
    
    if not result.all_results:
        print("No results to plot")
        return None
    
    # Extract data
    successful = [r for r in result.all_results if r.get('success', False)]
    if len(successful) < 10:
        print("Not enough successful simulations to plot")
        return None
    
    L_vals = np.array([r['L'] for r in successful]) * 1e9  # nH
    C1_vals = np.array([r['C1'] for r in successful]) * 1e12  # pF
    C2_vals = np.array([r['C2'] for r in successful]) * 1e12  # pF
    p_out_vals = np.array([r['p_out'] for r in successful]) * 1e6  # µW
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left: L vs C1 scatter with power as color
    sc1 = axes[0].scatter(L_vals, C1_vals, c=p_out_vals, cmap='hot', 
                          s=50, alpha=0.7, edgecolors='none')
    axes[0].scatter([result.L*1e9], [result.C1*1e12], marker='*', 
                    s=400, c='cyan', edgecolors='black', linewidths=2, 
                    label='Optimal', zorder=10)
    axes[0].set_xlabel('Inductance L (nH)')
    axes[0].set_ylabel('Capacitor C1 (pF)')
    axes[0].legend(loc='upper right')
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label('P_out (µW)')
    for spine in axes[0].spines.values():
        spine.set_linewidth(2)
    
    # Right: Power histogram with statistics
    axes[1].hist(p_out_vals, bins=30, color='steelblue', edgecolor='black', 
                 alpha=0.7, linewidth=1.5)
    axes[1].axvline(result.max_power_mW * 1000, color='red', linestyle='--', 
                    linewidth=2.5, label=f'Max: {result.max_power_mW*1000:.2f} µW')
    mean_p = np.mean(p_out_vals)
    axes[1].axvline(mean_p, color='orange', linestyle=':', linewidth=2,
                    label=f'Mean: {mean_p:.2f} µW')
    axes[1].set_xlabel('DC Output Power (µW)')
    axes[1].set_ylabel('Count')
    axes[1].legend(loc='upper right')
    for spine in axes[1].spines.values():
        spine.set_linewidth(2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"[INFO] Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


def optimize_pi_match_load_pull_vs_power(
    freq: float = 2.437e9,
    p_dBm_list: List[float] = None,
    r_load: float = 5000.0,
    c_in: float = 100e-12,
    c_out: float = 100e-12,
    ant_imp: float = 50.0,
    q_L: float = 50.0,
    q_C: float = 100.0,
    cap_q: float = 30.0,
    n_cycles: int = 100,
    diode_model_name: str = "SMS7630",
    diode_model_file: str = "diode_models.lib",
    n_points: int = 6,
    verbose: bool = True
) -> LoadPullSweepResult:
    """
    Full Load-Pull Characterization: Sweep BOTH input power AND matching values.
    
    This is the proper load-pull characterization:
    - Outer loop: Sweep input power levels (dBm)
    - Inner loop: For each power, sweep Pi-match values (L, C1, C2)
    - Find optimal matching at each power level
    - Track how optimal component values vary with input power
    
    This reveals the nonlinear nature of the rectifier - the optimal matching
    changes with input power level due to the diode's nonlinearity.
    
    Args:
        freq: Operating frequency (Hz)
        p_dBm_list: List of input powers (dBm) to sweep. Default: -30 to 0 dBm
        r_load: DC load resistance (Ohms)
        c_in, c_out: Coupling/filter capacitors (F)
        ant_imp: Antenna impedance (Ohms)
        q_L, q_C: Component Q factors
        cap_q: Rectifier capacitor Q
        n_cycles: Simulation cycles
        diode_model_name: Diode model
        diode_model_file: Model file
        n_points: Grid points per dimension (total sims per power = n_points³)
        verbose: Print progress
    
    Returns:
        LoadPullSweepResult with optimal values at each power level
    """
    if p_dBm_list is None:
        # Default: -30 dBm to 0 dBm in 6 steps (typical RF harvesting range)
        p_dBm_list = [-30, -25, -20, -15, -10, -5, 0]
    
    if verbose:
        print("\n" + "="*70)
        print("FULL LOAD-PULL CHARACTERIZATION vs INPUT POWER")
        print("="*70)
        print(f"  Frequency:      {freq/1e9:.3f} GHz")
        print(f"  Power levels:   {len(p_dBm_list)} ({min(p_dBm_list):.0f} to {max(p_dBm_list):.0f} dBm)")
        print(f"  Grid per power: {n_points}³ = {n_points**3} sims")
        print(f"  Total sims:     ~{len(p_dBm_list) * n_points**3}")
        print("="*70)
        
        # Show dBm to V_peak conversion
        print("\n  Power level reference:")
        for p in p_dBm_list:
            v = dBm_to_vpeak(p, ant_imp)
            print(f"    {p:>5.0f} dBm → {v*1000:>7.2f} mV peak")
        print()
    
    results_per_power = []
    optimal_L = []
    optimal_C1 = []
    optimal_C2 = []
    max_power = []
    efficiency_list = []
    total_sims = 0
    
    for i, p_dBm in enumerate(p_dBm_list):
        v_amp = dBm_to_vpeak(p_dBm, ant_imp)
        
        if verbose:
            print(f"\n--- Power Level {i+1}/{len(p_dBm_list)}: P_in = {p_dBm:.0f} dBm ({v_amp*1000:.2f} mV) ---")
        
        # Run load-pull optimization at this power level
        result = optimize_pi_match_load_pull(
            freq=freq,
            v_amp=v_amp,
            r_load=r_load,
            c_in=c_in,
            c_out=c_out,
            ant_imp=ant_imp,
            q_L=q_L,
            q_C=q_C,
            cap_q=cap_q,
            n_cycles=n_cycles,
            diode_model_name=diode_model_name,
            diode_model_file=diode_model_file,
            n_points=n_points,
            refine=True,
            verbose=False  # Suppress inner verbose
        )
        
        results_per_power.append(result)
        total_sims += result.n_simulations
        
        if result.success:
            optimal_L.append(result.L)
            optimal_C1.append(result.C1)
            optimal_C2.append(result.C2)
            max_power.append(result.max_power_mW / 1000)  # Convert to W
            efficiency_list.append(result.efficiency_percent)
            
            if verbose:
                p_out_uW = result.max_power_mW * 1000
                print(f"  Optimal: L={result.L*1e9:.3f}nH, C1={result.C1*1e12:.3f}pF, "
                      f"C2={result.C2*1e12:.3f}pF")
                print(f"  P_out={p_out_uW:.2f}µW, Eff={result.efficiency_percent:.1f}%")
        else:
            # Failed - use NaN
            optimal_L.append(np.nan)
            optimal_C1.append(np.nan)
            optimal_C2.append(np.nan)
            max_power.append(np.nan)
            efficiency_list.append(np.nan)
            if verbose:
                print(f"  FAILED at this power level")
    
    sweep_result = LoadPullSweepResult(
        success=any(r.success for r in results_per_power),
        power_levels_dBm=p_dBm_list,
        results_per_power=results_per_power,
        optimal_L_vs_power=optimal_L,
        optimal_C1_vs_power=optimal_C1,
        optimal_C2_vs_power=optimal_C2,
        max_power_vs_input=max_power,
        efficiency_vs_input=efficiency_list,
        total_simulations=total_sims
    )
    
    if verbose:
        print("\n" + sweep_result.summary())
    
    return sweep_result


def plot_load_pull_vs_power(result: LoadPullSweepResult,
                            save_dir: str = "temp_image",
                            show: bool = True):
    """
    Plot load-pull characterization vs input power.
    
    Creates individual plot files (no titles - filename is the identifier):
      - optimal_L_vs_power.png: Optimal inductance vs input power
      - optimal_C_vs_power.png: Optimal capacitances vs input power
      - output_power_vs_input.png: DC output power vs input power
      - efficiency_vs_input.png: Efficiency vs input power
    
    Args:
        result: LoadPullSweepResult from optimize_pi_match_load_pull_vs_power
        save_dir: Directory to save plots
        show: Whether to display plots interactively
    """
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # Apply publication style
    try:
        from utility import apply_pub_style
        apply_pub_style()
    except ImportError:
        pass
    
    if not result.success:
        print("No successful results to plot")
        return
    
    # Ensure save directory exists
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data (filter NaN)
    p_in = np.array(result.power_levels_dBm)
    L_opt = np.array(result.optimal_L_vs_power) * 1e9  # nH
    C1_opt = np.array(result.optimal_C1_vs_power) * 1e12  # pF
    C2_opt = np.array(result.optimal_C2_vs_power) * 1e12  # pF
    p_out = np.array(result.max_power_vs_input) * 1e6  # µW
    eff = np.array(result.efficiency_vs_input)
    
    # Mask NaN values
    valid = ~np.isnan(L_opt)
    p_in = p_in[valid]
    L_opt = L_opt[valid]
    C1_opt = C1_opt[valid]
    C2_opt = C2_opt[valid]
    p_out = p_out[valid]
    eff = eff[valid]
    
    if len(p_in) < 2:
        print("Not enough valid data points")
        return
    
    # ===== Plot 1: Optimal Inductance vs Input Power =====
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.plot(p_in, L_opt, 'k-', lw=2.5, marker='o', ms=12, 
             mfc='white', mew=2)
    ax1.set_xlabel('Input Power (dBm)')
    ax1.set_ylabel('Optimal Inductance L (nH)')
    ax1.grid(True, alpha=0.3, linestyle='--')
    for spine in ax1.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    fig1.savefig(save_path / 'optimal_L_vs_power.png', dpi=300, 
                 bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved: {save_path / 'optimal_L_vs_power.png'}")
    
    # ===== Plot 2: Optimal Capacitances vs Input Power =====
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.plot(p_in, C1_opt, 'b-', lw=2.5, marker='s', ms=10, 
             mfc='white', mew=2, label='C1 (input)')
    ax2.plot(p_in, C2_opt, 'r--', lw=2.5, marker='^', ms=10, 
             mfc='white', mew=2, label='C2 (output)')
    ax2.set_xlabel('Input Power (dBm)')
    ax2.set_ylabel('Optimal Capacitance (pF)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    for spine in ax2.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    fig2.savefig(save_path / 'optimal_C_vs_power.png', dpi=300, 
                 bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved: {save_path / 'optimal_C_vs_power.png'}")
    
    # ===== Plot 3: DC Output Power vs Input Power =====
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.semilogy(p_in, p_out, 'k-', lw=2.5, marker='o', ms=12, 
                 mfc='white', mew=2)
    ax3.set_xlabel('Input Power (dBm)')
    ax3.set_ylabel('Maximum DC Output Power (µW)')
    ax3.grid(True, which='both', alpha=0.3, linestyle='--')
    for spine in ax3.spines.values():
        spine.set_linewidth(2)
    
    # Add power conversion annotation
    try:
        # Linear fit in log space
        valid_p = p_out > 0
        if np.sum(valid_p) >= 2:
            slope, intercept = np.polyfit(p_in[valid_p], np.log10(p_out[valid_p]), 1)
            ax3.text(0.05, 0.95, f'Slope: {slope:.2f} dB/dB', 
                     transform=ax3.transAxes, fontsize=16, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except:
        pass
    
    plt.tight_layout()
    fig3.savefig(save_path / 'output_power_vs_input.png', dpi=300, 
                 bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved: {save_path / 'output_power_vs_input.png'}")
    
    # ===== Plot 4: Efficiency vs Input Power =====
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.plot(p_in, eff, 'k-', lw=2.5, marker='o', ms=12, 
             mfc='white', mew=2)
    ax4.set_xlabel('Input Power (dBm)')
    ax4.set_ylabel('Power Conversion Efficiency (%)')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(bottom=0)
    for spine in ax4.spines.values():
        spine.set_linewidth(2)
    plt.tight_layout()
    fig4.savefig(save_path / 'efficiency_vs_input.png', dpi=300, 
                 bbox_inches='tight', facecolor='white')
    print(f"[INFO] Saved: {save_path / 'efficiency_vs_input.png'}")
    
    if show:
        plt.show()
    else:
        plt.close('all')
    
    print(f"\n[INFO] All plots saved to '{save_dir}/'")


# =============================================================================
# Plotting Utilities for Optimization Results
# =============================================================================

def plot_optimization_result(opt_result: OptimizationResult,
                              save_path: str = None,
                              show: bool = True):
    """
    Plot the S-parameters of the optimized Pi-match network.
    """
    import matplotlib.pyplot as plt
    
    # Use publication style if available
    try:
        from utility import apply_pub_style, get_save_path
        apply_pub_style()
    except ImportError:
        pass
    
    pi = opt_result.pi_match
    f_center = pi.f_center
    
    # Frequency range
    freqs = np.linspace(f_center - 100e6, f_center + 100e6, 501)
    s11_db = np.array([pi.return_loss_db(f) for f in freqs])
    s21_db = np.array([pi.insertion_loss_db(f) for f in freqs])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # S11 plot
    ax1.plot(freqs/1e9, s11_db, 'k-', lw=2, label='|S11|')
    ax1.axhline(-20, color='r', ls='--', lw=1.5, label='-20 dB target')
    ax1.axhline(-10, color='orange', ls=':', lw=1.5, label='-10 dB')
    ax1.axvline(f_center/1e9, color='gray', ls=':', alpha=0.5)
    
    # Shade bandwidth region
    bw_result = pi.analyze_bandwidth(return_loss_threshold=-10)
    if bw_result['bandwidth_hz'] > 0:
        ax1.axvspan(bw_result['f_low']/1e9, bw_result['f_high']/1e9,
                    alpha=0.2, color='green', label=f"BW={bw_result['bandwidth_hz']/1e6:.1f}MHz")
    
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('|S11| (dB)')
    ax1.set_title('Return Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-40, 0])
    
    # S21 plot
    ax2.plot(freqs/1e9, s21_db, 'k-', lw=2, label='|S21|')
    ax2.axvline(f_center/1e9, color='gray', ls=':', alpha=0.5)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('|S21| (dB)')
    ax2.set_title('Insertion Loss')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-10, 2])
    
    # Add component values as text
    info_text = (f"L = {opt_result.L*1e9:.3f} nH\n"
                 f"C1 = {opt_result.C1*1e12:.3f} pF\n"
                 f"C2 = {opt_result.C2*1e12:.3f} pF")
    ax2.text(0.95, 0.05, info_text, transform=ax2.transAxes,
             fontsize=12, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved plot to {save_path}")
    
    if show:
        plt.show()
    
    return fig


# =============================================================================
# Main / Demo
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RF Energy Harvester - Pi-Match Optimization Demo")
    print("="*70)
    
    # Target specifications
    Z_ANT = 50          # Antenna impedance
    Z_RECT = 30         # Estimated rectifier input impedance
    F_CENTER = 2.45e9   # Center frequency
    Q_L = 50            # Realistic inductor Q at 2.45 GHz
    Q_C = 100           # Realistic capacitor Q at 2.45 GHz
    
    # Constraints
    TARGET_RL = -20     # Return loss < -20 dB
    TARGET_BW = 30      # Bandwidth > 30 MHz
    
    print(f"\nTarget Specifications:")
    print(f"  Source (antenna): {Z_ANT} Ω")
    print(f"  Load (rectifier): {Z_RECT} Ω")
    print(f"  Center frequency: {F_CENTER/1e9:.2f} GHz")
    print(f"  Return loss: < {TARGET_RL} dB")
    print(f"  Bandwidth: > {TARGET_BW} MHz")
    print(f"  Component Q: L={Q_L}, C={Q_C}")
    
    # Run optimization
    result = optimize_pi_match(
        z_source=Z_ANT,
        z_load=Z_RECT,
        f_center=F_CENTER,
        q_L=Q_L,
        q_C=Q_C,
        target_rl_dB=TARGET_RL,
        target_bw_MHz=TARGET_BW,
        method='auto',
        verbose=True
    )
    
    # Plot results
    print("\nGenerating S-parameter plot...")
    plot_optimization_result(result, save_path=None, show=True)
    
    # Print SPICE netlist
    print("\nSPICE Netlist for optimized Pi-match:")
    print("-"*50)
    print(result.pi_match.generate_spice_netlist())
    print("-"*50)
