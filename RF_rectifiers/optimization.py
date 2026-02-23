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
from dataclasses import dataclass
import warnings

# Scipy optimization
from scipy.optimize import minimize, differential_evolution, dual_annealing
from scipy.optimize import NonlinearConstraint, Bounds

# Import local pi_match module
from pi_match import (
    PiMatchNetwork, Inductor, Capacitor,
    create_pi_match_from_values, design_pi_match
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
