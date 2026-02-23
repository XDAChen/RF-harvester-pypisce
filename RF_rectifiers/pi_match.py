#!/usr/bin/env python3
"""
================================================================================
Pi-Matching Network Module for RF Energy Harvesting
================================================================================
Modular Pi-matching network design with non-ideal component models.

Features:
    - Non-ideal L and C with Q-factor modeling
    - S-parameter calculations (return loss, insertion loss)
    - Bandwidth analysis
    - SPICE netlist generation

Pi-Network Topology (Low-Pass):

                    L (series)
                ┌────LLLL────RL────┐
                │                  │
    IN o────────┼──────────────────┼────────o OUT
                │                  │
               ═══ C1             ═══ C2
               ─┬─                ─┬─
                │                  │
               GND                GND
    
    - C1, C2: Shunt capacitors (with ESR to ground)
    - L: Series inductor (RL = ESR from Q factor)
    
    This is a low-pass Pi-match for impedance transformation.


================================================================================
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict


# =============================================================================
# Non-Ideal Component Models
# =============================================================================

@dataclass
class Inductor:
    """Non-ideal inductor model with Q factor."""
    L: float           # Inductance in Henries
    Q: float = 50      # Quality factor at design frequency
    f_design: float = 2.45e9  # Design frequency for Q
    
    @property
    def esr(self) -> float:
        """Equivalent series resistance at design frequency."""
        omega = 2 * np.pi * self.f_design
        return omega * self.L / self.Q
    
    def impedance(self, freq: float) -> complex:
        """Complex impedance at given frequency."""
        omega = 2 * np.pi * freq
        # ESR is approximately constant, reactance scales with frequency
        esr_f = self.esr  # Can model frequency-dependent ESR if needed
        return esr_f + 1j * omega * self.L
    
    def __repr__(self):
        return f"Inductor(L={self.L*1e9:.3f}nH, Q={self.Q}, ESR={self.esr*1e3:.2f}mΩ)"


@dataclass
class Capacitor:
    """Non-ideal capacitor model with Q factor."""
    C: float           # Capacitance in Farads
    Q: float = 100     # Quality factor at design frequency
    f_design: float = 2.45e9  # Design frequency for Q
    
    @property
    def esr(self) -> float:
        """Equivalent series resistance at design frequency."""
        omega = 2 * np.pi * self.f_design
        return 1.0 / (omega * self.C * self.Q)
    
    def impedance(self, freq: float) -> complex:
        """Complex impedance at given frequency."""
        omega = 2 * np.pi * freq
        esr_f = self.esr
        return esr_f - 1j / (omega * self.C)
    
    def __repr__(self):
        return f"Capacitor(C={self.C*1e12:.3f}pF, Q={self.Q}, ESR={self.esr*1e3:.2f}mΩ)"


# =============================================================================
# Pi-Matching Network Class
# =============================================================================

@dataclass
class PiMatchNetwork:
    """
    Pi-Matching Network with non-ideal components.
    
    Topology:
        IN ──┬── C1 ──┬── L ──┬── C2 ──┬── OUT
             │        │       │        │
            GND      GND     GND      GND
    
    Attributes:
        L: Inductor object (series element)
        C1: Capacitor object (input shunt)
        C2: Capacitor object (output shunt)
        z_source: Source impedance (Ohms)
        z_load: Load impedance (Ohms)
        f_center: Center frequency (Hz)
    """
    L: Inductor
    C1: Capacitor
    C2: Capacitor
    z_source: float = 50.0     # Source impedance (antenna)
    z_load: float = 50.0       # Load impedance (rectifier input)
    f_center: float = 2.45e9   # Center frequency
    
    def calculate_s_parameters(self, freq: float) -> Dict[str, complex]:
        """
        Calculate S-parameters at given frequency using ABCD matrix method.
        
        Returns dict with S11, S21, S12, S22 (complex)
        """
        # ABCD matrix for Pi network
        # Pi = shunt(C1) * series(L) * shunt(C2)
        
        # Impedances
        Z_L = self.L.impedance(freq)
        Y_C1 = 1.0 / self.C1.impedance(freq)
        Y_C2 = 1.0 / self.C2.impedance(freq)
        
        # ABCD for shunt admittance: [[1, 0], [Y, 1]]
        # ABCD for series impedance: [[1, Z], [0, 1]]
        
        # Shunt C1
        A1 = np.array([[1, 0], [Y_C1, 1]], dtype=complex)
        
        # Series L (with ESR)
        A2 = np.array([[1, Z_L], [0, 1]], dtype=complex)
        
        # Shunt C2
        A3 = np.array([[1, 0], [Y_C2, 1]], dtype=complex)
        
        # Total ABCD = A1 * A2 * A3
        ABCD = A1 @ A2 @ A3
        A, B, C, D = ABCD[0, 0], ABCD[0, 1], ABCD[1, 0], ABCD[1, 1]
        
        # Convert ABCD to S-parameters
        Z0_s = self.z_source
        Z0_l = self.z_load
        
        # For different source/load impedances
        denom = A * Z0_l + B + C * Z0_s * Z0_l + D * Z0_s
        
        S11 = (A * Z0_l + B - C * Z0_s * Z0_l - D * Z0_s) / denom
        S21 = 2 * np.sqrt(Z0_s * Z0_l) / denom
        S12 = 2 * np.sqrt(Z0_s * Z0_l) * (A * D - B * C) / denom
        S22 = (-A * Z0_l + B - C * Z0_s * Z0_l + D * Z0_s) / denom
        
        return {'S11': S11, 'S21': S21, 'S12': S12, 'S22': S22}
    
    def return_loss_db(self, freq: float) -> float:
        """Return loss in dB at given frequency. More negative = better match."""
        s_params = self.calculate_s_parameters(freq)
        return 20 * np.log10(np.abs(s_params['S11']) + 1e-12)
    
    def insertion_loss_db(self, freq: float) -> float:
        """Insertion loss in dB at given frequency. More negative = more loss."""
        s_params = self.calculate_s_parameters(freq)
        return 20 * np.log10(np.abs(s_params['S21']) + 1e-12)
    
    def voltage_gain(self, freq: float) -> float:
        """Voltage gain magnitude at given frequency."""
        s_params = self.calculate_s_parameters(freq)
        # Voltage gain from S21 considering impedance transformation
        return np.abs(s_params['S21']) * np.sqrt(self.z_load / self.z_source)
    
    def analyze_bandwidth(self, f_start: float = None, f_stop: float = None, 
                          n_points: int = 501, return_loss_threshold: float = -10.0
                          ) -> Dict:
        """
        Analyze bandwidth of the matching network.
        
        Args:
            f_start: Start frequency (default: f_center - 100 MHz)
            f_stop: Stop frequency (default: f_center + 100 MHz)
            n_points: Number of frequency points
            return_loss_threshold: Threshold for bandwidth calculation (dB)
        
        Returns:
            Dict with frequency array, S11, S21, bandwidth info
        """
        if f_start is None:
            f_start = self.f_center - 100e6
        if f_stop is None:
            f_stop = self.f_center + 100e6
        
        freqs = np.linspace(f_start, f_stop, n_points)
        s11_db = np.array([self.return_loss_db(f) for f in freqs])
        s21_db = np.array([self.insertion_loss_db(f) for f in freqs])
        
        # Find minimum return loss (best match)
        min_idx = np.argmin(s11_db)
        f_min = freqs[min_idx]
        rl_min = s11_db[min_idx]
        
        # Calculate bandwidth at threshold
        below_threshold = s11_db < return_loss_threshold
        if np.any(below_threshold):
            # Find edges
            transitions = np.diff(below_threshold.astype(int))
            start_indices = np.where(transitions == 1)[0]
            stop_indices = np.where(transitions == -1)[0]
            
            if len(start_indices) > 0 and len(stop_indices) > 0:
                f_low = freqs[start_indices[0] + 1]
                f_high = freqs[stop_indices[0]]
                bandwidth = f_high - f_low
            elif len(start_indices) > 0:
                f_low = freqs[start_indices[0] + 1]
                f_high = freqs[-1]
                bandwidth = f_high - f_low
            elif len(stop_indices) > 0:
                f_low = freqs[0]
                f_high = freqs[stop_indices[0]]
                bandwidth = f_high - f_low
            else:
                f_low = freqs[0]
                f_high = freqs[-1]
                bandwidth = f_high - f_low
        else:
            f_low = f_high = f_min
            bandwidth = 0
        
        return {
            'freqs': freqs,
            'S11_dB': s11_db,
            'S21_dB': s21_db,
            'f_center_actual': f_min,
            'return_loss_min_dB': rl_min,
            'bandwidth_hz': bandwidth,
            'f_low': f_low,
            'f_high': f_high,
            'threshold_dB': return_loss_threshold
        }
    
    def generate_spice_netlist(self, node_in: str = 'ant_out', 
                                node_out: str = 'pi_out') -> str:
        """
        Generate SPICE netlist fragment for the Pi-match network.
        
        Args:
            node_in: Input node name (default: ant_out)
            node_out: Output node name (default: pi_out)
        
        Returns:
            SPICE netlist string (inline fragment)
        """
        return f"""* === Pi-Matching Network (Non-Ideal) ===
* L={self.L.L*1e9:.4f}nH (Q={self.L.Q}), C1={self.C1.C*1e12:.4f}pF, C2={self.C2.C*1e12:.4f}pF (Q={self.C1.Q})
* Input shunt capacitor C1 with ESR
C_pi1 {node_in} pi_c1_esr {self.C1.C}
R_pi1_esr pi_c1_esr 0 {self.C1.esr}
* Series inductor L with ESR
L_pi {node_in} pi_L_out {self.L.L}
R_pi_L_esr pi_L_out {node_out} {self.L.esr}
* Output shunt capacitor C2 with ESR
C_pi2 {node_out} pi_c2_esr {self.C2.C}
R_pi2_esr pi_c2_esr 0 {self.C2.esr}"""
    
    def __repr__(self):
        return (f"PiMatchNetwork(\n"
                f"  {self.L},\n"
                f"  {self.C1},\n"
                f"  {self.C2},\n"
                f"  Z_source={self.z_source}Ω, Z_load={self.z_load}Ω\n"
                f")")


# =============================================================================
# Design Functions
# =============================================================================

def design_pi_match(z_source: float, z_load: float, f_center: float,
                    q_L: float = 50, q_C: float = 100,
                    target_q: float = None) -> PiMatchNetwork:
    """
    Design a Pi-matching network using standard formulas.
    
    This uses the two-section L-match approach through a virtual resistance.
    
    Args:
        z_source: Source impedance (Ohms)
        z_load: Load impedance (Ohms)
        f_center: Center frequency (Hz)
        q_L: Quality factor of inductor
        q_C: Quality factor of capacitors
        target_q: Optional target Q for bandwidth control
    
    Returns:
        PiMatchNetwork object with calculated component values
    """
    omega = 2 * np.pi * f_center
    
    # For Pi network, virtual resistance Rv < min(Zs, Zl)
    # Higher target_q = narrower bandwidth = lower Rv
    if target_q is None:
        # Default: moderate Q
        target_q = 5
    
    Rv = min(z_source, z_load) / (1 + target_q**2)
    Rv = max(Rv, 0.1)  # Practical minimum
    
    # Q of each L-section
    Q1 = np.sqrt(z_source / Rv - 1)
    Q2 = np.sqrt(z_load / Rv - 1)
    
    # Component reactances
    X_L1 = Q1 * Rv
    X_L2 = Q2 * Rv
    X_C1 = z_source / Q1
    X_C2 = z_load / Q2
    
    # Component values
    L_val = (X_L1 + X_L2) / omega
    C1_val = 1 / (omega * X_C1)
    C2_val = 1 / (omega * X_C2)
    
    # Create component objects with Q factors
    L = Inductor(L=L_val, Q=q_L, f_design=f_center)
    C1 = Capacitor(C=C1_val, Q=q_C, f_design=f_center)
    C2 = Capacitor(C=C2_val, Q=q_C, f_design=f_center)
    
    return PiMatchNetwork(L=L, C1=C1, C2=C2, 
                          z_source=z_source, z_load=z_load, 
                          f_center=f_center)


def create_pi_match_from_values(L_val: float, C1_val: float, C2_val: float,
                                 z_source: float = 50.0, z_load: float = 50.0,
                                 f_center: float = 2.45e9,
                                 q_L: float = 50, q_C: float = 100) -> PiMatchNetwork:
    """
    Create a Pi-match network from specific component values.
    
    Args:
        L_val: Inductance in Henries
        C1_val: Input capacitance in Farads
        C2_val: Output capacitance in Farads
        z_source: Source impedance (Ohms)
        z_load: Load impedance (Ohms)
        f_center: Center frequency (Hz)
        q_L: Inductor Q factor
        q_C: Capacitor Q factor
    
    Returns:
        PiMatchNetwork object
    """
    L = Inductor(L=L_val, Q=q_L, f_design=f_center)
    C1 = Capacitor(C=C1_val, Q=q_C, f_design=f_center)
    C2 = Capacitor(C=C2_val, Q=q_C, f_design=f_center)
    
    return PiMatchNetwork(L=L, C1=C1, C2=C2,
                          z_source=z_source, z_load=z_load,
                          f_center=f_center)


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Pi-Matching Network Module - Test")
    print("="*70)
    
    # Design parameters
    Z_ANT = 50      # Antenna impedance
    Z_RECT = 30     # Rectifier input impedance estimate
    F_CENTER = 2.45e9
    Q_L = 50        # Inductor Q
    Q_C = 100       # Capacitor Q
    
    # Design network
    print(f"\nDesigning Pi-match: {Z_ANT}Ω → {Z_RECT}Ω at {F_CENTER/1e9:.2f} GHz")
    pi_match = design_pi_match(Z_ANT, Z_RECT, F_CENTER, Q_L, Q_C)
    print(f"\n{pi_match}")
    
    # Analyze at center frequency
    print(f"\nPerformance at {F_CENTER/1e9:.3f} GHz:")
    print(f"  Return Loss:    {pi_match.return_loss_db(F_CENTER):.2f} dB")
    print(f"  Insertion Loss: {pi_match.insertion_loss_db(F_CENTER):.2f} dB")
    print(f"  Voltage Gain:   {pi_match.voltage_gain(F_CENTER):.3f}")
    
    # Bandwidth analysis
    print("\nBandwidth Analysis (-10 dB return loss):")
    bw_result = pi_match.analyze_bandwidth(return_loss_threshold=-10)
    print(f"  Center (actual): {bw_result['f_center_actual']/1e9:.4f} GHz")
    print(f"  Min Return Loss: {bw_result['return_loss_min_dB']:.2f} dB")
    print(f"  Bandwidth:       {bw_result['bandwidth_hz']/1e6:.1f} MHz")
    
    # Generate SPICE netlist
    print("\nSPICE Netlist Fragment:")
    print("-"*40)
    print(pi_match.generate_spice_netlist())
    print("-"*40)
