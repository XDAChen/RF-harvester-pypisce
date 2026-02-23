# RF Energy Harvester - Matching Network Reference

Reference SPICE netlists and standalone simulation for Pi-matching network analysis at 2.45 GHz.

> **Note:** For the full simulation suite with optimization, see `../RF_rectifiers/`

## Overview

This folder contains reference SPICE netlists demonstrating RF energy harvesting with and without impedance matching.

### Circuit Topology

```
                    Pi-Matching Network (Low-Pass)
                         L (series)
                    ┌────LLLL────RL────┐
                    │                  │
Vsrc ── Rs ─────────┼──────────────────┼───|>|──┬── Vout
                    │                  │    D   │
                   ═══ C1             ═══ C2   ├── Cout
                    │                  │       └── Rload ── GND
                   GND                GND
```

## Design Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| f₀ | 2.45 GHz | Center frequency (ISM band) |
| Z_source | 50 Ω | Antenna impedance |
| Z_load | 30 Ω | Rectifier input impedance |

### Pi-Matching Network Components

| Component | Description |
|-----------|-------------|
| L | Series inductor with ESR (Q-dependent) |
| C1 | Input shunt capacitor |
| C2 | Output shunt capacitor |

## Files

| File | Description |
|------|-------------|
| harvester_matched.spice | SPICE netlist with Pi-matching network |
| harvester_direct.spice | SPICE netlist without matching |
| rf_professional_simulation.py | Standalone Python simulation |

## Usage

### SPICE Simulation

```bash
ngspice harvester_matched.spice
ngspice harvester_direct.spice
```

### Full Analysis Suite

For optimization, Monte Carlo, and comprehensive analysis:

```bash
cd ../RF_rectifiers
python halfwave_rectifier.py
```

## Theory

The Pi-matching network transforms the antenna impedance (50Ω) to match the rectifier input (~30Ω), maximizing power transfer. Non-ideal components with Q factors model realistic inductor ESR and capacitor losses.
