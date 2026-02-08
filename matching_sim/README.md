# RF Energy Harvester Simulation

Python simulation of an RF energy harvesting circuit with a Pi-matching network operating at 2.45 GHz.

## Overview

This project simulates a half-wave rectifier circuit for RF energy harvesting, comparing performance with and without an impedance matching network.

### Circuit Topology

```
                    Pi-Matching Network
    ┌─────────────────────────────────────────┐
    │                                         │
Vsrc ──┬── Rs ──┬── C1 ──┬── L ── RL ──┬── C2 ──┬──|>|──┬── Vout
       │        │        │             │        │   D   │
      GND      GND      GND           GND      GND     ├── Cout
                                                       └── Rout ── GND
```

## Design Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| f₀ | 2.45 GHz | Center frequency (ISM band) |
| Z₀ | 50 Ω | Source impedance (antenna) |
| Z_L | 30 Ω | Load impedance |
| BW | 20 MHz | Bandwidth |
| Q_loaded | 122.5 | Loaded quality factor |

### Pi-Matching Network Components

| Component | Value | Notes |
|-----------|-------|-------|
| L | 1.591 nH | Series inductor (Q=50) |
| C1 | 4.244 pF | Input shunt capacitor |
| C2 | 5.305 pF | Output shunt capacitor |

### Diode Model (SMS7630-like Schottky)

| Parameter | Value |
|-----------|-------|
| Is | 5 µA |
| n | 1.05 |
| Rs | 20 Ω |
| Cj0 | 0.18 pF |

### Rectifier Output Stage

| Component | Value |
|-----------|-------|
| C_out | 100 pF |
| R_out | 10 kΩ |
| τ | 1.00 µs |

## Performance Results

| Input Power | With Matching | Without Matching | Improvement |
|-------------|---------------|------------------|-------------|
| 0 dBm | 50.29 mV | 25.65 mV | +96% |
| -10 dBm | 5.64 mV | 2.48 mV | +127% |
| -20 dBm | 0.89 mV | 0.53 mV | +68% |

## Files

| File | Description |
|------|-------------|
| `rf_professional_simulation.py` | Main Python simulation script |
| `harvester_matched.spice` | SPICE netlist with Pi-matching network |
| `harvester_direct.spice` | SPICE netlist without matching (direct connection) |
| `transient_0dBm.png` | Transient response plot at 0 dBm |
| `transient_-10dBm.png` | Transient response plot at -10 dBm |

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Usage

### Python Simulation

```bash
cd rf_energy_harvester
source ../venv/bin/activate  # if using virtual environment
python rf_professional_simulation.py
```

### SPICE Simulation (optional)

```bash
ngspice harvester_matched.spice
ngspice harvester_direct.spice
```

## Output Plots

The simulation generates two publication-ready plots:
- **transient_0dBm.png**: DC output voltage transient at 0 dBm input power
- **transient_-10dBm.png**: DC output voltage transient at -10 dBm input power

## Theory

The Pi-matching network transforms the source impedance (50Ω) to the load impedance (30Ω) while providing bandpass filtering centered at 2.45 GHz. This impedance matching maximizes power transfer from the antenna to the rectifier, significantly improving harvested DC voltage.

The matching network is designed using two back-to-back L-sections through a virtual resistance, achieving a high loaded Q for narrow bandwidth and selectivity.

## License

MIT License
