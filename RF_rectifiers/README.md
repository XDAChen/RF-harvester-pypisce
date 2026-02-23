# RF Rectifier Simulations

SPICE-based simulations of RF rectifier circuits for 2.45 GHz WiFi energy harvesting with Pi-matching network optimization.

## Requirements

- Python 3.10+
- ngspice

## Setup

```bash
# Install ngspice
sudo apt install ngspice        # Linux
brew install ngspice            # macOS

# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
cd RF_rectifiers
python halfwave_rectifier.py
```

Output plots are saved to `temp_image/`.

## Analysis Suite

The halfwave rectifier script runs:

| Analysis | Output |
|----------|--------|
| Pi-match optimization | Optimal L, C1, C2 for impedance matching |
| WiFi OFDM spectrum (CommPy) | wifi_spectrum.png |
| Transient waveforms | halfwave_transient_waveforms.png |
| Matched vs Direct comparison | matched_rectifier_comparison.png |
| Harmonic analysis | halfwave_harmonics_*.png |
| Sensitivity (Vin, Cout) | halfwave_sens_combined.png |
| Frequency stability | halfwave_freq_stability.png |
| Monte Carlo (50 runs) | halfwave_mc_*.png |

## Files

| File | Description |
|------|-------------|
| halfwave_rectifier.py | Half-wave rectifier with full analysis suite |
| dickson_rectifier.py | 2-stage Dickson charge pump |
| pi_match.py | Pi-matching network module (non-ideal L/C with Q factors) |
| optimization.py | Optimization algorithms for Pi-match component values |
| utility.py | Simulation runners and plotting functions |
| diode_models.lib | RF Schottky diode SPICE models |
| how_to_code_netlist.txt | SPICE netlist reference guide |
| requirements.txt | Python dependencies |

## Pi-Matching Network

Transforms antenna impedance (50Ω) to rectifier input (~30Ω) for maximum power transfer.

**Topology (Low-Pass):**
```
                L (series)
            ┌────LLLL────RL────┐
            │                  │
IN o────────┼──────────────────┼────────o OUT
            │                  │
           ═══ C1             ═══ C2
            │                  │
           GND                GND
```

**Features:**
- Non-ideal components with Q-factor modeling (ESR)
- S-parameter analysis (return loss, insertion loss)
- Bandwidth analysis
- Optimization with constraints: return loss < -20 dB, bandwidth > 30 MHz

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| F_RF | 2.45 GHz | Operating frequency |
| V_RF_AMPLITUDE | 300 mV | Input amplitude |
| ANT_IMP | 50 Ω | Antenna impedance |
| RECT_IMP_EST | 30 Ω | Estimated rectifier input impedance |
| R_LOAD | 5 kΩ | Load resistance |
| C_IN, C_OUT | 100 pF | Capacitor values |
| CAP_Q | 30 | Rectifier capacitor Q |
| IND_Q | 50 | Pi-match inductor Q |
| PI_MATCH_CAP_Q | 100 | Pi-match capacitor Q |

## Diode Models

Available in diode_models.lib:

| Model | Description |
|-------|-------------|
| SMS7630 | Skyworks low-barrier Schottky (default) |
| HSMS2850 | Avago low-barrier Schottky |
| HSMS2860 | Avago medium-barrier Schottky |
| BAT15 | Infineon RF Schottky |

