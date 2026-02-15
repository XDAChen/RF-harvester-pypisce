# RF Rectifier Simulations

SPICE-based transient simulations of RF rectifier circuits for 2.45 GHz WiFi energy harvesting.

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
| WiFi OFDM spectrum (CommPy) | wifi_spectrum.png |
| Transient waveforms | halfwave_transient_waveforms.png |
| Harmonic analysis | halfwave_harmonics_*.png |
| Sensitivity (Vin, Cout) | halfwave_sens_combined.png |
| Frequency stability | halfwave_freq_stability.png |
| Monte Carlo (50 runs) | halfwave_mc_*.png |

## Files

| File | Description |
|------|-------------|
| halfwave_rectifier.py | Half-wave rectifier analysis |
| dickson_rectifier.py | 2-stage Dickson charge pump |
| utility.py | Simulation and plotting functions |
| diode_models.lib | RF Schottky diode SPICE models |
| requirements.txt | Python dependencies |

## Parameters

Edit at top of rectifier scripts:

| Parameter | Default | Description |
|-----------|---------|-------------|
| F_RF | 2.45 GHz | Operating frequency |
| V_RF_AMPLITUDE | 300 mV | Input amplitude |
| R_LOAD | 5 kohm | Load resistance |
| C_IN, C_OUT | 100 pF | Capacitor values |
| CAP_Q | 30 | Capacitor Q factor |

## Diode Models

Available in diode_models.lib:

| Model | Description |
|-------|-------------|
| SMS7630 | Skyworks low-barrier Schottky (default) |
| HSMS2850 | Avago low-barrier Schottky |
| HSMS2860 | Avago medium-barrier Schottky |
| BAT15 | Infineon RF Schottky |

