# RF Rectifier Simulations

Transient simulations of RF rectifier circuits at 2.45 GHz using ngspice.

## Requirements

- Python 3.10+
- ngspice (circuit simulator)

## Setup

1. Install ngspice:
   ```
   # Linux (Ubuntu/Debian)
   sudo apt install ngspice
   
   # macOS
   brew install ngspice
   
   # Windows
   # Download installer from https://ngspice.sourceforge.io/download.html
   # Add ngspice to system PATH after installation
   ```

2. Create virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate      # Linux/macOS
   .venv\Scripts\activate         # Windows
   ```

3. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run simulations from the RF_rectifiers directory:

```
cd RF_rectifiers
python halfwave_rectifier.py
python dickson_rectifier.py
```

Each script will:
- Print the generated SPICE netlist
- Run transient simulation
- Display results summary
- Save plot as PNG

## Files

| File | Description |
|------|-------------|
| halfwave_rectifier.py | Single-diode half-wave rectifier |
| dickson_rectifier.py | 2-stage Dickson charge pump |
| utility.py | Shared simulation and plotting functions |
| diode_models.lib | SPICE models for RF Schottky diodes |
| requirements.txt | Python package dependencies |

## Configuration

Edit parameters at the top of each rectifier script:

- `F_RF` - Operating frequency (default: 2.45 GHz)
- `V_RF_AMPLITUDE` - Input amplitude in Volts
- `R_SOURCE` - Source impedance (default: 50 ohms)
- `C_IN`, `C_OUT`, `C_STAGE` - Capacitor values
- `R_LOAD` - Load resistance
- `CAP_Q` - Capacitor Q factor for ESR modeling
- `DIODE_MODEL_NAME` - Diode model to use

## Diode Models

Available models in diode_models.lib:

| Model | Description |
|-------|-------------|
| SMS7630 | Skyworks low-barrier Schottky (default) |
| HSMS2850 | Avago low-barrier Schottky |
| HSMS2860 | Avago medium-barrier Schottky |
| BAT15 | Infineon RF Schottky |
| DEFAULT_SCHOTTKY | Generic RF Schottky |
| IDEAL_SCHOTTKY | Near-ideal (for comparison only) |

To change diode model, edit `DIODE_MODEL_NAME` in the rectifier script.

