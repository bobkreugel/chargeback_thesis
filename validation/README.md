# Validation Scripts

This folder contains validation scripts to verify that the fraud pattern implementations correctly match their theoretical specifications from the thesis.

## Scripts

### `validate_bin_attack_multiple_runs.py`
- **Purpose**: Validates BIN attack pattern implementations
- **Tests**: Card count per pattern, transaction amounts, time windows, chargeback rates, merchant reuse

### `validate_serial_chargeback_multiple_runs.py`
- **Purpose**: Validates serial chargeback pattern implementations
- **Tests**: Transaction count per pattern, time windows, chargeback rates, merchant reuse patterns
- **Usage**: `python validate_serial_chargeback_multiple_runs.py`

### `validate_friendly_fraud_multiple_runs.py`
- **Purpose**: Validates friendly fraud pattern implementations  
- **Tests**: Legitimate vs fraudulent phases, transaction timing, chargeback delays, merchant spreading
- **Usage**: `python validate_friendly_fraud_multiple_runs.py`

### `validate_geographic_mismatch.py`
- **Purpose**: Validates geographic mismatch pattern implementations
- **Tests**: Location diversity, transaction timing, chargeback rates, geographic spread
- **Usage**: `python validate_geographic_mismatch.py`

## Running Validation

To run all validation scripts:

```bash
# From the Final_code directory
cd validation

# Run individual validations
python validate_bin_attack_multiple_runs.py
python validate_serial_chargeback_multiple_runs.py
python validate_friendly_fraud_multiple_runs.py
python validate_geographic_mismatch.py
```

Each script will:
1. Generate multiple datasets with fraud patterns
2. Analyze pattern compliance with configuration parameters
3. Create validation plots showing parameter distributions
4. Save results and visualizations

## Output

Each script generates:
- Console output with validation statistics
- PNG plots showing parameter distributions vs expected ranges
- Detailed analysis of pattern compliance

## Requirements

- All validation scripts require the main `src` package to be available
- Matplotlib, NumPy, Pandas, and Seaborn for visualization
- The scripts automatically adjust import paths when run from this directory 