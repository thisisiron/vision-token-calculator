# Non-HF Model Tests

This directory contains tests for Vision Language Models that don't use HuggingFace processors.

## Adding Tests for a New Model

1. Copy `_template.py` to `test_<model_name>.py`
2. Update the configuration section:
   - `MODEL_NAME`: CLI model name (e.g., `"phi4-multimodal"`)
   - `ANALYST_CLASS_NAME`: Analyst class name (e.g., `"Phi4MultimodalAnalyst"`)
   - `BASIC_TEST_CASES`: Expected token counts for standard image sizes
   - `REQUIRED_KEYS_*`: Required output dictionary keys
   - `TOKEN_FORMULA_CASES`: Formula verification test cases
3. Update class names (replace `XxxModel` with your model name)
4. Implement model-specific tests in `TestXxxModelTokenFormula`

## Standard Test Class Structure

| Class | Purpose |
|-------|---------|
| `TestXxxModelBasic` | Basic token calculation and constants |
| `TestXxxModelRegistration` | Model registration and loading |
| `TestXxxModelProcessingModes` | Various image sizes and aspect ratios |
| `TestXxxModelOutputFormat` | Output dictionary required keys |
| `TestXxxModelTokenFormula` | Formula verification (parametrize) |
| `TestXxxModelVideo` | Video support (or NotImplementedError) |

## Running Tests

```bash
# Run all model tests
pytest tests/models/ -v

# Run specific model tests
pytest tests/models/test_phi4_multimodal.py -v

# Run with coverage
pytest tests/models/ --cov=vt_calculator.analysts
```
