# reliefNET-GNN-nomad

A Graph Neural Network (GNN) implementation for relief network analysis and optimization using the NOMAD framework.

## 📋 Overview

This project implements a graph neural network approach for analyzing and optimizing relief networks. The implementation leverages modern Python tooling with `uv` for fast, reliable dependency management.

## 🚀 Quick Start

### Prerequisites

- Python 3.x (see `.python-version` for exact version)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/dataguirre/reliefNET-GNN-nomad.git
   cd reliefNET-GNN-nomad
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```

   This will install all dependencies specified in `pyproject.toml` and create a virtual environment.

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

## 📁 Project Structure

```
reliefNET-GNN-nomad/
├── data/              # Dataset storage and data files
├── docs/              # Documentation files
├── results/           # Experimental results and outputs
├── scripts/           # Utility and execution scripts
├── src/               # Source code for the GNN implementation
├── tests/             # Unit and integration tests
├── .gitignore         # Git ignore patterns
├── .python-version    # Python version specification
├── pyproject.toml     # Project metadata and dependencies
├── uv.lock            # Locked dependency versions
└── README.md          # This file
```

## 🔧 Usage

### Running the Model

```bash
# Example command (adjust based on actual implementation)
python -m src.main
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Running Scripts

```bash
# Navigate to scripts directory and run utilities
python scripts/script_name.py
```

## 🧪 Development

### Setting up Development Environment

1. Install development dependencies:
   ```bash
   uv sync --all-extras
   ```

### Code Style

This project follows Python best practices. Ensure your code:
- Follows PEP 8 guidelines
- Includes docstrings for functions and classes
- Has appropriate type hints where applicable

### Running Linters

```bash
# Format code
black src/ tests/

# Check types
mypy src/

# Lint code
ruff check src/ tests/
```

## 📊 Data

Data files should be placed in the `data/` directory. The specific format and structure depend on the relief network datasets being used.

## 📈 Results

Experimental results, model outputs, and visualizations are stored in the `results/` directory.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project's license information is specified in the `pyproject.toml` file.

## 👥 Authors

- [@dataguirre](https://github.com/dataguirre)

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

## 🔗 References

Additional documentation can be found in the `docs/` directory.

---

**Note:** This README provides a general structure. Please refer to the actual implementation in the `src/` directory and any documentation in `docs/` for specific usage details and methodology.
