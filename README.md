# ReliefNET-GNN-nomad

A Graph Neural Network (GNN) implementation for identifying critical transportation links in humanitarian disaster response scenarios using real-world transportation networks.

## 📋 Overview

This project extends previous research on transportation resilience by applying deep learning methods to real transportation networks for humanitarian response optimization. ReliefNET-GNN leverages graph neural networks to identify critical links in transportation infrastructure that are essential for effective disaster relief operations.

### Research Background

This work builds upon and extends:

- **Aguirre Salamanca, D.** (2025). *Resilient routes: leveraging deep learning to identify critical transportation links for humanitarian response*. Universidad de los Andes. [Available at: https://hdl.handle.net/1992/76275](https://hdl.handle.net/1992/76275)

- **Aguirre Salamanca, D., Cardozo, N., Herrera, A.** (2026). *AI for Aid: Using ReliefNET-GNN to Enhance Transportation Resilience in Disaster Response*

The implementation leverages modern Python tooling with `uv` for fast, reliable dependency management and reproducibility.

### Key Features

- **Real-world Transportation Networks**: Analysis of actual transportation infrastructure
- **Critical Link Identification**: Deep learning-based detection of essential routes for disaster response
- **Humanitarian Focus**: Optimization specifically designed for relief operations and emergency scenarios
- **Graph Neural Networks**: State-of-the-art GNN architecture for network analysis
- **Reproducible Research**: Built with `uv` for consistent dependency management across environments

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

2. Install pre-commit hooks (if configured):
   ```bash
   pre-commit install
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

Transportation network data files should be placed in the `data/` directory. The project works with:

- Real-world road networks
- Transportation infrastructure graphs
- Historical disaster response routes
- Critical link annotations

Data formats may include graph representations (edge lists, adjacency matrices) and associated node/edge features relevant to humanitarian response scenarios.

## 📈 Results

Model outputs, critical link predictions, network visualizations, and experimental results are stored in the `results/` directory. This may include:

- Identified critical transportation links
- Network resilience metrics
- Comparative analysis with baseline methods
- Visualization of vulnerable infrastructure
- Performance metrics for disaster response scenarios

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{aguirre2025resilient,
  title={Resilient routes: leveraging deep learning to identify critical transportation links for humanitarian response},
  author={Aguirre Salamanca, David},
  year={2025},
  school={Universidad de los Andes},
  url={https://hdl.handle.net/1992/76275}
}

@article{aguirre2026ai,
  title={AI for Aid: Using ReliefNET-GNN to Enhance Transportation Resilience in Disaster Response},
  author={Aguirre Salamanca, David and Cardozo, Nicol{\'a}s and Herrera, Alberto},
  year={2026}
}
```

## 📝 License

This project's license information is specified in the `pyproject.toml` file.

## 👥 Authors

- [@dataguirre](https://github.com/dataguirre)

## 🙏 Acknowledgments

- Universidad de los Andes
- Research collaborators: Nicolás Cardozo, Alberto Herrera
- Transportation resilience research community
- Humanitarian logistics and disaster response organizations
- Graph Neural Network research community

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

## 🔗 References

### Primary Research

1. Aguirre Salamanca, D. (2025). *Resilient routes: leveraging deep learning to identify critical transportation links for humanitarian response*. Universidad de los Andes. https://hdl.handle.net/1992/76275

2. Aguirre Salamanca, D., Cardozo, N., Herrera, A. (2026). *AI for Aid: Using ReliefNET-GNN to Enhance Transportation Resilience in Disaster Response*

### Additional Documentation

Additional technical documentation and methodology details can be found in the `docs/` directory.

---

**Note:** This README provides a general structure. Please refer to the actual implementation in the `src/` directory and any documentation in `docs/` for specific usage details and methodology.
