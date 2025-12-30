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

```bash
# Clone the repository
git clone https://github.com/dataguirre/reliefNET-GNN-nomad.git
cd reliefNET-GNN-nomad

# Install dependencies
uv sync
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

## 📊 Data

Transportation network data and related files are stored in the `data/` directory.

## 📈 Results

Experimental results and model outputs are stored in the `results/` directory.

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
  author={Aguirre Salamanca, Daniel},
  year={2025},
  school={Universidad de los Andes},
  url={https://hdl.handle.net/1992/76275}
}

@article{aguirre2026ai,
  title={AI for Aid: Using ReliefNET-GNN to Enhance Transportation Resilience in Disaster Response},
  author={Aguirre Salamanca, Daniel and Cardozo, Nicol{\'a}s and Herrera, Andrea},
  year={2026}
}
```

## 📝 License

This project's license information is specified in the `pyproject.toml` file.

## 👥 Authors

- [@dataguirre](https://github.com/dataguirre)
- [@mvrobles](https://github.com/mvrobles)

## 🙏 Acknowledgments

- Universidad de los Andes
- Research collaborators: Nicolás Cardozo, Andrea Herrera
- Transportation resilience research community
- Humanitarian logistics and disaster response organizations

## 📧 Contact

For questions or feedback, please open an issue on the GitHub repository.

## 🔗 References

1. Aguirre Salamanca, D. (2025). *Resilient routes: leveraging deep learning to identify critical transportation links for humanitarian response*. Universidad de los Andes. https://hdl.handle.net/1992/76275

2. Aguirre Salamanca, D., Cardozo, N., Herrera, A. (2026). *AI for Aid: Using ReliefNET-GNN to Enhance Transportation Resilience in Disaster Response*

---

**Note:** Additional documentation can be found in the `docs/` directory.
