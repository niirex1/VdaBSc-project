# VdaBSc: Smart Contract Vulnerability Detection

## Introduction

The security and reliability of smart contracts are critical issues in the blockchain ecosystem. Existing methods often employ machine learning techniques and static analysis, which have limitations, such as computational inefficiency and incomplete code analysis. 

## About VdaBSc

We introduce **VdaBSc**, a comprehensive approach incorporating dynamic analysis, real-time runtime batch normalization, data augmentation, N-grams, and a hybrid architecture combining BiLSTM, CNN, and the Attention Mechanism to address these challenges. 

### Feature Representation

Our feature representation technique employs N-grams and one-hot encoding, capturing sequential dependencies between opcodes and representing each opcode as a binary vector.

### Hybrid Architecture

The VdaBSc model is built on a robust hybrid architecture that leverages BiLSTM for capturing temporal dynamics, CNN for local feature extraction, and the Attention Mechanism for context understanding.

## Evaluation

VdaBSc has been rigorously evaluated against existing methods and state-of-the-art deep learning techniques. Our evaluations, benchmarked against these methods, reveal that VdaBSc demonstrates superior performance across key metrics such as accuracy, precision, recall, and F1-Score.

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python (version 3.7 or higher)
- Git (for cloning the repository)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/niirex1/VdaBSc-project.git
   ```

2. **Navigate to the Project Directory**

   ```bash
   cd VdaBSc-project
   ```

3. **Install Required Python Packages**

   ```bash
   pip install -r requirements.txt
   ```

### Running VdaBSc

Please refer to the `example_usage.ipynb` notebook for a detailed guide on how to use the model for smart contract vulnerability detection.

---

## Contributing

We welcome contributions from the research community. For guidelines on contributing, please refer to the [Contributing](CONTRIBUTING.md) documentation.

---

## License

### MIT License

VdaBSc is licensed under the MIT License. For the full license text, refer to the `LICENSE` file in the repository or visit [MIT License](https://opensource.org/licenses/MIT).

---

## Contact & Support

For any questions, feedback, or suggestions regarding the VdaBSc project, please reach out to the project maintainers:

- **Rexford Sosu**
  - Email: rexfordsosu@outlook.com
  - GitHub: [@rexfordsosu](https://github.com/niirex1)
  - LinkedIn: [Rexford's LinkedIn](https://www.linkedin.com/in/rexford-sosu-b4593b57/)

We appreciate your interest in the VdaBSc project and look forward to your contributions!
