# Bayesian Gaussian Classifier

## Overview
The **Bayesian Gaussian Classifier** is a Python-based machine learning project that classifies data points using **Bayesian decision theory**. It estimates class priors, means, and covariances to compute posterior probabilities and assigns class labels accordingly. The project includes visualizations for decision boundaries.

## Features
- **Loads dataset from CSV files** (`hw02_data_points.csv`, `hw02_class_labels.csv`).
- **Estimates prior probabilities** for each class.
- **Computes class means and covariance matrices** for Gaussian classification.
- **Calculates Bayesian decision scores** to classify points.
- **Generates a confusion matrix** to evaluate model performance.
- **Visualizes decision boundaries** and saves results as PDFs.

## Installation
Ensure you have **Python 3.x** installed along with the required libraries.

### Steps:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/bayesian-gaussian-classifier.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd bayesian-gaussian-classifier
   ```
3. **Install dependencies:**
   ```bash
   pip install numpy matplotlib scipy pandas
   ```
4. **Run the script:**
   ```bash
   python beqHW.py
   ```

## Usage
1. **Modify `hw02_data_points.csv` and `hw02_class_labels.csv`** to use your own dataset.
2. **Run `beqHW.py`** to compute class probabilities, means, and covariances.
3. **Check output files**:
   - `hw02_result_different_covariances.pdf` (decision boundary using different covariances)
   - `hw02_result_shared_covariance.pdf` (decision boundary using shared covariance)

## Example Output
```
Class Priors:
[0.5 0.5]

Class Means:
[[ 2.1  3.5]
 [ 4.2  1.7]]

Confusion Matrix:
[[45  5]
 [ 3 47]]
```

## License
This project is licensed under the **MIT License**.

## Contributions
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`feature-new-feature`).
3. Commit and push your changes.
4. Open a pull request.

## Contact
For any questions or support, please open an issue on GitHub.

