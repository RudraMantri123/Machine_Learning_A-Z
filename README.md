# Machine Learning A-Z: Complete Dataset and Code Collection

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-A%20to%20Z-blue)
![Python](https://img.shields.io/badge/Python-3.x-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

This repository contains a comprehensive collection of machine learning algorithms, implementations, and datasets covering the complete spectrum from basic concepts to advanced deep learning techniques. All code is implemented in **Python** with Jupyter notebooks for interactive learning and standalone Python scripts for production use.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Content Overview](#content-overview)
- [How to Use This Repository](#how-to-use-this-repository)
- [Algorithm Categories](#algorithm-categories)
- [Datasets](#datasets)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository is based on the popular "Machine Learning A-Z" course and provides hands-on implementations of various machine learning algorithms. Each section includes:

- **Jupyter Notebooks** (`.ipynb`) for interactive learning and visualization
- **Python Scripts** (`.py`) for standalone execution
- **Sample Datasets** (`.csv`, `.tsv`) for testing and experimentation
- **Visualization Images** and results

## Repository Structure

```
Machine-Learning-A-Z-Codes-Datasets/
├── Part 1 - Data Preprocessing/
├── Part 2 - Regression/
├── Part 3 - Classification/
├── Part 4 - Clustering/
├── Part 5 - Association Rule Learning/
├── Part 6 - Reinforcement Learning/
├── Part 7 - Natural Language Processing/
├── Part 8 - Deep Learning/
├── Part 9 - Dimensionality Reduction/
└── Part 10 - Model Selection and Boosting/
```

## Prerequisites

Before running the code, ensure you have the following installed:

- **Python 3.7+**
- **Jupyter Notebook/Lab**
- **Essential Libraries** (see Installation section)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RudraMantri123/Machine_Learning_A-Z.git
   cd Machine_Learning_A-Z
   ```

2. **Install required packages**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
   ```

   Or using conda:
   ```bash
   conda install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

## Content Overview

### Part 1: Data Preprocessing
- **Section 2**: Data Preprocessing Tools
  - Importing datasets
  - Handling missing data
  - Encoding categorical variables
  - Feature scaling
  - Splitting datasets

### Part 2: Regression
- **Section 4**: Simple Linear Regression
- **Section 5**: Multiple Linear Regression
- **Section 6**: Polynomial Regression
- **Section 7**: Support Vector Regression (SVR)
- **Section 8**: Decision Tree Regression
- **Section 9**: Random Forest Regression

### Part 3: Classification
- **Section 14**: Logistic Regression
- **Section 15**: K-Nearest Neighbors (K-NN)
- **Section 16**: Support Vector Machine (SVM)
- **Section 17**: Kernel SVM
- **Section 18**: Naive Bayes
- **Section 19**: Decision Tree Classification
- **Section 20**: Random Forest Classification

### Part 4: Clustering
- **Section 24**: K-Means Clustering
- **Section 25**: Hierarchical Clustering

### Part 5: Association Rule Learning
- **Section 28**: Apriori Algorithm
- **Section 29**: Eclat Algorithm

### Part 6: Reinforcement Learning
- **Section 32**: Upper Confidence Bound (UCB)
- **Section 33**: Thompson Sampling

### Part 7: Natural Language Processing
- **Section 36**: Natural Language Processing
  - Text preprocessing
  - Bag of Words model
  - Sentiment analysis

### Part 8: Deep Learning
- **Section 39**: Artificial Neural Networks (ANN)
- **Section 40**: Convolutional Neural Networks (CNN)

### Part 9: Dimensionality Reduction
- **Section 43**: Principal Component Analysis (PCA)
- **Section 44**: Linear Discriminant Analysis (LDA)
- **Section 45**: Kernel PCA

### Part 10: Model Selection and Boosting
- **Section 48**: Model Selection
  - Grid Search
  - K-Fold Cross Validation
- **Section 49**: XGBoost

## How to Use This Repository

1. **Start with Data Preprocessing**: Begin with Part 1 to understand data handling basics
2. **Choose Your Algorithm**: Navigate to the relevant section based on your problem type
3. **Open Jupyter Notebook**: Use the `.ipynb` files for step-by-step learning
4. **Run Python Scripts**: Use `.py` files for quick execution
5. **Experiment with Datasets**: Modify parameters and try different datasets

## Algorithm Categories

### Supervised Learning
- **Regression**: Predicting continuous values
  - Linear, Polynomial, SVR, Decision Tree, Random Forest
- **Classification**: Predicting categorical values
  - Logistic Regression, K-NN, SVM, Naive Bayes, Decision Tree, Random Forest

### Unsupervised Learning
- **Clustering**: Grouping similar data points
  - K-Means, Hierarchical Clustering
- **Dimensionality Reduction**: Reducing feature dimensions
  - PCA, LDA, Kernel PCA
- **Association Rule Learning**: Finding relationships
  - Apriori, Eclat

### Advanced Topics
- **Reinforcement Learning**: Learning through interaction
  - UCB, Thompson Sampling
- **Deep Learning**: Neural networks
  - ANN, CNN
- **Natural Language Processing**: Text analysis and processing

## Datasets

This repository includes various datasets for different algorithms:

- **Salary_Data.csv**: For linear regression
- **50_Startups.csv**: For multiple regression
- **Position_Salaries.csv**: For polynomial and advanced regression
- **Social_Network_Ads.csv**: For classification algorithms
- **Mall_Customers.csv**: For clustering algorithms
- **Market_Basket_Optimisation.csv**: For association rule learning
- **Ads_CTR_Optimisation.csv**: For reinforcement learning
- **Restaurant_Reviews.tsv**: For NLP
- **Churn_Modelling.csv**: For deep learning
- **Wine.csv**: For dimensionality reduction

## Key Features

- **Complete Python Implementation**: All algorithms implemented from scratch
- **Jupyter Notebooks**: Interactive learning environment
- **Comprehensive Datasets**: Ready-to-use sample data
- **Visualizations**: Charts and plots for better understanding
- **Best Practices**: Clean, documented, and efficient code
- **Production Ready**: Standalone Python scripts
- **Beginner Friendly**: Step-by-step implementations

## Technologies Used

- **Python 3.x**
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning
- **Jupyter**: Interactive development

## Usage Examples

### Running a Jupyter Notebook
```bash
jupyter notebook "Part 2 - Regression/Section 4 - Simple Linear Regression/Python/simple_linear_regression.ipynb"
```

### Executing Python Script
```bash
python "Part 3 - Classification/Section 14 - Logistic Regression/Python/logistic_regression.py"
```

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

If you find this repository helpful, please consider giving it a star!

For questions or support, please open an issue in the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the Machine Learning A-Z course
- Inspired by the machine learning community
- Thanks to all contributors and the open-source community

---

**Happy Learning!**

*"The best way to learn machine learning is by doing machine learning."*
