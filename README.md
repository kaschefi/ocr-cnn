# MNIST CNN Classifier & Visualizer 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A robust Convolutional Neural Network (CNN) implementation for the MNIST handwritten digit dataset using Keras/TensorFlow.

Beyond standard training, this project focuses on **interpretability**. It automatically generates detailed visualizations of the model's inner workings, including feature maps, weight filters, and training dynamics, saving them to organized directories.

##  Key Features

* **Deep CNN Architecture**: Multi-stage convolutional blocks with Max Pooling and Dropout regularization.
* **Advanced Learning Rate Schedules**: Includes implementations for **Step Decay** (active), Cosine Decay, and Exponential Decay.
* **Comprehensive Visualization Pipeline**:
    * **Loss Curves**: Training vs. Validation loss over epochs.
    * **Confusion Matrix**: Visual heatmap of classification errors.
    * **Feature Maps**: See what the network "sees" (activation layers) for specific test images.
    * **Weight Filters**: Visualize the actual kernels learned by the convolutional layers.
* **Automated Reporting**: Generates text files with Precision, Recall, and F1-scores.

## 🛠️ Installation

Ensure you have Python installed. Install the required dependencies:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

##  Usage

1.  Clone the repository.
2.  Run the script:

```bash
python 2025 Exercise 3 - Handwritten OCR CNN.py
```

3.  The script will create a new directory (named based on your hyperparameters) containing all logs and plots.

##  Configuration

The code is designed to be easily configurable. You can adjust the following variables in **Section 4** of the script:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `n_epochs` | 15 | Number of training epochs |
| `batch_size` | 128 | Size of data batches |
| `dropout` | 0.4 | Dropout rate for regularization |
| `n_cnn1planes` | 15 | Number of filters in the first layer |
| `n_cnn1kernel` | 3 | Kernel size (3x3) |

###  Switching Optimizers & Schedulers

The code includes pre-written blocks for different learning rate strategies in **Section 6**.

**Current Active Method: Step Decay**
Drops the learning rate at specific epoch boundaries (Epoch 5 and 10).

**To use Cosine Decay:**
1. Comment out the `PiecewiseConstantDecay` block.
2. Uncomment the `CosineDecay` block:
```python
# learning_rate = CosineDecay(
#    initial_learning_rate=initial_lr,
#    decay_steps=total_decay_steps
# )
```

**To use Exponential Decay:**
1. Uncomment the `ExponentialDecay` block.
2. Uncomment `optimizer = SGD(...)` or `Adam(...)` accordingly.

##  Output Structure

Every run creates a folder named after the configuration (e.g., `CNN_Handwritten_OCR_CNN15_KERNEL3...`). Inside, you will find:

* `..._loss.png`: Graph of training performance.
* `..._confusion_matrix.png`: Heatmap of predictions vs actual labels.
* `..._initial_weights.png`: Visualization of weights *before* training.
* `..._weights.png`: Visualization of weights *after* training.
* `..._activations_test_image_X.png`: Visualization of how a specific digit activates the neural network layers.
* `..._classification_report.txt`: Detailed metrics text file.

##  Example Output

* **Confusion Matrix**: Quickly identify if 5s are being confused with 3s.
* **Activations**: Observe how early layers detect edges, while deeper layers detect shapes and loops.
