# 🧠 Distributed MNIST Recognition (MapReduce)

A distributed machine learning engine built from scratch to recognize handwritten digits (MNIST dataset). It utilizes **MapReduce** architecture and **Python Multiprocessing** to parallelize the training process across CPU cores.

## ✨ Features

* **Custom Neural Network:** Built entirely with `NumPy`. No high-level frameworks like TensorFlow or PyTorch used.
* **Distributed Training:** Implements a Master-Worker architecture using Python's `multiprocessing` to calculate gradients in parallel (MapReduce pattern).
* **Smart GUI:** A Tkinter-based application for real-time testing.
    * **Adaptive Preprocessing:** Automatically detects ink density (thick marker vs. thin pen) and applies smart erosion/dilation algorithms.
    * **Center of Mass Alignment:** Replicates MNIST's mathematical centering logic for high accuracy.
    * **Real-time Debug View:** Shows exactly what the AI sees before prediction.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ramt11n/Distributed-MNIST.git](https://github.com/ramt11n/Distributed-MNIST.git)
    cd Distributed-MNIST
    ```

2.  **Create a virtual environment & install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install numpy pillow
    ```

3.  **Download Data:**
    Because the dataset is binary, it is not included in the repo. Download the 4 files below from [Yann LeCun's MNIST website](http://yann.lecun.com/exdb/mnist/) and place them in the `data/` folder:
    * `train-images-idx3-ubyte.gz`
    * `train-labels-idx1-ubyte.gz`
    * `t10k-images-idx3-ubyte.gz`
    * `t10k-labels-idx1-ubyte.gz`

## 🚀 Usage

### 1. Train the Model (The Backend)
Run the master node to start distributed training. This uses your CPU cores to train the model and generates the `my_model.npz` file.

```bash
python3 main.py
2. Run the Application (The Frontend)Launch the GUI to draw digits or upload images for prediction.Bashpython3 app.py
🏗️ Project ArchitectureFileDescriptionsrc/neural_net.pyThe mathematical brain (Matrix multiplication, Forward/Backward prop).src/worker.pyThe worker process logic (Map phase).src/mnist_loader.pyParsers for reading binary IDX files.main.pyThe Master process. Splits data, manages workers, and updates weights (Reduce phase).app.pyThe GUI. Handles drawing, image processing, skeletonization, and prediction.Created by Ramtin Neshat