# 🧠 Distributed MNIST Recognition (MapReduce)

A distributed machine learning engine built from scratch to recognize handwritten digits (MNIST dataset). It uses a **MapReduce** architecture and **Python Multiprocessing** to parallelize training across CPU cores.

---

## ✨ Features

- **Custom Neural Network:** Built entirely with `NumPy` — no TensorFlow or PyTorch.
- **Distributed Training:** Master–Worker architecture using `multiprocessing` (MapReduce pattern).
- **Smart GUI:** Tkinter-based application for real-time digit testing.
  - Adaptive preprocessing based on ink density.
  - Center of Mass alignment (MNIST-style).
  - Real-time debug view.

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/ramt11n/Distributed-MNIST.git
cd Distributed-MNIST
```

### 2. Create a virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install numpy pillow
```

### 3. Download the MNIST dataset
Download the following files from Yann LeCun’s website and put them in the `data/` folder:

- train-images-idx3-ubyte.gz  
- train-labels-idx1-ubyte.gz  
- t10k-images-idx3-ubyte.gz  
- t10k-labels-idx1-ubyte.gz  

---

## 🚀 Usage

### 1. Train the model (Backend)
```bash
python3 main.py
```

### 2. Run the application (Frontend)
```bash
python3 app.py
```

---

## 🏗️ Project Architecture

| File | Description |
|------|-------------|
| src/neural_net.py | Matrix ops, forward/backward propagation. |
| src/worker.py | Worker process logic (Map phase). |
| src/mnist_loader.py | IDX file parsers. |
| main.py | Master process (Reduce phase). |
| app.py | GUI for drawing, preprocessing, prediction. |

---

Created by **Ramtin Neshat**
