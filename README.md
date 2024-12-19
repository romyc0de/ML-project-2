# BERT Project Dependencies and Execution Guide

## Libraries and Dependencies
To ensure smooth execution and reproducibility of your BERT-based project, install the following libraries and tools with their respective versions:

### Core Dependencies
- **Python (3.8 or later)**
  - Install from [python.org](https://www.python.org/downloads/).

- **PyTorch (1.10.0 or later)**
  - Install using:
    ```bash
    pip install torch torchvision torchaudio
    ```

- **Transformers Library by Hugging Face**
  - Provides pre-trained BERT models and utilities.
  - Install using:
    ```bash
    pip install transformers
    ```

- **Datasets Library by Hugging Face**
  - For efficient dataset handling.
  - Install using:
    ```bash
    pip install datasets
    ```

- **Scikit-learn (1.0.2 or later)**
  - For evaluation metrics and preprocessing.
  - Install using:
    ```bash
    pip install scikit-learn
    ```

- **Pandas and NumPy**
  - For data handling and numerical computations.
  - Install using:
    ```bash
    pip install pandas numpy
    ```

### Additional Libraries
- **Matplotlib and Seaborn** (optional for visualization)
  - Install using:
    ```bash
    pip install matplotlib seaborn
    ```

- **TQDM**
  - For progress bars in training and evaluation loops.
  - Install using:
    ```bash
    pip install tqdm
    ```

- **PyYAML**
  - For handling configuration files.
  - Install using:
    ```bash
    pip install pyyaml
    ```

- **torchsummary** (optional, for model summaries)
  - Install using:
    ```bash
    pip install torchsummary
    ```

### Optional Tools
- **Jupyter Notebook or JupyterLab**
  - For interactive experimentation.
  - Install using:
    ```bash
    pip install notebook
    pip install jupyterlab
    ```

- **TensorFlow** (optional, for TensorFlow-specific features)
  - Install using:
    ```bash
    pip install tensorflow
    ```

### GPU Acceleration
- **CUDA Toolkit (11.3 or later)**
  - Install the correct version of PyTorch compatible with your CUDA version from the [PyTorch Installation Guide](https://pytorch.org/get-started/locally/).

- **Metal Performance Shaders (MPS)**
  - For macOS users, MPS can be utilized to accelerate embedding generation using the BERT model on Apple Silicon devices. Ensure you are using a PyTorch version that supports MPS. Activate MPS support in PyTorch as follows:
    ```python
    import torch
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    ```

---

## Execution Steps

### Option 1 (Tested but Not Recommended)
1. **Run `BERT.py`**
   - This script prepares or initializes the base BERT model for training or fine-tuning.

2. **Execute `Pipeline-BERT-logistic.py` or `Pipeline-BERT-nn.py`**
   - Choose one of these scripts to run your pipeline with the pre-trained BERT model for the specific task.

### Option 2 (Best Results)
- **Execute `run.py`**
  - This script fine-tunes the BERT model and is designed to deliver the best results. It is self-contained and can be executed independently.

---

## Recommended Environment Setup

### Virtual Environment
Create a virtual environment to manage dependencies:
```bash
python3 -m venv bert_env
source bert_env/bin/activate  # For Linux/Mac
bert_env\Scripts\activate  # For Windows
```

### Install Dependencies from `requirements.txt`
Create a `requirements.txt` file with the following content:
```plaintext
torch>=1.10.0
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
pyyaml
tensorflow
```
Install using:
```bash
pip install -r requirements.txt
```

---

## Dataset Preparation
Ensure the following dataset files are correctly formatted and placed where the scripts expect them:
- `test_data.txt`
- `train_pos.txt`
- `train_neg.txt`
- `train_pos_full.txt`
- `train_neg_full.txt`

---

## Execution Time Warning
**Warning:** Training or fine-tuning BERT models as well as generating the embeddings can be computationally intensive and may take several hours to days depending on the size of your dataset, the complexity of your model, and the hardware you are using. It is highly recommended to use a GPU for faster execution. Ensure you have sufficient time and resources before starting the execution.

---

## Notes
- **Reproducibility**: Ensure all dependencies are correctly installed, and provide any pre-trained models or additional files required by the scripts.
- **CUDA Compatibility**: If using a GPU, verify the installed PyTorch version supports your CUDA version.
- **MPS Compatibility**: For Apple Silicon devices, use MPS for accelerated embedding generation and processing.
- **Documentation**: Include a `README.md` file describing your project structure and how to run each script.

---

## Example Commands
- Run `BERT.py`:
  ```bash
  python BERT.py
  ```

- Run `Pipeline-BERT.py`:
  ```bash
  python Pipeline-BERT.py
  ```

- Run `Pipeline-BERT-nn.py`:
  ```bash
  python Pipeline-BERT-nn.py
  ```

- Run `run.py`:
  ```bash
  python run.py
  ```
