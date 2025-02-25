# FedOSR Implementation with CIFAR and SVHN Datasets

This repository contains the implementation of **FedOSR (Federated Open Set Recognition)** using the **CIFAR** and **SVHN** datasets. The model is based on a **custom ResNet-18 architecture**.

## 🚀 Getting Started

### **Clone the Repository**
First, clone this repository and navigate into the project directory:
```sh
git clone https://github.com/mzhaks/FedOSR.git
cd FedOSR
```

### **Run The Project**
1. Pretraining the Model
```sh
python main.py --mode Pretrain
```
2. Finetuning the Model
```sh
python main.py --mode Finetune
```
