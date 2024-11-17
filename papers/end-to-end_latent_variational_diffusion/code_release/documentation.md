# ***End-to-End Latent Variational Diffusion Model***

## I. **How to run the code?**

First go into the directory that the `train_baseline.py` and `train_lvd.py` files.

### **Baseline**

`python train_baseline.py <model_type> <options_file> <training_file>`

- `model_types` specifies which model to train
    - `cinn` for conditional invertible neural network
    - `cvae` for traditional conditional variational autoencoder
    - `vae` for the variational diffusion models

### **VLD (Proposed model)**

`python train_lvd.py <options file> <training file>`

The command above will call the `train_lvd.py` using the an options file and training files as command-line arguments. 
Greater detail about the functionality will be documented in section *II. Python Files and Pipeline*

## II. **Python Files and Pipeline**

### **Baseline Models**

### **VLD (Proposed) Model**

**Necessary Files**

`train_lvd.py`: This particular file acts as the 

**Pipeline**

### **Libraries Explanation**

`omegaconf`: handles merging configurations from multiple sources
 

