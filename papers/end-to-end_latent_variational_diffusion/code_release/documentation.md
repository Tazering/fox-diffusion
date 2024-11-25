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

What happens when you call `python train_lvd.py <options file> <training file>`?

In `train_lvd.py`, the function 
```python
def train(options_file: str,
    training_file: str,
    checkpoint_file: str,
    start_batch: int,
    name: str,
    weights_file)
```

is called and randomly initializes the values for setup. During the initial setup, the code uses the `training_file` and converts it into a `Dataset` class while the `options_files` is used for configuring the model. The `option_files` can be seen in */options/finetune.yaml* or */options/train.yaml*. 

```python
# initialize cuda
    jax.random.normal(jax.random.PRNGKey(0))

    # create the dataset
    print("Loading Data")
    dataset = Dataset(training_file, weights_file=weights_file)

    # setup the configuration
    config = Config(
        **OmegaConf.load(options_file),
        parton_dim=dataset.parton_dim,
        detector_dim=dataset.detector_dim,
        met_dim=dataset.met_dim
    )

    # create a dataloader from a dataset
    dataloader = dataset.create_dataloader(config.batch_size)
    single_device_batch = jax.tree_map(lambda x: x[0], next(dataloader))
```

The next step is creating the latent variational diffusion model and setting up its optimizers
```python
    # creates the variational diffusion model
    variation_diffusion_model, noise_scheduler, vdm_step, gamma_step = create_model(
        config)

    # make optimizers
    vdm_optimizer = make_optimizer(
        config.learning_rate, config.gradient_clipping)
    
    # gamma optimizer
    gamma_optimizer = make_optimizer(
        config.learning_rate, config.gradient_clipping)
```

Then, the program will check if a checkpoint file exists and initialize the model using the created diffusion model and optimizers described above.

```python
    # Initialize Model on GPU 0
    # -------------------------------------------------------------------------
    print("Initializing Model")
    # initialize using pseudo-random number generator
    random_key = jax.random.PRNGKey(config.seed)
    random_key, vdm_key, gamma_key = jax.random.split(random_key, 3)

    # checks if there is a checkpoint file available
    # I am guessing this is mostly to save time
    if checkpoint_file is not None: # if a checkpoint file does exist
        with open(checkpoint_file, 'rb') as file:
            training_state = pickle.load(file)

    else: # create new variational diffusion model


        vdm_params, vdm_state = variation_diffusion_model.init(
            vdm_key, single_device_batch)
        gamma_params = noise_scheduler.init(gamma_key, single_device_batch)

        vdm_optimizer_state = vdm_optimizer.init(vdm_params)
        gamma_optimizer_state = gamma_optimizer.init(gamma_params)

        vdm_state["~"] = dataset.statistics

        training_state = TrainingState(
            vdm_params,
            gamma_params,
            vdm_state,

            vdm_optimizer_state,
            gamma_optimizer_state
        )
```


### **Baseline Models**

### **VLD (Proposed) Model**

**Necessary Files**

`train_lvd.py`: This particular file acts as the 

**Pipeline**

### **Libraries Explanation**

`omegaconf`: handles merging configurations from multiple sources
 

