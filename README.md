# **GRAPPA** - **Gra**ph neural network for **P**redicting the **P**arameters of the **A**ntoine equation


GRAPPA is a machine learning model based on a graph neural network architecture that allows predicting the parameters of the Antoine equation only based on molecular structure.
This repository contains the trained model and examples to show how to calculate vapor pressure, the underlying Antoine parameters and boiling temperatures. The model only requires the SMILES representation of the molecule as input.

GRAPPA is based on the Antoine equation in the form:

$$ \ln(p^\mathrm{s} / \mathrm{kPa}) = A - \frac{B}{T / \mathrm{K} + C} $$

where $p^\mathrm{s}$ is the vapor pressure, $T$ is the temperature, and $A$, $B$, and $C$ are the Antoine parameters.

## Installing GRAPPA
1. **Clone the repository**
    ```bash
    git clone https://github.com/marco-hoffmann/GRAPPA.git
    cd GRAPPA
    ```
2. **Create an environment with all required packages.** 
    
    To install the required packages, simply create a new conda environment from the `grappa_env.yml` file:
    ```bash
    conda env create -f grappa_env.yml
    ```
 3. **Activate the environment**
    ```bash
    conda activate grappa_env
    ```

## Using GRAPPA
The notebook file `GRAPPA_examples.ipynb` contains examples on how to use the GRAPPA model to predict vapor pressure, Antoine parameters and boiling temperatures.

## Citing GRAPPA
If you use GRAPPA in your research, please cite the following publication:
```bibtex
@misc{Hoffmann2025,
      title={GRAPPA - A Hybrid Graph Neural Network for Predicting Pure Component Vapor Pressures}, 
      author={Marco Hoffmann and Hans Hasse and Fabian Jirasek},
      year={2025},
      eprint={2501.08729},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.08729}, 
}
```
## License
The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
