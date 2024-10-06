# GRAPPA - **Gra**ph neural network for **P**redicting the **P**arameters of the **A**ntoine equation

GRAPPA is a machine learning model based on a graph neural network architecture that allows predicting the parameters of the Antoine equation only based on molecular structure.
This repository contains the trained model and examples to show how to calculate vapor pressure, the underlying Antoine parameters and (normal) boiling points. The model only requires the SMILES representation of the molecule as input.

The following packages are required to use GRAPPA:
- torch, torch_geometric, rdkit, numpy

Three applications of GRAPPA are already implemented and available as examples. Only the molecular structure (in the form of a SMILES string) is required.

## Direct Vapor Pressure Prediction


```python
from src.utils import *

# Define inputs
smiles_list = ['CCO', 'c1ccccc1O'] # Define smiles for Ethanol and Phenol
temperature_list = [323.0, 293.15] # Define desired temperatures

# Define predictor
predictor = GRAPPAdirect()

# Make prediction
prediction_list = predictor(smiles_list, temperature_list)

# Print prediction
for smiles, temperature, prediction in zip(smiles_list, temperature_list, prediction_list):
    print(f'The vapor pressure of {smiles} at {temperature} K is {prediction:.2f} kPa')

```

## Prediction of Antoine Parameters

```python
from src.utils import *

# Define inputs
smiles_list = ['CCO', 'c1ccccc1O']

# Define predictor
predictor = GRAPPAantoine()

# Make prediction
prediction_list = predictor(smiles_list)

# Print prediction
for smiles, prediction in zip(smiles_list, prediction_list):
    print(f"The Antoine parameters of {smiles} are: A = {prediction[0]:.2f}, B = {prediction[1]:.2f}, C = {prediction[2]:.2f}")
```

## Prediction of Normal Boiling Points

```python
from src.utils import *

# Define inputs
smiles_list = ['CCO', 'c1ccccc1O', 'COC']

# Define predictor
predictor = GRAPPAnormalbp()

# Make prediction
prediction_list = predictor(smiles_list)

# Print prediction
for smiles, prediction in zip(smiles_list, prediction_list):
    print(f"The normal boiling point of {smiles} is {prediction:.2f} K.")
```


## License

MIT License

Copyright (c) 2024 marco-hoffmann

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
