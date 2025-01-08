import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
from rdkit.Chem import AllChem
from src.gnn import GRAPPA


###########################################################################################
# These are the classes used for encoding the node and edge features in the molecular graph.
# To rebuilt the model from the paper, they may not be altered.
########################################################################################### 
possible_atom_list = ['C','N','O','Cl','S','F','Br','I','P']
possible_hybridization = [Chem.rdchem.HybridizationType.S,
                          Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3]
possible_num_bonds = [0,1,2,3,4]
possible_num_Hs  = [0,1,2,3] 
possible_stereo  = [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE]

def one_of_k_encoding(x, allowable_set):
    """Apply onehot encoding to feature."""

    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def atom_feature(atom):
    """Extract atom features from instance of rdkit molecule. Check if atom has formal charge or radical electrons and throws error if this is the case."""
        
    symbol        = atom.GetSymbol()
    Type_atom     = one_of_k_encoding(symbol, possible_atom_list) # kept to detect atoms not in possible_atom_list and raise error
    
    # throw error if atom has formal charge
    if atom.GetFormalCharge() != 0:
        raise Exception("Atom has formal charge!")
    
    # throw error if atom has radical electrons
    if atom.GetNumRadicalElectrons() != 0:
        raise Exception("Atom has radical electrons!")
    
    Ring_atom     = [atom.IsInRing()]
    Aromaticity   = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding(atom.GetHybridization(), possible_hybridization)
    Bonds_atom    = one_of_k_encoding(len(atom.GetNeighbors()), possible_num_bonds)
    num_Hs        = one_of_k_encoding(atom.GetTotalNumHs(), possible_num_Hs)
    

    results = Type_atom + Ring_atom + Aromaticity + Hybridization + Bonds_atom + num_Hs

    return np.array(results).astype(np.float32)

def bond_feature(bond):
    """Extract bond features from instance of rdkit molecule."""

    bt = bond.GetBondType()
    
    type_stereo = one_of_k_encoding(bond.GetStereo(), possible_stereo)
    
    # Bond level features
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()] + \
        type_stereo

    return np.array(bond_feats).astype(np.float32)

def mol_to_pyg(mol, temperature):
    """Convert rdkit mol object to pytorch Data object. If conversion fails, return None. If illegal atom properties are detected, throw error."""

    # Return None if rdkit fails to produce molecule
    if mol is None:
      return None
    
    # Return error if molecule does not contain at least one carbon atom
    if not any([atom.GetSymbol() == 'C' for atom in mol.GetAtoms()]):
        raise Exception("Molecule does not contain at least one carbon atom.")

    numHDonors = AllChem.CalcNumHBD(mol)
    numHAcceptors = AllChem.CalcNumHBA(mol)

    # For a molecule get the bonded atoms
    id_pairs = ((b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds())
    atom_pairs = [z for (i, j) in id_pairs for z in ((i, j), (j, i))]
    bonds = (mol.GetBondBetweenAtoms(i, j) for (i, j) in atom_pairs)
    
    # Construct node and edge features
    atom_features = np.array([atom_feature(a) for a in mol.GetAtoms()])
    bond_features = np.array([bond_feature(b) for b in bonds])

    # Create input data graph
    d = Data(edge_index=torch.tensor(np.array(list(zip(*atom_pairs))), dtype=torch.int64),
             x=torch.FloatTensor(atom_features), 
             edge_attr=torch.FloatTensor(bond_features),  #
             numHAcceptors=torch.tensor([numHAcceptors]),
             numHDonors=torch.tensor([numHDonors]),
             temperature=torch.tensor([temperature]))
    
    # Check if valid Data object. Raises error if not.
    d.validate()
    return d

def load_model():
    model = GRAPPA(24, 9, 'GAT', 32, 16, 3, 0.0, 'attention', 4, 3, 2)
    model.load_state_dict(torch.load('./models/GRAPPA_state_dict.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model

def preprocess(smiles_list: list, temperature_list: list):
    """Preprocess the input data for the model."""
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    input_loader = DataLoader([mol_to_pyg(mol, temperature) for mol, temperature in zip(mol_list, temperature_list)], batch_size=32)

    return input_loader

class GRAPPAdirect(torch.nn.Module):
    '''Module for the direct prediction of vapor pressures using GRAPPA.'''
    def __init__(self):
        super(GRAPPAdirect, self).__init__()
        self.model = load_model()

    def forward(self, smiles_list: list, temperature_list: list):
        '''
        Gives a list of vapor pressures calculated with GRAPPA.
        Requires a list of smiles and the corresponding temperatures as input.
        '''
        input_loader = preprocess(smiles_list, temperature_list)
        prediction_list = []
        for batch in input_loader:
            prediction_list.extend(self.model(batch.x, batch.temperature, batch.edge_index, batch.edge_attr, batch.numHDonors, batch.numHAcceptors, batch.batch).detach().numpy())
        # Write prediction and unit into dictionary
        return np.exp(prediction_list).tolist()

class GRAPPAantoine(torch.nn.Module):
    '''Module for the direct prediction of vapor pressures using GRAPPA.'''
    def __init__(self):
        super(GRAPPAantoine, self).__init__()
        self.model = load_model()

    def forward(self, smiles_list):
        '''
        Gives a list of Antoine parameters calculated with GRAPPA.
        Requires a list of smiles as input.
        '''
        input_loader = preprocess(smiles_list, [293.15]*len(smiles_list))
        prediction_list = []
        for batch in input_loader:
            prediction_list.extend(self.model.get_antoine_parameters(batch.x, batch.temperature, batch.edge_index, batch.edge_attr, batch.numHDonors, batch.numHAcceptors, batch.batch).detach().numpy().tolist())
        # Write prediction and unit into dictionary
        return prediction_list
    
class GRAPPAnormalbp(torch.nn.Module):
    '''Module for the prediction of normal boiling points using GRAPPA.'''
    def __init__(self):
        super(GRAPPAnormalbp, self).__init__()
        self.model = load_model()

    def forward(self, smiles_list):
        '''
        Gives a list of normal boiling temperatures calculated with GRAPPA.
        Requires a list of smiles as input.
        '''
        input_loader = preprocess(smiles_list, [293.15]*len(smiles_list))
        prediction_list = []
        for batch in input_loader:
            antoine_params = self.model.get_antoine_parameters(batch.x, batch.temperature, batch.edge_index, batch.edge_attr, batch.numHDonors, batch.numHAcceptors, batch.batch).detach().numpy()
            prediction = (antoine_params[:,1]/(antoine_params[:,0]-np.log(101.325)) - antoine_params[:,2])
            prediction_list.extend(prediction)
        # Write prediction and unit into dictionary
        return prediction_list
