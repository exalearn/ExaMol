"""Generate the initial database and search space"""
import pandas as pd

# Get the search space from the GitHub
data = pd.read_csv("https://github.com/hieuadoan/QM-ActiveLearning-Paper/raw/master/Oxidation-HBE-b3lyp-results-clean.csv")
with open('search-space.smi', 'w') as fp:
    for smiles in data['Reactant']:
        print(smiles, file=fp)
