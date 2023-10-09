# Pipeline for Creating the Starting Surfaces

The result of these notebooks is a starting point for adsorption calculations.
The starting input is a bulk crystal, which we then:

1. Relax using CP2K with PBE configured to ExaMol's standard settings
2. Produce slab structures and enumerate adsorbate sites with CatKit
3. Relax the slab structures with CP2K
4. Assemble a ExaMol-format description of the structures

Each notebook uses CatKit to produce surface slabs and a find surface sites.