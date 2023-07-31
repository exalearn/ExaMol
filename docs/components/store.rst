Store
=====

The Store module handles capturing data about molecules and using collected data to compute derived properties.

Data Models
-----------

The :class:`~examol.store.models.MoleculeRecord` captures information on a molecule.\ [1]_
Information includes identifiers (e.g., SMILES string, project-specific names),
organizational metadata (e.g., membership in project-specific subsets),
energies of the molecule in different geometries (i.e., conformers),
and properties derived from the energies.

Energies are stored as a list of :class:`~examol.store.models.Conformer` objects.
Each of the Conformer objects are different geometries\ [2]_ and we store the energies under different conditions
(e.g., computational methods, charge states) as :class:`~examol.store.models.EnergyEvaluation` objects.

Create a record and populate information about it by
creating a blank Record from a molecule identifier (i.e., SMILES)
then providing a simulation result to its `add_energies` method.

.. code-block:: python

    record = MoleculeRecord.from_identifier('C')
    sim_result = SimResult(
        xyz='5\nmethane\n0.0000...'
        charge=0,
        energy=-1,
        config_name='test',
        solvent=None
    )  # an example result (normally not created manually)
    record.add_energies(sim_result)

You can then look up the stored energies for a molecule from the record.
For example, ExaMol provides a utility operation for finding the lowest-energy conformer:

.. code-block:: python

    conf, energy = record.find_lowest_conformer(config_name='test', charge=0, solvent=None)
    assert isclose(energy, -1)
    assert conf.xyz.startswith('5\nmethane\n0.0000')

Technical Details
~~~~~~~~~~~~~~~~~

The data models are implemented as MongoEngine :class:`~mongoengine.Document` objects
so that they are easy to store in MongoDB, convert to JSON objects, etc.

Recipes
-------

Recipes define how to compute property of a molecule from multiple energy computations.
All are based on the :class:`~examol.store.recipes.PropertyRecipe` object, and provide a
function to compute the property from a molecule data record
and second to generate the list of computations required to complete a computation.

Use an existing recipe by specifying details on the property (e.g., which solvent?) and
the target level of accuracy.
Consult the `API docs <../api/examol.store.html#module-examol.store.recipes>`_ for properties available in ExaMol.

The recipe will then create an informative name for the property and a level of accuracy:

.. code-block:: python

    recipe = RedoxEnergy(charge=1, config_name='test', solvent='acn', vertical=False)
    print(recipe.name)  # reduction_potential
    print(recipe.level)  # test_acn_vertical


You can then use the recipe to determine what is left to do for a recipe

.. code-block:: python

    to_do = recipe.suggest_computations(record)

or compute the property then store it in a data record.

.. code-block:: python

    recipe.update_record(record)
    print(record.properties['reduction_potential']['test_acn_vertical'])  # Value of the property


.. [1] We define a molecule as unique based on its chemical formula (including H's), connectivity, and stereochemistry.
    Stereoisomers are different molecules, molecules that only differ by charge are the same.

.. [2] Geometries are the same atom positions do not different displaced by more than 10\ :sup:`-3` Å,
    when both have a center of mass at the origin. We do not attempt to determine if molecules have different rotations.
