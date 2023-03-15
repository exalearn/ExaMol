from pytest import fixture

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class FakeRecipe(PropertyRecipe):
    pass


@fixture()
def recipe() -> PropertyRecipe:
    return FakeRecipe('test', 'fast')


@fixture()
def training_set(recipe) -> list[MoleculeRecord]:
    """Fake training set"""

    output = []
    for s, y in zip(['C', 'CC', 'CCC'], [1, 2, 3]):
        record = MoleculeRecord.from_identifier(s)
        record.properties[recipe.name] = {recipe.level: y}
        output.append(record)
    return output
