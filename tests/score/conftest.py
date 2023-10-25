from pytest import fixture

from examol.store.models import MoleculeRecord
from examol.store.recipes.base import PropertyRecipe


class FakeRecipe(PropertyRecipe):
    pass


@fixture()
def recipe() -> PropertyRecipe:
    return FakeRecipe('test', 'fast')


@fixture()
def multifi_recipes(recipe) -> list[PropertyRecipe]:
    return [recipe, FakeRecipe(recipe.name, 'medium'), FakeRecipe(recipe.name, 'slow')]


@fixture()
def training_set(multifi_recipes) -> list[MoleculeRecord]:
    """Fake training set"""

    output = []
    for s, y in zip(['C', 'CC', 'CCC'], [1, 2, 3]):
        record = MoleculeRecord.from_identifier(s)
        record.properties[multifi_recipes[0].name] = dict(
            (recipe.level, y + i) for i, recipe in enumerate(multifi_recipes)
        )
        output.append(record)
    return output
