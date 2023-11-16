from pathlib import Path

from pytest import fixture, mark

from examol.select.baseline import RandomSelector
from examol.start.fast import RandomStarter
from examol.solution import MultiFidelityActiveLearning
from examol.steer.multifi import PipelineThinker
from examol.store.recipes import RedoxEnergy


@fixture()
def thinker(queues, recipe, search_space, scorer, database, tmpdir) -> PipelineThinker:
    run_dir = Path(tmpdir / 'run')
    scorer, model = scorer
    solution = MultiFidelityActiveLearning(
        steps=[[RedoxEnergy(1, energy_config='xtb', vertical=True)]],
        scorer=scorer,
        starter=RandomStarter(),
        models=[[model, model]],
        selector=RandomSelector(10),
        minimum_training_size=4,
        num_to_run=3,
        pipeline_target=1.
    )
    return PipelineThinker(
        queues=queues,
        run_dir=run_dir,
        recipes=[recipe],
        database=database,
        num_workers=1,
        solution=solution,
        search_space=[search_space],
    )


def test_initialize(thinker):
    assert thinker.num_levels == 2
    assert thinker.steps[0][0].level == 'xtb-vertical'
    assert thinker.steps[1][0].level == 'mopac_pm7-vertical'


def test_detect_level(thinker, recipe):
    assert thinker.get_level('C') == 0

    # Add the lowest level, we should then be a level 1
    thinker.database.get_or_make_record('C').properties[recipe.name] = {'xtb-vertical': 1}
    assert thinker.get_level('C') == 1

    # Add the highest level, we should then be a level 2
    thinker.database.get_or_make_record('C').properties[recipe.name][recipe.level] = 1
    assert thinker.get_level('C') == 2


@mark.timeout(15)
def test_iterator(thinker, recipe, mocker):
    # Make sure we return xTB regardless of what the system asks for
    for choice in [0, 1]:
        with mocker.patch('numpy.random.choice', return_value=choice):
            thinker.task_queue = [('C', 0)]
            record, request = next(thinker.task_iterator)
            assert record.identifier.smiles == 'C'
            assert request.config_name == 'xtb'
            assert len(thinker.task_queue) == 0

    # Test after we add data for C
    #  We should default to C if it's the only item in the task queue
    thinker.database.get_or_make_record('C').properties[recipe.name] = {'xtb-vertical': 1}
    assert thinker.get_level('C') == 1
    for choice in [0, 1]:
        with mocker.patch('numpy.random.choice', return_value=choice):
            thinker.task_queue = [('C', 0)]
            record, request = next(thinker.task_iterator)
            assert record.identifier.smiles == 'C'
            assert request.config_name == 'mopac_pm7', f'choice={choice}'
            assert len(thinker.task_queue) == 0
