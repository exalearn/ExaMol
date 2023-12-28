import logging
from pathlib import Path

from pytest import fixture, mark

from examol.select.baseline import RandomSelector
from examol.start.fast import RandomStarter
from examol.solution import MultiFidelityActiveLearning
from examol.steer.multifi import PipelineThinker
from examol.store.recipes import RedoxEnergy
from examol.utils.chemistry import get_inchi_key_from_molecule_string


@fixture()
def thinker(queues, recipe, search_space, scorer, database, tmpdir, pool) -> PipelineThinker:
    run_dir = Path(tmpdir / 'run')
    scorer, model = scorer
    solution = MultiFidelityActiveLearning(
        steps=[[RedoxEnergy(1, energy_config='xtb', vertical=True)]],
        scorer=scorer,
        starter=RandomStarter(),
        models=[[model, model]],
        selector=RandomSelector(10),
        minimum_training_size=4,
        num_to_run=8,
        pipeline_target=0.5
    )
    return PipelineThinker(
        queues=queues,
        run_dir=run_dir,
        recipes=[recipe],
        database=database,
        num_workers=1,
        solution=solution,
        pool=pool,
        search_space=[search_space]
    )


def test_initialize(thinker):
    assert thinker.num_levels == 2
    assert thinker.steps[0][0].level == 'xtb-vertical'
    assert thinker.steps[1][0].level == 'mopac_pm7-vertical'
    assert len(thinker.already_in_db) == 0


def test_detect_relevant(thinker):
    assert len(thinker.get_relevant_database_records()) == 0

    thinker.database.get_or_make_record('C')
    assert thinker.get_relevant_database_records() == {'VNWKTOKETHGBQD-UHFFFAOYSA-N'}


def test_submit_inference(thinker):
    # Without any relevant data in our database
    for recipe_ind, models in enumerate(thinker.models):
        for model_ind, _ in enumerate(models):
            thinker.ready_models.put((recipe_ind, model_ind))
    all_smiles, all_is_done, all_results = thinker.submit_inference()
    assert len(all_smiles) == 1

    # Ensure at least one molecule is read from the database
    thinker.database.get_or_make_record('C')
    thinker.already_in_db.add(get_inchi_key_from_molecule_string('C'))
    for recipe_ind, models in enumerate(thinker.models):
        for model_ind, _ in enumerate(models):
            thinker.ready_models.put((recipe_ind, model_ind))
    all_smiles, all_is_done, all_results = thinker.submit_inference()
    assert len(all_smiles) == 2
    assert all_results[-1].shape == (1, 1, len(thinker.models[0]))


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
        mocker.patch('numpy.random.choice', return_value=choice)
        thinker.task_queue = [('C', 0)]
        record, recipes, request = next(thinker.task_iterator)
        assert record.identifier.smiles == 'C'
        assert request.config_name == 'xtb'
        assert recipes[0].level == 'xtb-vertical'
        assert len(thinker.task_queue) == 0

    # Test after we add data for C
    #  We should default to C if it's the only item in the task queue
    thinker.database.get_or_make_record('C').properties[recipe.name] = {'xtb-vertical': 1}
    assert thinker.get_level('C') == 1
    for choice in [0, 1]:
        mocker.patch('numpy.random.choice', return_value=choice)
        thinker.task_queue = [('C', 0)]
        record, recipes, request = next(thinker.task_iterator)
        assert record.identifier.smiles == 'C'
        assert recipes[0].level == 'mopac_pm7-vertical'
        assert request.config_name == 'mopac_pm7', f'choice={choice}'
        assert len(thinker.task_queue) == 0


@mark.timeout(120)
def test_thinker(thinker: PipelineThinker, database, caplog):
    caplog.set_level(logging.ERROR)

    # Make sure it is set up right
    assert len(thinker.search_space_smiles) == 1

    # Run it
    thinker.run()
    assert len(caplog.records) == 0, caplog.records[0]

    # Make sure that at least a few records made it to a new level
    assert any(thinker.get_level(record.identifier.smiles) > 0 for record in thinker.database.iterate_over_records())
