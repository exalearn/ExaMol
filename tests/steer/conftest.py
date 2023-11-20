from pathlib import Path
import sys

from colmena.queue import ColmenaQueues, PipeQueues
from colmena.task_server import ParslTaskServer
from parsl import Config, HighThroughputExecutor
from proxystore.connectors.file import FileConnector
from proxystore.store import Store, register_store
from pytest import fixture
from sklearn.pipeline import Pipeline

from examol.score.rdkit import RDKitScorer, make_knn_model
from examol.simulate.ase import ASESimulator
from examol.store.db.base import MoleculeStore
from examol.store.db.memory import InMemoryStore
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy


@fixture()
def recipe() -> RedoxEnergy:
    return RedoxEnergy(charge=1, energy_config='mopac_pm7', vertical=True)


@fixture()
def database(recipe, tmpdir) -> MoleculeStore:
    """Make a starting training set"""
    store = InMemoryStore(tmpdir / 'store.json')
    with store:
        for i, smiles in enumerate(['CCCC', 'CCO']):
            record = MoleculeRecord.from_identifier(smiles)
            record.properties[recipe.name] = {recipe.level: i}
            store.update_record(record)
        yield store


@fixture()
def search_space(tmp_path) -> Path:
    path = tmp_path / 'search-space.smi'
    with path.open('w') as fp:
        for s in ['C', 'N', 'O', 'Cl', 'S']:
            print(s, file=fp)
    return path


@fixture()
def scorer() -> tuple[RDKitScorer, Pipeline]:
    pipeline = make_knn_model(n_neighbors=1)
    return RDKitScorer(), pipeline


@fixture()
def simulator(tmp_path) -> ASESimulator:
    return ASESimulator(scratch_dir=tmp_path / 'ase-temp')


@fixture()
def queues(recipe, scorer, simulator, tmp_path) -> ColmenaQueues:
    """Make a start the task server"""
    # Unpack inputs
    scorer, _ = scorer

    # Make the queues
    store = Store(name='file', connector=FileConnector(store_dir=str(tmp_path)))
    register_store(store, exist_ok=True)
    queues = PipeQueues(topics=['inference', 'simulation', 'train'], proxystore_name=store.name)

    # Make parsl configuration
    config = Config(
        run_dir=str(tmp_path),
        executors=[HighThroughputExecutor(start_method='spawn', max_workers=1, address='127.0.0.1')]
    )

    doer = ParslTaskServer(
        queues=queues,
        methods=[scorer.score, simulator.optimize_structure, simulator.compute_energy, scorer.retrain],
        config=config,
        timeout=15,
    )
    doer.start()

    yield queues
    queues.send_kill_signal()
    doer.join()


@fixture()
def on_mac():
    return sys.platform == 'darwin'
