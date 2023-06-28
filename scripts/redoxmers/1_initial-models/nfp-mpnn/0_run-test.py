import pandas as pd

from examol.store.recipes import RedoxEnergy, PropertyRecipe
from examol.score.nfp import NFPScorer, make_simple_network
from examol.store.models import MoleculeRecord
from sklearn.model_selection import train_test_split
from sklearn import metrics
from argparse import ArgumentParser
from pathlib import Path
from hashlib import md5
import json

database = '../../check-chemistry-settings/database.json'

if __name__ == "__main__":
    # Parse input arguments
    parser = ArgumentParser()
    parser.add_argument('--atom-features', default=32, help='How many features to use to describe each atom/bond', type=int)
    parser.add_argument('--message-steps', default=4, help='How many message-passing steps', type=int)
    parser.add_argument('--output-layers', default=(64, 32, 32), help='Number of hidden units in the output layers', nargs='*', type=int)
    parser.add_argument('--reduce-op', default='sum', help='Operation used to combine atom- to -molecule-level features')
    parser.add_argument('--atomwise', action='store_true', help='Whether to combine to molecule-level features before or after output layers')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite the previous model run')
    parser.add_argument('--num-epochs', type=int, default=4, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Number of records per batch')
    parser.add_argument('property', choices=['ea', 'ip'], help='Name of the property to assess')
    parser.add_argument('level', help='Accuracy level of the property to predict')

    args = parser.parse_args()

    # Load in the training data
    recipe: PropertyRecipe = {
        'ea': RedoxEnergy(-1, energy_config=args.level, vertical=False, solvent='acn'),
        'ip': RedoxEnergy(1, energy_config=args.level, vertical=False, solvent='acn')
    }[args.property]
    all_records: list[MoleculeRecord] = []
    database_hasher = md5()
    with open(database) as fp:
        for line in fp:
            record = MoleculeRecord.from_json(line)
            try:
                recipe.update_record(record)
                all_records.append(record)
                database_hasher.update(record.key.encode())
            except ValueError:
                continue
    if len(all_records) == 0:
        raise ValueError(f'Did not find any training data for {args.property}//{args.level}')
    database_hash = database_hasher.hexdigest()
    print(f'Found {len(all_records)} records. Data hash: {database_hash}')

    # Make a run directory
    run_settings = args.__dict__.copy()
    run_settings.pop('overwrite')
    settings_hash = md5(json.dumps(args.__dict__).encode()).hexdigest()[-8:]
    run_dir = Path(f'runs/f={args.atom_features}-T={args.message_steps}-r={args.reduce_op}-atomwise={args.atomwise}-hash={settings_hash}')
    if run_dir.exists() and not args.overwrite:
        raise ValueError('Run already done')
    run_dir.mkdir(exist_ok=True, parents=True)
    (run_dir / 'params.json').write_text(json.dumps(run_settings))
    (run_dir / 'database.md5').write_text(database_hash)

    # Split the data into training and test sets
    train_records, test_records = train_test_split(all_records, shuffle=True, random_state=1)
    print(f'Split off {len(train_records)} for a training set')

    # Make the network
    model = make_simple_network(
        message_steps=args.message_steps,
        atom_features=args.atom_features,
        output_layers=args.output_layers,
        reduce_op=args.reduce_op,
        atomwise=args.atomwise,
    )
    print('Made the model')

    # Run the training
    scorer = NFPScorer(retrain_from_scratch=True)
    train_inputs = scorer.transform_inputs(train_records)
    train_outputs = scorer.transform_outputs(train_records, recipe)
    model_msg = scorer.prepare_message(model, training=True)
    update_msg = scorer.retrain(model_msg, train_inputs, train_outputs, verbose=True, num_epochs=args.num_epochs, batch_size=args.batch_size)
    scorer.update(model, update_msg)

    # Save the model and training log
    model.save(run_dir / 'model.h5')
    pd.DataFrame(update_msg[1]).to_csv(run_dir / 'log.csv', index=False)

    # Measure the performance on the hold-out set
    test_inputs = scorer.transform_inputs(test_records)
    test_outputs = scorer.transform_outputs(test_records, recipe)
    model_msg = scorer.prepare_message(model)
    test_preds = scorer.score(model_msg, test_inputs)

    summary = dict(
        (f.__name__, f(test_outputs, test_preds)) for f in
        [metrics.mean_absolute_error, metrics.r2_score, metrics.mean_squared_error]
    )
    (run_dir / 'test_summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame({
        'smiles': [r.identifier.smiles for r in test_records],
        'true': test_outputs.squeeze(),
        'pred': test_preds.squeeze()
    }).to_csv(run_dir / 'test_results.csv', index=False)
