from examol.solution import MultiFidelityActiveLearning
from examol.store.recipes import RedoxEnergy


def test_match():
    solution = MultiFidelityActiveLearning(
        steps=[[RedoxEnergy(charge=1, energy_config='low')]]
    )
    levels = solution.get_levels_for_property(RedoxEnergy(charge=1, energy_config='high'))
    assert [r.level for r in levels] == ['low-adiabatic', 'high-adiabatic']
