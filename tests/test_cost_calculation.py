from prismalab.storage import PrismaLabStore


def test_calculate_fast_costs_mixed_providers():
    costs = PrismaLabStore._calculate_fast_costs(
        fast_seedream=10,
        fast_nano=5,
        cost_fast_photo=0.03,
        cost_nano_banana=0.07,
        usd_rub=90.0,
    )
    assert costs["fast_seedream"] == 27.0
    assert costs["fast_nano"] == 31.5
    assert costs["fast_total"] == 58.5


def test_calculate_fast_costs_legacy_seedream_only():
    costs = PrismaLabStore._calculate_fast_costs(
        fast_seedream=8,
        fast_nano=0,
        cost_fast_photo=0.035,
        cost_nano_banana=0.07,
        usd_rub=90.0,
    )
    assert costs["fast_seedream"] == 25.2
    assert costs["fast_nano"] == 0.0
    assert costs["fast_total"] == 25.2


def test_calculate_fast_costs_zero_generations():
    costs = PrismaLabStore._calculate_fast_costs(
        fast_seedream=0,
        fast_nano=0,
        cost_fast_photo=0.035,
        cost_nano_banana=0.07,
        usd_rub=90.0,
    )
    assert costs == {"fast_seedream": 0.0, "fast_nano": 0.0, "fast_total": 0.0}
