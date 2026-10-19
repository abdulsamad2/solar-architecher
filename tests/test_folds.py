import datetime as dt

from src.folds import blocked_folds


def test_blocked_folds_are_contiguous_disjoint_and_purged():
    days = [dt.date(2018, 1, 1) + dt.timedelta(days=k) for k in range(50) if k % 7 != 3]
    folds = blocked_folds(days, n_folds=5, purge_days=1)
    all_val = [d for f in folds for d in f["val_days"]]
    assert sorted(all_val) == sorted(days)                      # every day validated exactly once
    for f in folds:
        val, train = set(f["val_days"]), set(f["train_days"])
        assert not val & train
        assert f["val_days"] == sorted(f["val_days"])
        # contiguity: no training day falls inside the validation span
        assert not [d for d in train if f["val_days"][0] <= d <= f["val_days"][-1]]
        # purge: no training day within 1 calendar day of the block edges
        assert not [d for d in train if abs((d - f["val_days"][0]).days) <= 1 or abs((d - f["val_days"][-1]).days) <= 1]
