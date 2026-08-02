from workflow.confirmer import ConsecutiveConfirmer


def test_confirmer_requires_full_window_before_any_result():
    c = ConsecutiveConfirmer(window=5, min_hits=4)
    for _ in range(4):
        assert c.update(("t1", "x"), True) is False  # window not full yet


def test_confirmer_confirms_once_min_hits_reached_within_window():
    c = ConsecutiveConfirmer(window=5, min_hits=4)
    results = [c.update(("t1", "x"), True) for _ in range(5)]
    assert results[-1] is True


def test_confirmer_rejects_single_frame_spike():
    c = ConsecutiveConfirmer(window=5, min_hits=4)
    hits = [False, False, True, False, False]
    results = [c.update(("t1", "x"), h) for h in hits]
    assert results[-1] is False  # only 1 hit out of 5, min_hits=4 not met


def test_confirmer_keys_are_independent():
    c = ConsecutiveConfirmer(window=3, min_hits=3)
    for _ in range(3):
        c.update(("track_a", "weapon"), True)
    assert c.update(("track_b", "weapon"), False) is False


def test_forget_prefix_clears_all_keys_for_a_track():
    c = ConsecutiveConfirmer(window=3, min_hits=3)
    c.update((7, "helmet"), True)
    c.update((7, "mask"), True)
    c.update((8, "helmet"), True)
    c.forget_prefix(7)
    assert (7, "helmet") not in c._history
    assert (7, "mask") not in c._history
    assert (8, "helmet") in c._history
