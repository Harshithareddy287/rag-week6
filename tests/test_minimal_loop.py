from src.minimal_loop import HOTEL_TEXT, PTO_TEXT, run_minimal_loop


def test_expected_texts_rank_first():
    ranked_a, ranked_b = run_minimal_loop()
    assert ranked_a[0] == PTO_TEXT
    assert ranked_b[0] == HOTEL_TEXT
