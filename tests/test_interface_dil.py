from dilnaz.train.interface.interface_dil import build_auto_mapping


def test_auto_mapping_allows_single_word_probe():
    assert build_auto_mapping(["gerçekleştirebileceğimizi"], [[1.0]]) == {}


def test_auto_mapping_swaps_two_words():
    assert build_auto_mapping(["araba", "kitap"], [[1.0, 0.3], [0.3, 1.0]]) == {0: 1, 1: 0}
