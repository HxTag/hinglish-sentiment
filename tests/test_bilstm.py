from src.bilstm import Vocabulary


def test_vocabulary_is_train_only_and_handles_oov() -> None:
    vocabulary = Vocabulary.build(["mast movie", "movie achhi"], min_frequency=1, max_size=10)
    assert vocabulary.encode("mast unknown", max_length=10) == [vocabulary.tokens_to_id["mast"], 1]
    assert len(vocabulary) == 5
