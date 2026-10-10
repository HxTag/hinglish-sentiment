from pathlib import Path

from src.convert_sentimix import convert


def test_convert_rebuilds_tweets_and_keeps_labels(tmp_path: Path) -> None:
    train = tmp_path / "train.txt"
    dev = tmp_path / "dev.txt"
    output = tmp_path / "dataset.csv"
    train.write_text("meta\t10\tpositive\nMast\tHin\nmovie\tEng\n\n", encoding="utf-8")
    dev.write_text("meta\t11\tnegative\nBekar\tHin\nservice\tEng\n", encoding="utf-8")
    assert convert(train, dev, output, overwrite=False) == 2
    contents = output.read_text(encoding="utf-8")
    assert "Mast movie,positive,official_train,10" in contents
    assert "Bekar service,negative,official_dev,11" in contents
