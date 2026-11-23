import pytest
from PIL import Image

from src.Tools.smsImageClassifier import (
    SMSClassifierDataset,
    _split_images,
    _unique_images,
    train,
)


def save_image(path, color):
    Image.new("RGB", (4, 4), color).save(path)


def test_real_images_keep_their_class_labels(tmp_path):
    sms = tmp_path / "sms.png"
    other = tmp_path / "other.png"
    save_image(sms, "white")
    save_image(other, "red")
    dataset = SMSClassifierDataset([sms], non_sms_image_paths=[other])

    assert len(dataset) == 2
    assert dataset[0][1] == 0
    assert dataset[1][1] == 1


def test_duplicates_are_removed_before_splitting(tmp_path):
    save_image(tmp_path / "first.png", "white")
    save_image(tmp_path / "duplicate.png", "white")
    save_image(tmp_path / "second.png", "red")
    images = list(_unique_images(tmp_path).values())

    assert len(images) == 2
    training, validation = _split_images(images, 0.2)
    assert len(training) == len(validation) == 1
    assert set(training).isdisjoint(validation)


def test_conflicting_labels_are_rejected(tmp_path):
    sms = tmp_path / "sms"
    non_sms = tmp_path / "non_sms"
    sms.mkdir()
    non_sms.mkdir()
    save_image(sms / "first.png", "white")
    save_image(sms / "second.png", "red")
    save_image(non_sms / "same.png", "white")

    with pytest.raises(ValueError, match="Identical images"):
        train(sms, non_sms_dir=non_sms)


def test_unreadable_image_is_not_replaced_with_synthetic_data(tmp_path):
    path = tmp_path / "bad.png"
    path.write_text("not an image")
    dataset = SMSClassifierDataset([path], non_sms_image_paths=[])

    with pytest.raises(OSError):
        dataset[0]
