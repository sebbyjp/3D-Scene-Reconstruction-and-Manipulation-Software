import pytest
import cv2 as cv
import numpy as np
import os
import tempfile
from utils import load_input_data, preprocess_input_data, save_output_data, evaluate_output_data

@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def sample_video():
    return [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]

def test_load_input_data(tmp_path):
    # Test image loading
    image_path = tmp_path / "test_image.jpg"
    cv.imwrite(str(image_path), np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    assert len(load_input_data(str(image_path))) == 1

    # Test video loading
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(str(video_path), fourcc, 20.0, (100, 100))
    for _ in range(10):
        out.write(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    out.release()
    assert len(load_input_data(str(video_path))) == 10

def test_preprocess_input_data(sample_image, sample_video):
    # Test single image preprocessing
    processed_image = preprocess_input_data([sample_image])[0]
    assert processed_image.shape == (600, 600, 3)
    assert np.max(processed_image) <= 1.0

    # Test video preprocessing
    processed_video = preprocess_input_data(sample_video)
    assert len(processed_video) == 10
    assert all(frame.shape == (600, 600, 3) for frame in processed_video)
    assert all(np.max(frame) <= 1.0 for frame in processed_video)

def test_save_output_data(tmp_path, sample_image):
    output_data = [sample_image]

    # Test saving as pickle
    pickle_path = tmp_path / "test_output.pkl"
    save_output_data(output_data, str(pickle_path))
    assert pickle_path.exists()

    # Test saving as HDF5
    h5_path = tmp_path / "test_output.h5"
    save_output_data(output_data, str(h5_path))
    assert h5_path.exists()

    # Test saving as OBJ
    obj_path = tmp_path / "test_output.obj"
    save_output_data([(1.0, 2.0, 3.0)], str(obj_path))
    assert obj_path.exists()

    # Test saving as MP4
    mp4_path = tmp_path / "test_output.mp4"
    save_output_data(output_data, str(mp4_path))
    assert mp4_path.exists()

    # Test saving as GIF
    gif_path = tmp_path / "test_output.gif"
    save_output_data(output_data, str(gif_path))
    assert gif_path.exists()

def test_evaluate_output_data(capsys, sample_image):
    input_data = [sample_image]
    output_data = [sample_image]  # Using the same data for simplicity

    evaluate_output_data(output_data, input_data)
    captured = capsys.readouterr()
    
    assert "Mean Squared Error:" in captured.out
    assert "Structural Similarity Index:" in captured.out
    assert "Frames Per Second:" in captured.out
