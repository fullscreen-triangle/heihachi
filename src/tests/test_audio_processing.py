from core.audio_processing import process_audio

def test_process_audio():
    """
    Test the process_audio function with a mock audio file.
    """
    mock_audio = "mock_data/mock_audio.wav"
    amen_break = "mock_data/mock_amen.wav"
    # Create mock files
    with open(mock_audio, "wb") as f:
        f.write(b"\x00" * 1000)
    with open(amen_break, "wb") as f:
        f.write(b"\x00" * 1000)

    features = process_audio(mock_audio, amen_break)
    assert features is not None
    assert "tempo" in features
    assert "key" in features
    assert "alignments" in features
