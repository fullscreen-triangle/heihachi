from alignment.drum_alignment import align_amen_break
import librosa

def test_align_amen_break():
    """
    Test aligning the Amen Break with a mock segment.
    """
    mock_audio = librosa.tone(440, sr=22050, length=1000)
    mock_amen = librosa.tone(220, sr=22050, length=1000)
    alignment = align_amen_break(mock_audio, 22050, mock_amen, 22050)
    assert alignment is not None
