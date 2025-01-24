import os
from core.pipeline import analyze_mix

def test_analyze_mix():
    """
    Test the analyze_mix function with a mock mix and Amen Break.
    """
    os.makedirs("mock_data", exist_ok=True)
    mix_file = "mock_data/mix.wav"
    amen_file = "mock_data/amen_break.wav"
    # Create mock files
    with open(mix_file, "wb") as f:
        f.write(b"\x00" * 1000)
    with open(amen_file, "wb") as f:
        f.write(b"\x00" * 1000)

    metrics, graph = analyze_mix(mix_file, amen_file, db_dir="mock_db", output_dir="mock_results")

    assert os.path.exists(metrics)
    assert os.path.exists(graph)

    # Clean up
    os.system("rm -rf mock_data mock_db mock_results")
