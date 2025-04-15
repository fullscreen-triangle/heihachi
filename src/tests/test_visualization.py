from utils.visualization import plot_heatmap

def test_plot_heatmap():
    """
    Test generating a heatmap with mock data.
    """
    metrics = {
        "tempo": [174, 176, 178],
        "beat_salience": [0.8, 0.7, 0.9]
    }
    times = [0.0, 30.0, 60.0]
    plot_heatmap(metrics, times, output_path="mock_heatmap.png")
    assert os.path.exists("mock_heatmap.png")

    # Clean up
    os.remove("mock_heatmap.png")
