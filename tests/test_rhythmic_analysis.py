from alignment.similarity import update_similarity_graph, cluster_segments

def test_update_similarity_graph():
    """
    Test updating the similarity graph with mock tracks.
    """
    mock_tracks = {
        "segment_1": {"tempo": 174, "key": "D minor"},
        "segment_2": {"tempo": 176, "key": "C major"}
    }
    graph = update_similarity_graph(mock_tracks, "mock_graph.gpickle")
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1

def test_cluster_segments():
    """
    Test clustering segments with mock features.
    """
    mock_tracks = {
        "segment_1": {"tempo": 174},
        "segment_2": {"tempo": 176},
        "segment_3": {"tempo": 180}
    }
    clusters = cluster_segments(mock_tracks, eps=5, min_samples=1)
    assert len(clusters) > 0
