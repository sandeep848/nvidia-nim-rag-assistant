from rag_core import Chunk, lexical_rank, reciprocal_rank_fusion


def test_lexical_rank_prefers_matching_chunk():
    chunks = [
        Chunk(0, "robot motion planning", "a.pdf", 1),
        Chunk(1, "database transaction isolation", "b.pdf", 2),
    ]
    assert lexical_rank("robot planning", chunks)[0] == 0


def test_rrf_combines_rankings():
    fused = reciprocal_rank_fusion([1, 2, 3], [3, 2, 4])
    assert fused[0] in {2, 3}
    assert set(fused) == {1, 2, 3, 4}
