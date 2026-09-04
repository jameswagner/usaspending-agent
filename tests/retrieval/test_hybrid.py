from backend.app.retrieval.hybrid import merge_candidates


def test_chunk_found_by_both_keeps_both_sets_of_scores():
    dense = {"c1": {"id": "c1", "text": "hello", "dense_rank": 1, "dense_distance": 0.5}}
    sparse = {"c1": {"id": "c1", "text": "hello", "sparse_rank": 1, "sparse_score": 9.0}}

    merged = merge_candidates(dense, sparse)

    assert len(merged) == 1
    assert merged[0]["dense_rank"] == 1
    assert merged[0]["dense_distance"] == 0.5
    assert merged[0]["sparse_rank"] == 1
    assert merged[0]["sparse_score"] == 9.0


def test_chunk_found_only_by_dense_has_no_sparse_fields():
    dense = {"c1": {"id": "c1", "text": "hello", "dense_rank": 1, "dense_distance": 0.5}}
    sparse = {}

    merged = merge_candidates(dense, sparse)

    assert len(merged) == 1
    assert "sparse_rank" not in merged[0]
    assert "sparse_score" not in merged[0]


def test_chunk_found_only_by_sparse_has_no_dense_fields():
    dense = {}
    sparse = {"c2": {"id": "c2", "text": "world", "sparse_rank": 1, "sparse_score": 5.0}}

    merged = merge_candidates(dense, sparse)

    assert len(merged) == 1
    assert "dense_rank" not in merged[0]
    assert "dense_distance" not in merged[0]


def test_union_includes_chunks_unique_to_each_side():
    dense = {"c1": {"id": "c1", "text": "a", "dense_rank": 1, "dense_distance": 0.1}}
    sparse = {"c2": {"id": "c2", "text": "b", "sparse_rank": 1, "sparse_score": 3.0}}

    merged = merge_candidates(dense, sparse)

    ids = {c["id"] for c in merged}
    assert ids == {"c1", "c2"}


def test_empty_inputs_produce_empty_output():
    assert merge_candidates({}, {}) == []
