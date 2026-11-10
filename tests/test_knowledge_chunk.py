from __future__ import annotations

from aiolocal.knowledge import MAX_CHUNKS_PER_DOC, chunk_text


def test_long_block_without_blank_lines_is_split_by_size():
    text = "\n".join(f"Record {i}: value " + ("x " * 200) for i in range(2000))
    chunks = chunk_text(text, target_tokens=360, overlap_tokens=60)
    assert len(chunks) > 20
    assert len(chunks) <= MAX_CHUNKS_PER_DOC
    for chunk in chunks:
        # Every chunk must stay close to the target instead of becoming one blob.
        assert len(chunk["text"]) < 4000
    joined = "".join(chunk["text"] for chunk in chunks)
    assert "Record 0:" in joined
    assert "Record 1999:" in joined


def test_single_huge_line_is_hard_sliced():
    text = "A" * 50_000
    chunks = chunk_text(text, target_tokens=360, overlap_tokens=0)
    assert len(chunks) > 10
    assert all(len(chunk["text"]) <= 1600 for chunk in chunks)
    assert "".join(chunk["text"] for chunk in chunks).count("A") >= 49_000


def test_markdown_headings_are_kept_on_chunks():
    text = "# Alpha\n" + ("word " * 500) + "\n\n# Beta\n" + ("more " * 500)
    chunks = chunk_text(text)
    headings = {chunk["heading"] for chunk in chunks}
    assert "Alpha" in headings
    assert "Beta" in headings
