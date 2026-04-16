from llm_hallucination_detector.utils.text import chunk_text, parse_json_array


def test_parse_json_array_valid():
    assert parse_json_array('["a", "b"]') == ["a", "b"]


def test_parse_json_array_with_noise():
    assert parse_json_array('prefix ["a"] suffix') == ["a"]


def test_chunk_text_overlap():
    text = "one two three four five six"
    chunks = chunk_text(text, chunk_size=3, overlap=1)
    assert chunks == ["one two three", "three four five", "five six"]
