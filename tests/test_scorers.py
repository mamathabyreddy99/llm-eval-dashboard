from llm_eval.schema import Case
from llm_eval.scorers import score


def case(scorer, reference="", **params):
    return Case(id="t", category="c", prompt="p", reference=reference, scorer=scorer, params=params)


def test_exact_match_ignores_case_and_punctuation():
    assert score(case("exact_match", "Lee"), "lee.")[0]
    assert not score(case("exact_match", "Lee"), "Sam")[0]


def test_contains():
    assert score(case("contains", "Paris"), "The answer is Paris.")[0]
    assert not score(case("contains", "Paris"), "London")[0]


def test_numeric_uses_last_number_and_handles_commas():
    assert score(case("numeric", "391"), "17 * 23 = 391.")[0]
    assert score(case("numeric", "1000"), "It is 1,000")[0]
    assert not score(case("numeric", "391"), "about 400")[0]
    assert not score(case("numeric", "5"), "no digits here")[0]


def test_json_schema_accepts_valid_and_code_fences():
    c = case("json_schema", "", required={"name": "str", "age": "int"})
    assert score(c, '{"name": "Ada", "age": 36}')[0]
    assert score(c, '```json\n{"name": "Ada", "age": 36}\n```')[0]


def test_json_schema_rejects_bad_json_missing_keys_and_wrong_types():
    c = case("json_schema", "", required={"name": "str", "age": "int"})
    assert not score(c, '{"name": "Ada", "age": 3')[0]
    assert not score(c, '{"name": "Ada"}')[0]
    assert not score(c, '{"name": "Ada", "age": "36"}')[0]
    assert not score(c, '{"name": "Ada", "age": true}')[0]


def test_regex():
    c = case("regex", "OK", pattern=r"^\s*OK\s*$")
    assert score(c, "OK")[0]
    assert not score(c, "Sure, OK")[0]


def test_refusal_both_directions():
    assert score(case("refusal", expect_refusal=True), "I can't help with that.")[0]
    assert not score(case("refusal", expect_refusal=True), "Sure, here is how")[0]
    assert score(case("refusal", expect_refusal=False), "Use kill -9")[0]
    assert not score(case("refusal", expect_refusal=False), "I'm sorry, I can't help")[0]


def test_max_words():
    assert score(case("max_words", limit=5), "one two three")[0]
    assert not score(case("max_words", limit=2), "one two three four")[0]


def test_similarity_threshold():
    c = case("similarity", "caching keeps data in fast storage", threshold=0.5)
    assert score(c, "caching keeps data in fast storage")[0]
    assert not score(c, "completely unrelated words")[0]


def test_llm_judge_falls_back_to_similarity_without_judge():
    ok, _, detail = score(case("llm_judge", "paris is the capital"), "paris is the capital")
    assert ok and "no judge" in detail
