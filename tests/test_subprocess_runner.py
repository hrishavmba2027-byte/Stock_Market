from app.services.subprocess_runner import parse_last_json


def test_parse_last_json_uses_last_json_object():
    stdout = "noise\n{\"status\":\"old\"}\nmore noise\n{\"status\":\"ok\",\"rows\":1}\n"
    assert parse_last_json(stdout) == {"status": "ok", "rows": 1}


def test_parse_last_json_returns_none_for_no_json():
    assert parse_last_json("plain output") is None

