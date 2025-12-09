from scripts.guardrails import validate_input, apply_output_guardrails


def test_input_guardrail_blocks_off_topic():
    res = validate_input("write a love poem about rivers")
    assert res.fail is True
    assert res.reason == "off_topic"


def test_input_guardrail_soft_when_station_missing():
    res = validate_input("show me water levels near phoenix")
    assert res.fail is False
    assert res.soft_flag is True
    assert res.station_ids == []


def test_output_guardrail_triggers_when_no_extract_success():
    steps = [
        (
            {"tool": "extract_highcharts_series"},
            {"success": False, "error": "no chart"}
        )
    ]
    og = apply_output_guardrails("value is 3.2", steps, ["12345678"])
    assert og.triggered is True
    assert "No recent chart data" in og.answer


def test_output_guardrail_passes_when_extract_success():
    steps = [
        (
            {"tool": "extract_highcharts_series"},
            {"success": True, "data": [1, 2, 3]}
        )
    ]
    og = apply_output_guardrails("value is 3.2", steps, ["12345678"])
    assert og.triggered is False
    assert og.answer == "value is 3.2"


def test_zero_series_adds_caveat():
    steps = [
        (
            {"tool": "extract_highcharts_series"},
            {"success": True, "data": [0.0, 0.0, 0.0, 0.0]}
        )
    ]
    og = apply_output_guardrails("all zero", steps, ["12345678"])
    assert og.triggered is False
    assert "at or near zero" in og.answer


def test_no_recent_data_reason_when_list_empty():
    steps = [
        (
            {"tool": "list_highcharts_series"},
            {"success": True, "data": []}
        )
    ]
    og = apply_output_guardrails("draft", steps, ["12345678"])
    assert og.triggered is True
    assert og.reason == "no_recent_data"
    assert "no recent data" in og.answer.lower()


def test_time_window_without_timestamps_adds_note():
    steps = [
        (
            {"tool": "extract_highcharts_series"},
            {"success": True, "data": [1.0, 2.0]}
        )
    ]
    og = apply_output_guardrails("draft", steps, ["12345678"], time_window="24 hours")
    assert og.triggered is False
    assert "lacked timestamps" in og.answer.lower()


def test_invalid_time_window_blocks():
    res = validate_input("Show me water levels for the last 1000 days in Colorado.")
    assert res.fail is True
    assert "time" in (res.reason or "") or "Please use a recent" in (res.user_message or "")


def test_invalid_station_id_blocks():
    res = validate_input("What is happening at station ABC123XYZ?")
    assert res.fail is True
    assert res.reason == "invalid_station_id"
