import os
from unittest.mock import patch, MagicMock

from daily_arxiv_feed.llm import get_client, chat, parse_json_response


def test_get_client_reads_env():
    with patch.dict(os.environ, {"WEBEX_TOKEN": "test-token-123"}):
        client = get_client()
        assert client.api_key == "test-token-123"
        assert "llm-proxy" in client.base_url.host


def test_get_client_missing_token_raises():
    env = os.environ.copy()
    env.pop("WEBEX_TOKEN", None)
    with patch.dict(os.environ, env, clear=True):
        try:
            get_client()
            assert False, "Should have raised"
        except ValueError as e:
            assert "WEBEX_TOKEN" in str(e)


def test_chat_calls_openai_create():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"result": "ok"}'
    mock_client.chat.completions.create.return_value = mock_response

    result = chat(
        client=mock_client,
        system="You are helpful.",
        user="Hello",
        json_mode=True,
    )
    assert result == '{"result": "ok"}'
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["model"] == "gpt-5.2"


def test_parse_json_response_plain():
    assert parse_json_response('{"key": "val"}') == {"key": "val"}


def test_parse_json_response_markdown_fenced():
    text = '```json\n{"key": "val"}\n```'
    assert parse_json_response(text) == {"key": "val"}


def test_parse_json_response_with_preamble():
    text = 'Here is the result:\n{"key": "val"}\nDone!'
    assert parse_json_response(text) == {"key": "val"}


def _status_error(status, detail):
    err = Exception(detail)
    err.status_code = status
    err.body = {"detail": detail}
    return err


def test_get_client_disables_sdk_retries_and_sets_timeout():
    with patch.dict(os.environ, {"WEBEX_TOKEN": "t"}):
        client = get_client()
        assert client.max_retries == 0
        assert client.timeout <= 180


def test_chat_fails_fast_on_long_retry_after():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _status_error(
        429, "Rate limit exceeded. retry after 287829.82 seconds."
    )
    with patch("daily_arxiv_feed.llm.time.sleep") as sleep:
        try:
            chat(client=mock_client, system="s", user="u")
            assert False, "Should have raised"
        except RuntimeError as e:
            assert "quota" in str(e).lower()
    assert mock_client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_chat_honors_short_retry_after():
    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = "hi"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        _status_error(429, "Rate limit exceeded. retry after 7.5 seconds."),
        ok,
    ]
    with patch("daily_arxiv_feed.llm.time.sleep") as sleep:
        assert chat(client=mock_client, system="s", user="u") == "hi"
    assert sleep.call_args.args[0] >= 7.5


def test_chat_retries_timeouts():
    import httpx
    from openai import APITimeoutError

    ok = MagicMock()
    ok.choices = [MagicMock()]
    ok.choices[0].message.content = "hi"
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        APITimeoutError(request=httpx.Request("POST", "http://x")),
        ok,
    ]
    with patch("daily_arxiv_feed.llm.time.sleep"):
        assert chat(client=mock_client, system="s", user="u") == "hi"
