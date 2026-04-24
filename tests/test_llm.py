import os
from unittest.mock import patch, MagicMock

from daily_arxiv_feed.llm import get_client, chat


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
