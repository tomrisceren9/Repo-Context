from unittest.mock import MagicMock, patch

from air_context_engine.benchmarks.benchmark_llm import SUPPORTED_PROVIDERS
from air_context_engine.benchmarks.llm_providers import LocalLLMClient


def test_supported_providers_local_only() -> None:
    assert SUPPORTED_PROVIDERS == ("local",)


def test_local_llm_client_initialization() -> None:
    client = LocalLLMClient(
        base_url="http://mock-local-llm:8000/v1",
        api_key="synthetic-test-key",
    )
    try:
        assert client.base_url == "http://mock-local-llm:8000/v1"
        assert client.client.headers["Authorization"] == "Bearer synthetic-test-key"
        assert client.client.headers["Content-Type"] == "application/json"
    finally:
        client.close()


@patch("httpx.Client.post")
def test_local_llm_generation(mock_post: MagicMock) -> None:
    mock_response = MagicMock()
    mock_response.is_error = False
    mock_response.json.return_value = {
        "id": "mock_id",
        "created": 123456789,
        "model": "mock-model",
        "choices": [
            {
                "message": {"content": "This is a mock answer."},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    mock_post.return_value = mock_response

    client = LocalLLMClient(base_url="http://mock-local-llm:8000/v1")
    try:
        result = client.generate(
            model="mock-model",
            prompt="Hello",
            temperature=0.0,
            max_output_tokens=100,
        )

        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "chat/completions"
        assert kwargs["json"]["model"] == "mock-model"
        assert kwargs["json"]["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["answer"] == "This is a mock answer."
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["resolved_model_version"] == "mock-model"
    finally:
        client.close()
