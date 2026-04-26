from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ExternalAPIClient:
    api_key: str
    api_url: str
    model_name: str
    timeout: int = 60

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        endpoint = normalize_api_url(self.api_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if endpoint.endswith("/responses"):
            payload = {
                "model": self.model_name,
                "input": messages,
            }
        else:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.2,
            }

        response = self._post_json(endpoint, payload)
        return parse_model_response(response)

    def _post_json(self, endpoint: str, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API returned HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"API request failed: {exc.reason}") from exc

        try:
            parsed = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API returned non-JSON response: {raw_body[:500]}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("API returned an unsupported JSON response.")

        return parsed


def normalize_api_url(api_url: str) -> str:
    cleaned = api_url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("API_URL must be a valid http(s) URL.")

    if cleaned.endswith("/chat/completions") or cleaned.endswith("/responses"):
        return cleaned
    if cleaned.endswith("/v1"):
        return f"{cleaned}/chat/completions"
    return f"{cleaned}/v1/chat/completions"


def parse_model_response(response: dict) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    ]
                    combined = "".join(parts).strip()
                    if combined:
                        return combined

            text = first_choice.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()

    output = response.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        combined = "".join(parts).strip()
        if combined:
            return combined

    raise RuntimeError("API response did not contain generated text.")
