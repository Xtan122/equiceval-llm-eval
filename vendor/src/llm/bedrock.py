"""
AWS Bedrock LLM Client — Amazon Bedrock Runtime API.

Supports models such as Claude (Anthropic), Llama (Meta), Titan (Amazon), etc.
Default: anthropic.claude-3-5-sonnet-20241022-v2:0

Setup:
    pip install boto3 python-dotenv
    # Option 1: IAM credentials via environment variables
    export AWS_ACCESS_KEY_ID="..."
    export AWS_SECRET_ACCESS_KEY="..."
    export AWS_DEFAULT_REGION="us-east-1"   # or set in .env / __init__ param

    # Option 2: AWS CLI profile (~/.aws/credentials)
    aws configure
"""

import json
import os
import time

from .base import LLMClient, LLMResponse


class BedrockClient(LLMClient):
    """
    Calls Amazon Bedrock Runtime via boto3 (converse API).

    Supported model families (pass model_id accordingly):
      - Anthropic Claude : anthropic.claude-3-5-sonnet-20241022-v2:0
                           anthropic.claude-3-haiku-20240307-v1:0
      - Meta Llama       : meta.llama3-70b-instruct-v1:0
      - Amazon Titan     : amazon.titan-text-express-v1
      - Mistral          : mistral.mistral-large-2402-v1:0

    Credentials resolution order (standard boto3 chain):
      1. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars (or .env)
      2. ~/.aws/credentials profile
      3. IAM instance/task role (EC2 / ECS / Lambda)
    """

    provider = "bedrock"

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        region_name: str | None = None,
    ):
        try:
            import boto3
        except ImportError:
            raise ImportError("Cài thư viện: pip install boto3 python-dotenv")

        self.model_name = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        region = region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )

    def call(self, user_message: str) -> LLMResponse:
        """
        Send user_message to the model using the Bedrock Converse API
        and return a unified LLMResponse.
        """
        t0 = time.time()

        response = self._client.converse(
            modelId=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": user_message}],
                }
            ],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )

        latency = time.time() - t0

        # Extract text from the response
        output_message = response.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        text = "".join(block.get("text", "") for block in content_blocks)

        # Token usage
        usage = response.get("usage", {})
        in_tokens = usage.get("inputTokens", len(user_message.split()))
        out_tokens = usage.get("outputTokens", len(text.split()))

        return LLMResponse(
            raw_text=text,
            latency_s=latency,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            model_name=self.model_name,
            provider=self.provider,
        )
