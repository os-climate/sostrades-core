'''
Copyright 2024 Capgemini
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import json
import logging
import os
from uuid import uuid4

import requests
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class InvokeException(BaseException):
    pass

class GenerativeEngineService:
    """
    Service for interacting with the Generative Engine API
    """
    def __init__(self, url: str, key: str, session_name=None, workspace_id=None) -> None:

        self.__api_key = key
        self.url = url

        if session_name is None:
            self.session = str(uuid4())

        else:
            self.session = session_name

        self.workspace_id = workspace_id

        self.default_model_kwargs = {
            "streaming": False,
            "maxTokens": 2000,
            "temperature": 0.5,
            "topP": 0.6,
        }

    def __get_headers(self):
        """Get the headers to pass to the API"""
        return {"Accept": "application/json", "x-api-key": self.__api_key}

    def list_models(self):
        """
        List the available models
        """
        url = os.path.join(self.url, "v1/models").replace("\\", "/")
        logger.info(f"Listing models from {url}")
        r = requests.get(url, headers=self.__get_headers())

        return r.json()

    def _parse_invoke_response(self, response_text: str):
        """
        Parse the raw response text into a dictionary
        :param response_text: The response from the invoke endpoint in the API
        """
        start_index = response_text.find("{")

        if start_index == -1:
            return None

        data = json.loads(response_text[start_index:])
        action = data.get("action", None)

        if action == "error":
            # If there was an error returned in the JSON response, raise it
            content = data.get("data", {}).get("content", None)
            raise InvokeException(content)

        return InvokeResponse(**data)

    def run(self, prompt: str, provider: str, model: str, **model_kwargs: dict):
        """
        Runs inference using the Generative Engine API
        :param prompt:
            The prompt to send to the model
        :param provider:
            The provider of the model [bedrock/azure]
        :param model:
            The model to use
        """
        generation_kwargs = {**self.default_model_kwargs, **(model_kwargs or {})}
        logger.debug(f"Using kwargs: {generation_kwargs}")
        data = {
            "action": "run",
            "modelInterface": "langchain",
            "data": {
                "mode": "chain",
                "text": prompt,
                "modelName": model,
                "provider": provider,
                "sessionId": self.session,
                "modelKwargs": generation_kwargs,
            },
        }
        if self.workspace_id is not None:
            data["data"].update({"workspaceId": self.workspace_id})
        url = os.path.join(self.url, "v1/llm/invoke").replace("\\", "/")
        r = requests.post(url, headers=self.__get_headers(), json=data)
        response = self._parse_invoke_response(r.text)

        return response


class ResponseMetadata(BaseModel):
    modelId: str
    modelKwargs: dict
    mode: str
    sessionId: str
    prompts: list[list[str]]

class ResponseData(BaseModel):
    sessionId: str
    type: str
    content: str
    metadata: ResponseMetadata

class InvokeResponse(BaseModel):
    """
    Response format returned from the "invoke" endpoint in the Generative Engine API
    (there may be more than just these fields returned)
    """
    type: str
    action: str
    connectionId: str
    timestamp: int
    direction: str
    data: ResponseData
