from typing import Optional, Mapping, Any, List, AsyncIterator, Iterator

from langchain.callbacks.manager import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema.output import GenerationChunk

import watsonx
from watsonx import FoundationModel, DecodingMethods, GenParams
from watsonx.ai import Model


class WatsonxLLM(LLM):
    model_id: Optional[str] = FoundationModel.FLAN_UL2.value
    decoding_method: Optional[str] = DecodingMethods.GREEDY
    min_new_tokens: Optional[int] = 40
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 10
    repeat_penalty: Optional[float] = 1

    watsonx_model_instance: Model = None

    def __init__(
        self,
        model_id: Optional[FoundationModel] = None,
        decoding_method: Optional[DecodingMethods] = None,
        min_new_tokens: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[float] = None,
        repeat_penalty: Optional[float] = None,
    ):
        super(WatsonxLLM, self).__init__()
        model_params = {
            GenParams.DECODING_METHOD: decoding_method if decoding_method is not None else self.decoding_method,
            GenParams.MIN_NEW_TOKENS: min_new_tokens if min_new_tokens is not None else self.min_new_tokens,
            GenParams.MAX_NEW_TOKENS: max_new_tokens if max_new_tokens is not None else self.max_new_tokens,
            GenParams.TEMPERATURE: temperature if temperature is not None else self.temperature,
            GenParams.TOP_P: top_p if top_p is not None else self.top_p,
            GenParams.TOP_K: top_k if top_k is not None else self.top_k,
            GenParams.REPETITION_PENALTY: repeat_penalty if repeat_penalty is not None else self.repeat_penalty,
        }
        self.model_id = model_id.value if model_id else self.model_id
        self.watsonx_model_instance = watsonx.ai.llm(model_id=self.model_id, params=model_params)

    @property
    def _llm_type(self) -> str:
        return self.model_id

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        result = self.watsonx_model_instance.generate_text(prompt=prompt)
        return result

    @property
    def _default_params(self):
        """Get the model default parameters."""
        return {
            "decoding_method": self.decoding_method,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_id": self.model_id, "model_parameters": self._default_params}

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        raise NotImplementedError

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Iterator[GenerationChunk]:
        raise NotImplementedError

    def _astream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> AsyncIterator[GenerationChunk]:
        raise NotImplementedError
