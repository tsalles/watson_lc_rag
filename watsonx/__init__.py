from enum import Enum

from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from watsonx import ai


class ModelTypes(Enum):
    """Supported foundation models."""
    FLAN_T5_XXL = "google/flan-t5-xxl"
    FLAN_UL2 = "google/flan-ul2"
    MT0_XXL = "bigscience/mt0-xxl"
    GPT_NEOX = 'eleutherai/gpt-neox-20b'
    MPT_7B_INSTRUCT2 = 'ibm/mpt-7b-instruct2'
    LLAMA2 = "meta-llama/llama-2-70b-chat"


FoundationModel = ModelTypes
DecodingMethods = DecodingMethods
GenParams = GenParams
