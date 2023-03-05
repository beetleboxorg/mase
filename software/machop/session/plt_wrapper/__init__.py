from .vision import VisionModelWrapper
from .nlp import NLPClassificationModelWrapper, NLPLanguageModelingModelWrapper
from machop.models import nlp_models, vision_models


def get_model_wrapper(name, task):
    if name in vision_models:
        return VisionModelWrapper
    elif name in nlp_models:
        if task in ['classification', 'cls']:
            return NLPClassificationModelWrapper
        elif task in ['language_modeling', 'lm']:
            return NLPLanguageModelingModelWrapper
        else:
            raise NotImplementedError(
                f"Task {task} not implemented for NLP models")
    else:
        raise ValueError(f"Model {name} not implemented")
