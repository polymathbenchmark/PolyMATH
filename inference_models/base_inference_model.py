from typing import List
from abc import abstractmethod, ABC


class BaseInferenceModel(ABC):
    def __init__(self, model_id):
        self.model_id = model_id

    @abstractmethod
    def get_template(self,
                     prompt: str,
                     image_path: List[str]
                     ):
        pass

    @abstractmethod
    def get_response(self,
                     prompt: str,
                     image_path: List[str]
                     ):
        pass
