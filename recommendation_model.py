from abc import ABC, abstractmethod


class RecommendationModel(ABC):
    @abstractmethod
    def train(self, trainset):
        pass

    @abstractmethod
    def evaluate(self, testset):
        pass

    @abstractmethod
    def recommend(self, user_id, k=20):
        pass

    @abstractmethod
    def save(self, model_path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_path):
        pass
