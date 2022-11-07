from abc import ABC, abstractmethod
from .blueprint import Blueprint
from app.core.components.collector import Collector

class ComponentBuilder(ABC):
    
    @abstractmethod
    def build(self, blueprint: Blueprint):
        """Build a component based on a Blueprint.
        """
        pass

    @abstractmethod
    def post_build(self):
        """Modifies the behavior of the Component after the build is done. 
        """
        pass
    


class CollectorBuilder(ComponentBuilder):
    
    def build(self, blueprint: Blueprint) -> Collector:
        return Collector(*self.blueprint)
    