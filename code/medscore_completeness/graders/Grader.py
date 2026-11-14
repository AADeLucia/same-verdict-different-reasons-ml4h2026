"""Basic methods for LLM API calls"""
import os
from typing import List, Any, Dict, Optional, Iterable, Type, Tuple, override
from argparse import ArgumentParser
import inspect
import logging

from medscore_completeness.graders.LLMAPIModel import APIModel

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger("Grader")


class Grader(APIModel):
    """Base class for all Graders, with an automatic registry."""

    # This dictionary will hold the mapping from grader names to grader classes.
    _registry: Dict[str, Type["Grader"]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called whenever a class inherits from Grader.
        It automatically registers the new subclass.
        """
        super().__init_subclass__(**kwargs)

        # We'll require each subclass to have a 'grader_name' attribute.
        # Ones without are skipped and not registered
        if hasattr(cls, 'grader_name'):
            cls._registry[cls.grader_name] = cls

    @classmethod
    def get_grader(
        cls,
        grader_name: str,
        grader_params: Dict[str, Any]
    ) -> "Grader":
        """
        Factory method for Graders.
        It now uses the central registry to find the correct class.
        """
        grader_class = cls._registry.get(grader_name)
        if not grader_class:
            raise ValueError(f"Grader '{grader_name}' not found. Available graders: {list(cls._registry.keys())}")

        # Filter provided params to only those relevant for the selected grader
        sig = inspect.signature(grader_class.__init__)
        valid_params = {p.name for p in sig.parameters.values()}
        filtered_params = {k: v for k, v in grader_params.items() if k in valid_params}

        return grader_class(**filtered_params)

    @classmethod
    def get_available_graders(cls) -> List[str]:
        """A helper method to see all registered graders."""
        return list(cls._registry.keys())

    @classmethod
    def add_arguments_to_parser(cls, parser: ArgumentParser):
        """
        Inspects all registered Grader subclasses and adds their __init__
        parameters to the provided ArgumentParser.

        NOTE: SWITCHED TO CONFIG FILE USAGE INSTEAD OF ARGS
        """
        # Hacky, assumes all metrics use the same models
        parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
        parser.add_argument("--model-server", type=str, default="https://api.together.xyz/v1")
        parser.add_argument("--api-key", type=str, default=os.environ.get("TOGETHER_API_KEY"))

