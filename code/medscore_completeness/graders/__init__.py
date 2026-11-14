# code/medscore_completeness/graders/__init__.py

# 1. Import the base class first. It is now fully defined and available.
from .LLMAPIModel import APIModel
from .Grader import Grader

# 2. Now, import all of your specific grader modules.
#    When Python loads these files, the subclasses will be defined and
#    will automatically register themselves with the (already existing) Grader class.
from . import healthbench_response_graders
from . import ICLCompletenessGrader
from . import HumanCompletenessGrader
from . import prometheus_graders

# 3. (Optional but good practice) Define what `from .graders import *` imports.
__all__ = [
    "Grader",
]