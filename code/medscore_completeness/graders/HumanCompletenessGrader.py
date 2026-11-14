"""Human completeness grader implementation.

Reads pre-computed human completeness scores from item metadata.
Scores must be embedded in the dataset before running this grader
(see get_human_completeness_score on each dataset class).
"""
from typing import Any, Dict, Optional
import logging

from medscore_completeness.graders import Grader
from medscore_completeness import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


class HumanCompletenessGrader(Grader):
    """Passthrough grader that reads pre-computed human completeness scores.

    Reads meta["human_completeness_score"] from each item, which is populated
    at dataset load time by the dataset's get_human_completeness_score() method.
    Does not make any API calls.
    """
    grader_name = "completeness-human"

    def __init__(self, max_concurrent_requests: int = 64, **kwargs):
        # No API calls needed — skip APIModel.__init__ entirely
        self.max_concurrent_requests = int(max_concurrent_requests)
        self.model_name = "Human"

    async def _individual_call(
            self,
            input: Dict[str, Any],
    ) -> Dict[str, Any]:
        grading = {"scoring_model": "Human"}
        grading.update(input["meta"]["human_completeness_score"])
        return {"id": input["id"], self.grader_name: grading}

