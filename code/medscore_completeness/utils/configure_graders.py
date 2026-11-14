"""Load and parse YAML config files"""
import os
import logging
from typing import Dict, Any
import yaml

from medscore_completeness.graders import Grader
from medscore_completeness import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger(__name__)


def env_constructor(
        loader: yaml.Loader,
        node: yaml.nodes.ScalarNode
) -> Any:
    """Constructor for the !env tag in YAML configs.

    Attempts to convert the environment variable to appropriate Python types:
    - int for integer strings
    - float for decimal strings
    - bool for 'true'/'false' strings
    - str otherwise
    """
    value = loader.construct_scalar(node)
    env_value = os.environ.get(value)
    logger.debug(f"Replacing {value} with its environment variable.")

    if env_value is None:
        return None

    # Try to convert to appropriate type
    # Check for boolean
    if env_value.lower() == "true":
        return True
    if env_value.lower() == "false":
        return False

    # Check for numeric types
    try:
        # Try int first (more specific)
        if "." not in env_value and "e" not in env_value.lower():
            return int(env_value)
    except ValueError:
        pass

    try:
        # Try float
        return float(env_value)
    except ValueError:
        pass

    # Return as string if no other type matches
    return env_value

yaml.add_constructor('!env', env_constructor, Loader=yaml.SafeLoader)


def load_config(
        config_path: str,
):
    """
    Loads a YAML config file with !env support and returns parameters for each grader.
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        logger.debug(f"Loaded config from {config_path}: {config_data}")
    except (ValidationError, FileNotFoundError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file: {e}")
        sys.exit(1)

    return config_data

