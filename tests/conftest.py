import pytest
import os
import shutil
from typing import Generator

# This file contains "fixtures" for our tests.
# Fixtures are helper functions that set up the environment before a test runs
# and clean it up afterwards.

@pytest.fixture
def temp_output_dir() -> Generator[str, None, None]:
    """
    Creates a temporary directory for test output and deletes it afterwards.
    This ensures our tests don't leave behind messy files.
    """
    # 1. Setup: Create a unique directory name
    dir_path = "tests/temp_output"
    os.makedirs(dir_path, exist_ok=True)
    
    # 2. Yield: Give the path to the test function
    yield dir_path
    
    # 3. Teardown: Delete the directory and everything in it
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)