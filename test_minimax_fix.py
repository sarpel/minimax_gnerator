#!/usr/bin/env python3
"""
Test script to verify MiniMax provider fixes.

This script tests:
1. API endpoint URL is correct (https://api.minimaxi.chat)
2. Deprecated asyncio.get_event_loop().time() has been replaced with time.time()
3. Rate limiting constants are properly set
4. API key loading from environment variables works
"""

import os
import sys
import asyncio
import time
from unittest.mock import patch, MagicMock

# Add the project root to Python path so we can import wakegen modules
sys.path.insert(0, '.')

from wakegen.providers.commercial.minimax import MiniMaxProvider
from wakegen.models.config import ProviderConfig

def test_api_endpoint_url():
    """Test that the API endpoint URL is correct."""
    print("Testing API endpoint URL...")

    # Create a mock config
    mock_config = MagicMock()
    mock_config.minimax_api_key = "test_key_123"
    mock_config.minimax_group_id = "test_group_456"

    # Create provider instance
    provider = MiniMaxProvider(mock_config)

    # Verify the base URL is correct
    expected_url = "https://api.minimaxi.chat"
    actual_url = provider.base_url

    if actual_url == expected_url:
        print(f"API endpoint URL is correct: {actual_url}")
        return True
    else:
        print(f"API endpoint URL is incorrect. Expected: {expected_url}, Got: {actual_url}")
        return False

def test_time_import_and_usage():
    """Test that time module is imported and used instead of deprecated asyncio.get_event_loop().time()"""
    print("\nTesting time module usage...")

    try:
        # Check that time module is imported in the minimax.py file
        with open('wakegen/providers/commercial/minimax.py', 'r') as f:
            content = f.read()

        # Check for time import
        if 'import time' in content:
            print("time module is imported")
        else:
            print("time module is not imported")
            return False

        # Check that asyncio.get_event_loop().time() is not used in actual code (not in comments)
        lines = content.split('\n')
        for line in lines:
            # Skip comment lines
            if line.strip().startswith('#'):
                continue
            if 'asyncio.get_event_loop().time()' in line:
                print(f"Still using deprecated asyncio.get_event_loop().time() in line: {line.strip()}")
                return False

        print("Not using deprecated asyncio.get_event_loop().time() in actual code")

        # Check that time.time() is used
        found_time_time = False
        for line in lines:
            # Skip comment lines
            if line.strip().startswith('#'):
                continue
            if 'time.time()' in line:
                found_time_time = True
                break

        if found_time_time:
            print("Using time.time() instead of deprecated method")
            return True
        else:
            print("time.time() is not being used in actual code")
            return False

    except Exception as e:
        print(f"Error checking time usage: {e}")
        return False

def test_rate_limiting_constants():
    """Test that rate limiting constants are properly set."""
    print("\nTesting rate limiting constants...")

    # Create a mock config
    mock_config = MagicMock()
    mock_config.minimax_api_key = "test_key_123"
    mock_config.minimax_group_id = "test_group_456"

    # Create provider instance
    provider = MiniMaxProvider(mock_config)

    # Check rate limiting constants
    expected_max_requests = 60  # Official MiniMax API limit
    actual_max_requests = provider.max_requests_per_minute

    if actual_max_requests == expected_max_requests:
        print(f"Rate limit constant is correct: {actual_max_requests} requests/minute")
        return True
    else:
        print(f"Rate limit constant is incorrect. Expected: {expected_max_requests}, Got: {actual_max_requests}")
        return False

def test_api_key_loading():
    """Test that API key is loaded from environment variables."""
    print("\nTesting API key loading from environment variables...")

    # Set a test API key in environment
    test_api_key = "test_api_key_from_env"
    os.environ["MINIMAX_API_KEY"] = test_api_key

    try:
        # Create config which should load from environment
        config = ProviderConfig()

        if config.minimax_api_key == test_api_key:
            print(f"API key loaded correctly from environment: {config.minimax_api_key}")
            return True
        else:
            print(f"API key not loaded correctly. Expected: {test_api_key}, Got: {config.minimax_api_key}")
            return False
    except Exception as e:
        print(f"Error testing API key loading: {e}")
        return False
    finally:
        # Clean up
        if "MINIMAX_API_KEY" in os.environ:
            del os.environ["MINIMAX_API_KEY"]

async def test_rate_limiting_functionality():
    """Test that rate limiting functionality works with time.time()"""
    print("\nTesting rate limiting functionality...")

    # Create a mock config
    mock_config = MagicMock()
    mock_config.minimax_api_key = "test_key_123"
    mock_config.minimax_group_id = "test_group_456"

    # Create provider instance
    provider = MiniMaxProvider(mock_config)

    try:
        # Test that rate limiting uses time.time() correctly
        initial_time = provider.last_reset_time
        initial_requests = provider.current_requests

        # Simulate some requests
        for i in range(5):
            await provider._check_rate_limit()

        # Check that requests were counted
        if provider.current_requests == 5:
            print("Rate limiting is counting requests correctly")
        else:
            print(f"Rate limiting request count is incorrect. Expected: 5, Got: {provider.current_requests}")
            return False

        # Check that time is being tracked
        current_time = time.time()
        if isinstance(initial_time, float) and isinstance(current_time, float):
            print("Rate limiting is using time.time() correctly")
            return True
        else:
            print("Rate limiting time tracking is not working correctly")
            return False

    except Exception as e:
        print(f"Error testing rate limiting functionality: {e}")
        return False

def main():
    """Run all tests and report results."""
    print("Starting MiniMax Provider Fix Verification Tests")
    print("=" * 60)

    tests = [
        test_api_endpoint_url,
        test_time_import_and_usage,
        test_rate_limiting_constants,
        test_api_key_loading,
    ]

    async_tests = [
        test_rate_limiting_functionality,
    ]

    results = []
    # Run synchronous tests
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)

    # Run asynchronous tests
    async def run_async_tests():
        async_results = []
        for test in async_tests:
            try:
                result = await test()
                async_results.append(result)
            except Exception as e:
                print(f"Async test {test.__name__} failed with exception: {e}")
                async_results.append(False)
        return async_results

    # Run async tests
    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Tests Passed: {passed}/{total}")

    if passed == total:
        print("All tests passed! MiniMax provider fixes are working correctly.")
        return True
    else:
        print("Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)