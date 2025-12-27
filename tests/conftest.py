import pytest
import logging
import os

# Set up logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'tests.log')),
        logging.StreamHandler()
    ]
)

# Capture and log pytest output
@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    logging.info("Pytest configuration started")
    config.addinivalue_line("markers", "log: mark test to log output")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    logging.info(f"Setting up test: {item.name}")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item):
    logging.info(f"Running test: {item.name}")

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item):
    logging.info(f"Tearing down test: {item.name}")

@pytest.hookimpl(tryfirst=True)
def pytest_unconfigure(config):
    logging.info("Pytest configuration ended")