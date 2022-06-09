# See https://docs.pytest.org/en/6.2.x/fixture.html#conftest-py-sharing-fixtures-across-multiple-files 
# for the details of the purpose of this file.

import pytest

def pytest_addoption(parser):
    parser.addoption("--timesensitive", action="store_true", default=False, help="Skips tests with timeouts")

def pytest_configure(config):
    config.addinivalue_line("markers", "timesensitive: test is time sensitive and should not run with Coverage")

def pytest_collection_modifyitems(config, items):
    if config.getoption("--timesensitive"):
        # --timesensitive given in cli: do not skip time sensitive tests
        return
    time_sensitive = pytest.mark.skip(reason="need --timesensitive option to run")
    for item in items:
        if "timesensitive" in item.keywords:
            item.add_marker(time_sensitive)
    