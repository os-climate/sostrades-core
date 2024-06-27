"""
Pytest test configuration
"""
import pytest
from _pytest.config import Config
from _pytest.nodes import Item


def pytest_configure(config: Config):
    """
    Pytest dynamic configuration
    :param config:
    :return:
    """
    config.addinivalue_line("markers", "l0: mark l0 tests")
    config.addinivalue_line("markers", "l1: mark l1 tests")
    config.addinivalue_line("markers", "l2: mark l2 tests")
    config.addinivalue_line("markers", "uc: mark usecase tests")


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """
    On-the-fly test tagging
    :param config:
    :param items:
    :return:
    """
    l_tests_marker = {
        "/l0_": pytest.mark.l0,
        "/l1_": pytest.mark.l1,
        "/l2_": pytest.mark.l2,
    }
    for item in items:
        for matching_name, marker in l_tests_marker.items():
            if matching_name in item.parent.nodeid:
                item.add_marker(marker)
