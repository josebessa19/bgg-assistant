import pytest

from bgg.config import settings


@pytest.fixture
def app_settings():
    return settings
