import pytest

from src.configs import config
from src.Tools import storage


# Point the app at a throwaway SQLite file for each test.
@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(config, "DB_PATH", db_path)
    storage.init_db()
    return db_path
