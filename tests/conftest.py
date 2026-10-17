import pytest

from tapedigitizer import exporter, recognize


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Tests never call the real AudD service, use your token, or write to your Drive."""
    monkeypatch.delenv("AUDD_API_TOKEN", raising=False)
    monkeypatch.setattr(recognize, "TOKEN_FILE", tmp_path / "no-token.txt")
    drive = tmp_path / "Drive" / "Music" / "Tapes"
    monkeypatch.setattr(exporter, "find_drive_folder", lambda: drive)
    from tapedigitizer import cli

    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "config.json")
    return drive
