"""Целостность артефакта: sha256 пишется в манифест при обучении и сверяется до unpickle.

joblib-файл — это pickle, а ``joblib.load`` исполняет код прямо при чтении.
Поэтому байты сверяются с манифестом ДО загрузки: подменённый или битый файл
отвергается, не успев выполниться.

Хеш лежит в том же релизе, что и файл, — значит, он ловит порчу, перепутанные
версии и неполную замену. От злоумышленника, который перезальёт оба ассета,
защищают attestation (``gh attestation verify`` в docker.yml) и immutable releases.
"""

import hashlib
from pathlib import Path

_CHUNK = 1 << 20  # 1 МиБ: стек весит десятки МБ, целиком в память не читаем


class ArtifactIntegrityError(RuntimeError):
    """Артефакт не совпал с манифестом — загружать его нельзя."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def artifact_record(path: Path) -> dict:
    """Запись для манифеста: имя, размер и sha256 файла модели."""
    path = Path(path)
    return {"file": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def verify_artifact(path: Path, manifest: dict) -> None:
    """Бросает ArtifactIntegrityError, если файл не тот, что описан в манифесте.

    Манифест без поля ``artifact`` тоже отвергается: «не с чем сверить» не
    должно молча превращаться в «загрузить не глядя».
    """
    path = Path(path)
    record = manifest.get("artifact")
    if not record or "sha256" not in record:
        raise ArtifactIntegrityError(
            f"в манифесте нет artifact.sha256 — {path.name} не загружаю. "
            "Артефакт обучен до введения проверки: нужен новый релиз модели."
        )
    if record.get("file") != path.name:
        raise ArtifactIntegrityError(
            f"манифест описывает {record.get('file')!r}, а загружается {path.name!r}"
        )
    actual = file_sha256(path)
    if actual != record["sha256"]:
        raise ArtifactIntegrityError(
            f"sha256 {path.name} не совпал с манифестом: "
            f"ожидался {record['sha256'][:16]}…, получен {actual[:16]}…"
        )
