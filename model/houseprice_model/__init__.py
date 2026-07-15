from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("houseprice-model")
except PackageNotFoundError:  # пакет не установлен (например, статический анализ)
    __version__ = "0.0.0"
