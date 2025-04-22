import sys
from pathlib import Path

from src.config_reader import read_config
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing

sys.path.append(str(Path(__file__).resolve().parent.parent))

data_ingestion = DataIngestion(read_config("config/config.yaml"))
data_ingestion.run()

data_processing = DataProcessing(read_config("config/config.yaml"))
data_processing.run()
