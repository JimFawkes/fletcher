"""Project Fletcher Configuration

Define all necessary configuration in this file.
"""
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

ROOT_DIR = Path.cwd()
PROJECT_DIR = ROOT_DIR / "mcnulty"


def singleton(cls):
    """Ensure that only one instance of the class ever exists."""
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return getinstance


@singleton
class Config:
    """Configurations for mcnulty"""

    def __init__(self):
        self.root_dir = ROOT_DIR
        self.project_dir = PROJECT_DIR
        self.model_dir = PROJECT_DIR / "models"
        self.data_dir = os.getenv("DATA_DIR", default=None) or self.root_dir / "data/"
        self.log_dir = os.getenv("LOG_DIR", default=None) or self.root_dir / "logs/"

    def __repr__(self):
        return "Config()"


