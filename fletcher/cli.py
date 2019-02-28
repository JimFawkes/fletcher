"""CLI to run all aspects of the mcnulty pipeline


"""

import argparse
import sys
from loguru import logger

import train
from config import Config

_log_file_name = __file__.split("/")[-1].split(".")[0]
logger.add(f"logs/{_log_file_name}.log", rotation="1 day")

config = Config()

help_text = """
Project Fletcher - Metis

The pipeline consists fo the following steps:
    1.) 
    ....


"""

epilog = """

Written as first draft by Moritz Eilfort.

"""

parser = argparse.ArgumentParser(
    prog="fletcher",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=help_text,
    epilog=epilog,
)

parser.add_argument("--train_pv", action="store_true", help="Train Paragraph Vectors.")

parser.add_argument(
    "--train_model", action="store_true", help="Train Paragraph Vectors."
)

parser.add_argument(
    "--train_from_pickle",
    action="store_true",
    help="Train Paragraph Vectors from a pickled docs file.",
)
parser.add_argument(
    "--run_preprocessing",
    action="store_true",
    help="Run Preprocessing on all news articles.",
)
#
# parser.add_argument(
#     "--create_tables",
#     action="store_true",
#     help="Create the necessary tables in the database.",
# )
#
# parser.add_argument(
#     "--drop_tables", action="store_true", help="Drop all tables in the database."
# )
#
# parser.add_argument(
#     "--import_weather", help="Import weather data for a given year.", type=str
# )
#
# parser.add_argument("--run_pipeline", action="store_true", help="Run pipeline.")
#
# parser.add_argument("--upload_logs", action="store_true", help="Upload logs to S3.")


def train_paragraph_vectors():
    return train.train()


def train_model():
    return train.train_model()


def run_preprocessing():
    return train.run_preprocessing()


def train_from_pickle():
    return train.train_from_pickle()


def main():
    args = parser.parse_args()
    # logger.debug(f"Starting Pipline")

    # Add Commands

    if args.train_pv:
        logger.info(f"Train Paragraph Vectors")
        model = train_paragraph_vectors()
        sys.exit(0)

    if args.train_model:
        logger.info("Train model")
        model = train_model()
        sys.exit(0)

    if args.run_preprocessing:
        logger.info("Run Preprocessing")
        docs = run_preprocessing()
        sys.exit(0)

    if args.train_from_pickle:
        logger.info("Train Paragraph Vectors from pickled docs file.")
        model = train_from_pickle()
        sys.exit(0)

    sys.exit(1)


if __name__ == "__main__":
    main()

