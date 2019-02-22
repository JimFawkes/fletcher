"""CLI to run all aspects of the mcnulty pipeline


"""

import argparse
import sys
from loguru import logger

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

# parser.add_argument(
#     "--setup_db", "-s", help="Setup the database. Create necessary Tables."
# )
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


def main():
    args = parser.parse_args()
    logger.debug(f"Starting Pipline")

    # Add Commands

 #    if args.upload_logs:
 #        logger.warning(f"Upload logs part is not yet implemented.")
 #        sys.exit(1)


if __name__ == "__main__":
    main()

