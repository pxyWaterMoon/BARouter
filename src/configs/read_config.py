import argparse
import yaml
import os


def argument_parser():
    """
    Parse command line arguments for configuring the system.
    """
    parser = argparse.ArgumentParser(description="System Configuration Parser")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./src/configs/xgb_AUPD_routerbench.yaml",
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="Name of the project for logging purposes."
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Path to the log file where system logs will be saved."
    )
    parser.add_argument(
        "--budget",
        type=float,
        help="Total budget for the system."
    )
    parser.add_argument(
        "--T",
        type=int,
        help="Number of rounds for the system."
    )
    parser.add_argument(
        "--allow_null",
        type=bool,
        default=False,
        help="Allow null action or not"
    )
    parser.add_argument(
        "--mu",
        type=float,
        help="The mu parameter for the Carrot algorithm"
    )
    return parser.parse_args()

def load_config(args):
    """
    Configure the system based on the provided arguments and configuration file.
    """
    config_path = args.config_path
    
    # Check if the config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    # Load the configuration from the YAML file
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    # update the config with arguments
    if args.project_name:
        config["project_name"] = args.project_name
    if args.budget:
        config["budget"] = args.budget
    if args.T:
        config["T"] = args.T
    if args.log_dir:
        config["log_dir"] = args.log_dir
    if args.allow_null:
        config["agent"]["allow_null"] = args.allow_null
    if args.mu:
        config["agent"]["mu"] = args.mu
    return config
