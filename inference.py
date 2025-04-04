from Experiment.experiment_memory_vqmivc_pretrain_pseudo import Facevoice_memory_vqmivc_pretrain_pseudo
from configparser import ConfigParser
import warnings
import argparse
import os
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = ConfigParser()
    config.read(os.path.join(args.config_file))

    cur_exp = Facevoice_memory_vqmivc_pretrain_pseudo(config)
    cur_exp.run_inference()