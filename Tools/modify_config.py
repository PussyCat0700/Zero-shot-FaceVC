from configparser import ConfigParser
import argparse
from glob import glob
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',required=True)
    parser.add_argument('--inference_gpu',required=True, type=str)
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()
    return args
    

if __name__ == '__main__':
    args = parse_args()
    conf_path = args.config_file
    config = ConfigParser()
    config.read(conf_path)
    print('------------------------------')
    print(conf_path)

    try:
        checkpoint = config.get("output", "checkpoint")
    except:
        print(' Change config file done ')
        config.set("input","is_train","False")
        config.set("output","checkpoint",os.path.join(args.output_root, "checkpoint.pt"))
        config.set("hparams","infer_gpu", args.inference_gpu)
        with open(conf_path,'w') as configfile:
            config.write(configfile)