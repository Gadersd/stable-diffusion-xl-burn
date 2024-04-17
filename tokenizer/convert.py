import json
import argparse

def write_to_file(file_name, data):
    with open(file_name, 'w') as f:
        for item in data:
            f.write("%s\n" % item)

def process_config_file(config_file):
    with open(config_file) as json_file:
        data = json.load(json_file)

    vocab = data['model']['vocab']
    merges = data['model']['merges']

    write_to_file('vocab.txt', vocab)
    write_to_file('merges.txt', merges)

if __name__== "__main__":
    parser = argparse.ArgumentParser(description='Process json configuration file.')
    parser.add_argument('config_file', type=str, help='Configuration file name')

    args = parser.parse_args()

    process_config_file(args.config_file)