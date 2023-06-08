import argparse
from collections import namedtuple

# Declaring namedtuple()
Config = namedtuple('Config', ['pdf_path',
                               'persist_dir',
                               'collection_name'])


def fetch_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pdf_path", default='data/mroark_resume.pdf',
                        nargs=1, help="File to ingest")
    parser.add_argument("-n", "--name", default='pdf_inspect',
                        nargs=1, help="ChromaDB collection name")
    parser.add_argument("-d", "--dir", default="data/chroma",
                        nargs=1, help="ChromaDB persistence dir")
    # TODO: add logging
    args = parser.parse_args()
    return Config(args.pdf_path, args.dir, args.name)
