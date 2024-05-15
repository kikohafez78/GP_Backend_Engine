# -*- coding: utf-8 -*-
"""
Main file.
This file should be executed to extract coreference informations.
"""
import argparse
import configparser
import csv
import logging
import multiprocessing
import os

from coreference.document import Document
from coreference.multipass_resolution import MultiPassResolution
from coreference.sieves.exact_match_sieve import ExactMatchSieve
from coreference.sieves.precise_construct_sieve import PreciseConstructsSieve
from coreference.sieves.relax_modifiers_sieve import StrictHeadRelaxModifiers
from coreference.sieves.relax_inclusion_sieve import StrictHeadRelaxInclusion
from coreference.sieves.strict_head_match_sieve import StrictHeadMatchSieve

# Make sure local files are found even if main.py is
# not executed from root directory of project.
ROOT = os.path.dirname(os.path.abspath(__file__))

# Map config names to sieve classes.
SIEVE_DICT = {"exact_match_sieve": ExactMatchSieve,
              "precise_constructs_sieve": PreciseConstructsSieve,
              "strict_head_match_sieve": StrictHeadMatchSieve,
              "strict_head_relax_modifiers": StrictHeadRelaxModifiers,
              "strict_head_relax_inclusion": StrictHeadRelaxInclusion}

logging.basicConfig(filename=os.path.join(ROOT, "main.log"),
                    level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')


def find_language_dir(path, language="english"):
    """Finds directory of the specified language and returns it.

    Args:
        path:
            A string or path-like object from which should be looked
            for the directory.
        language (str): Name of directory that should be looked for.

    Returns: path to language directory.

    Raises: OSError if directory couldn't be found.
    """
    for root, dirs, files in os.walk(path):
        if language in dirs:
            lang_dir = os.path.join(root, language)
            break
    else:
        logging.critical(f"No dir '{language}' found in {path}")
        raise OSError(f"Language Directory {language} not found.")
    return lang_dir


def extract_files(path, ext):
    """From specified path on extract all
    files that end with given extension recursively.

    Args:
        path (str): The path from which should be searched.
        ext (str): The file extension to extract.

    Returns:
        List of strings representing file paths.
    """
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                file_paths.append(os.path.join(root, file))
    if not file_paths:
        logging.warning(f"No files with extension '{ext}' found in {path}.")
    return file_paths


def get_sieves(config_file):
    """Extract sieves from a config file.

    Config file should have section 'Sieves' under
    which sieves are specified. Sieves are sorted according to their values.

    Args:
        config_file (str): A path to a config_file

    Returns:
        list of sieve instances.

    Raises:
        OSError if config file has no section 'Sieves.' or if given
        file does not exist.
    """
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Config file {config_file} not found.")
    sieve_key = "Sieves"
    sieves = []
    config = configparser.ConfigParser()
    config.read(config_file)
    if sieve_key not in config.sections():
        raise OSError(f"Invalid config file: Need key '{sieve_key}'")
    for key in config[sieve_key]:
        try:
            sieve = SIEVE_DICT[key]
        except KeyError:
            logging.warning(f"Unknown sieve '{key}' in config file.")
            continue
        else:
            try:
                position = config["Sieves"].getint(key)
            except ValueError:
                logging.warning("Invalid literal for int(). "
                                "Order of sieves might differ.")
                position = len(sieves)
            # -1 tells us to ingore a sieve.
            if position == -1:
                continue
            # Add sieve instance to list with specified position.
            sieves.append((sieve(), position))
    # Sort list according to values.
    sieves.sort(key=lambda x: x[1])
    # Return only sieves, not values.
    return list(map(lambda x: x[0], sieves))


def write_eval_summary(path, eval_list):
    """Writes a csv file that contains evaluation for each document.

    Args:
        path (str): A path where file should be written to.
        eval_list (list):
            A list of 4-tuples with
            (filename, precision value, recall value, f1 value)

    Returns: None
    """
    # No extracted values found.
    if not eval_list:
        return
    # Compute averages.
    sum_doc = len(eval_list)
    avg_recall = sum(recall for _, _, recall, _ in eval_list)/sum_doc
    avg_precision = sum(prec for _, prec, _, _ in eval_list)/sum_doc
    avg_f1 = sum(f1 for _, _, _, f1 in eval_list)/sum_doc
    # Write information to file.
    with open(path, "w", encoding="utf-8", newline="") as file:
        csv_writer = csv.writer(file, delimiter=";")
        csv_writer.writerow(["file", "precision", "recall", "f1-score"])
        csv_writer.writerow(["average", avg_precision, avg_recall, avg_f1])
        # Write values for each document to file.
        for filename, prec, rec, f1 in eval_list:
            csv_writer.writerow([filename, prec, rec, f1])
    logging.info(f"Summary file written to {path}")


def coreference_resolution(file, sieves, path_out):
    """Applies sieves to a document, writes results to file
    and returns evaluation metrics.

    Args:
        file (str): Path to file from Ontonotes-Corpus.
        sieves (list): List of sieve instances.
        path_out (str): Path to write output file to.

    Returns:
        4-tuple (filename (str), precision (float), recall (float), f1 (float))
    """
    doc = Document(file)
    coref = MultiPassResolution(doc, sieves)
    coref.resolve()
    file_name = f"{doc.filename()}.csv"
    # This might happen because we extract from nested directories.
    if file_name in os.listdir(path_out):
        logging.warning(f"File name {file_name} is a duplicate and was overwritten.")
    coref.to_csv(os.path.join(path_out, file_name))
    prec, rec, f1 = coref.evaluate()
    return (file_name, prec, rec, f1)


def main():
    parser = argparse.ArgumentParser(description="Coreference Resolution")
    parser.add_argument("in_dir", help="Input directory with conll files. Can be nested.")
    parser.add_argument("out_dir", help="Name of output directory.")
    parser.add_argument("--config", nargs="?", default="config.txt",
                        help="Path to config file. Default is 'config.txt'")
    parser.add_argument("--ext", nargs="?", default="conll",
                        help=("File extensions that should be considered. "
                              "Default is 'conll'"))
    parser.add_argument("--lang", nargs="?", default=None,
                        help=("A subdirectory of in_dir "
                              "from which files should be extracted. "
                              "If default is used, all subdirectories are searched."))
    args = parser.parse_args()
    # Get absolute paths to files.
    path_in = os.path.join(ROOT, args.in_dir)
    path_out = os.path.join(ROOT, args.out_dir)
    if args.lang:
        path_in = find_language_dir(path_in, args.lang)
    files = extract_files(path_in, args.ext)
    sieves = get_sieves(os.path.join(ROOT, args.config))
    # Avoid overwritting existing directories
    if os.path.isdir(path_out):
        raise OSError(f"Output directory {args.out_dir} already exists")
    os.mkdir(path_out)
    eval_list = []
    # Each document is independent so we can parallelize this process.
    with multiprocessing.Pool() as pool:
        # Specify arguments for coreference function
        data = [(file, sieves, path_out) for file in files]
        eval_list.extend(pool.starmap(coreference_resolution, data))
    print(f"Output files written to {path_out}")
    summary_file = os.path.join(path_out, "_summary.csv")
    write_eval_summary(summary_file, eval_list)


if __name__ == "__main__":
    try:
        main()
    except OSError as e:
        print(e)
