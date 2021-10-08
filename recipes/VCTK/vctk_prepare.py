"""
Data preparation.

Download: https://datashare.ed.ac.uk/handle/10283/3443

Author
------
Mirco Ravanelli, Ju-Chieh Chou, Loren Lugosch 2020
"""

import os
import csv
import logging
import torchaudio
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import (
    load_pkl,
    save_pkl,
)

logger = logging.getLogger(__name__)
OPT_FILE = "opt_vctk_prepare.pkl"
SPK_FILE = "speaker-info.txt"
SAMPLERATE = 48000


def prepare_vctk(
    data_folder,
    save_folder,
    tr_spks=None,
    dev_spks=[],
    te_spks=[],
    select_n_sentences=None,
):
    """
    This function prepares the csv files for the VCTK dataset.
    Download link: https://datashare.ed.ac.uk/handle/10283/3443

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original VCTK dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    tr_spks : list
        Default : None
        if not None, the list contains train speakers.
        If None, the training speakers will be all the speakers except for those in dev and test.
    dev_spks : list
        List of dev speakers.
    te_spks : list
        List of test speakers.
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.

    Example
    -------
    >>> data_folder = 'datasets/VCTK-Corpus'
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_vctk(data_folder, save_folder)
    """
    conf = {
        "select_n_sentences": select_n_sentences,
        "tr_spks": tr_spks,
        "dev_spks": dev_spks,
        "te_spks": te_spks,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # training split and the corresponding speakers
    splits, spk_set = [], []
    if tr_spks is None or len(tr_spks) > 0:
        splits.append("train")
        spk_set.append(tr_spks)
    if len(dev_spks) > 0:
        splits.append("dev")
        spk_set.append(dev_spks)
    if len(te_spks) > 0:
        splits.append("test")
        spk_set.append(te_spks)

    # Check if this phase is already done (if so, skip it)
    if skip(splits, save_folder, conf):
        print("Skipping preparation, completed in previous run.")
        return
    else:
        print("Data_preparation...")

    # Additional checks to make sure the data folder contains VCTK corpus
    # check_vctk_folders(data_folder, splits)

    # create csv files for each split
    for split_index in range(len(splits)):

        split, spks = splits[split_index], spk_set[split_index]

        if split == "train" and tr_spks is None:
            wav_lst = get_all_files(
                os.path.join(data_folder, "wav48"),
                match_and=[".wav"],
                exclude_or=dev_spks + te_spks,
            )

            text_lst = get_all_files(
                os.path.join(data_folder, "txt"),
                match_and=[".txt"],
                exclude_or=dev_spks + te_spks,
            )
        else:
            wav_lst = get_all_files(
                os.path.join(data_folder, "wav48"),
                match_and=[".wav"],
                match_or=spks,
            )

            text_lst = get_all_files(
                os.path.join(data_folder, "txt"),
                match_and=[".txt"],
                match_or=spks,
            )
        text_dict = text_to_dict(text_lst)

        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)

        create_csv(
            save_folder, wav_lst, text_dict, split, n_sentences,
        )

    # saving options
    save_pkl(conf, save_opt)


def create_csv(
    save_folder, wav_lst, text_dict, split, select_n_sentences,
):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    print(msg)

    csv_lines = [
        [
            "ID",
            "duration",
            "wav",
            "wav_format",
            "wav_opts",
            "spk_id",
            "spk_id_format",
            "spk_id_opts",
            "wrd",
            "wrd_format",
            "wrd_opts",
            "char",
            "char_format",
            "char_opts",
        ]
    ]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for wav_file in wav_lst:

        snt_id = wav_file.split("/")[-1].replace(".wav", "")
        spk_id = snt_id.split("_")[0]
        wrds = text_dict[snt_id]

        signal, fs = torchaudio.load(wav_file)
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE

        # replace space to <space> token
        chars_lst = [c for c in wrds]
        chars = " ".join(chars_lst)

        csv_line = [
            snt_id,
            str(duration),
            wav_file,
            "wav",
            "",
            spk_id,
            "string",
            "",
            str(" ".join(wrds.split("_"))),
            "string",
            "",
            str(chars),
            "string",
            "",
        ]

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1

        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s sucessfully created!" % (csv_file)
    print(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the vctk data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the seave directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip


def text_to_dict(text_lst):
    """
    This converts lines of text into a dictionary-

    Arguments
    ---------
    text_lst : str
        Path to the file containing the VCTK text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.

    """
    # Initialization of the text dictionary
    text_dict = {}
    # Reading all the transcription files is text_lst
    for file in text_lst:
        snt_id = file.split("/")[-1].replace(".txt", "")
        with open(file, "r") as f:
            # Reading all line of the transcription file
            # Converting to lower case
            text = f.read().strip().lower().split(" ")
            text_dict[snt_id] = "_".join(text)
    return text_dict


# def check_librispeech_folders(data_folder, splits):
#    """
#    Check if the data folder actually contains the LibriSpeech dataset.
#
#    If it does not, an error is raised.
#
#    Returns
#    -------
#    None
#
#    Raises
#    ------
#    OSError
#        If LibriSpeech is not found at the specified path.
#    """
#    # Checking if all the splits exist
#    for split in splits:
#        split_folder = os.path.join(data_folder, split)
#        if not os.path.exists(split_folder):
#            err_msg = (
#                "the folder %s does not exist (it is expected in the "
#                "Librispeech dataset)" % split_folder
#            )
#            raise OSError(err_msg)


def read_spk_info(spk_info):
    """
    Read the speaker-info.txt provided by the corpus.

    Arguments
    ---------
    spk_info : str
        Path to the file speaker-info.txt.

    Returns
    -------
    dict
        The dictionary containing the information of each speakers (AGE, GENDER, ACCENT).

    """
    spk_info_dict = {}
    with open(spk_info, "r") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            spk, age, gender, accent = line.strip().split(None, maxsplit=3)
            spk_info_dict[spk] = {
                "AGE": age,
                "GENDER": gender,
                "ACCENTS": accent,
            }
    return spk_info_dict
