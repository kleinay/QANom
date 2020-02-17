import io

import pandas as pd

COL_NAMES = ["tok-id", "tok", "factuality", "pos", "dep-head", "dep-label", "lemma"]

def loadLineSplitFiles(file_name, deserialize_block_func=lambda blockString: blockString,
                       skip_first_line=False,
                       encoding="utf-8"):
    """ Return an iterable of (deserialized) blocks of a file, where blocks are delimited by an empty line.
       Useful for loading CoNLL or CoNLL-like annotation files.
    :param file_name:
    :param deserialize_block_func:
        If provided, the function will yield the block deserialized into an object.
    :param skip_first_line:
    :param encoding:
    :return: yield deserialize_block_func(block) for block in the file
    """
    with io.open(file_name, "r", encoding=encoding) as f:
        # skip first line in file in case skip_first_line
        if skip_first_line:
            firstLine = next(f)
        block = ""
        for line in f:
            if not line.strip(): # empty line - declaring a new block
                # wrap and parse last block
                yield deserialize_block_func(block)

                # logging.debug("collected annotation for sentence: " + <sentence-string>)
                block = ""
            else:   # continue with collecting the block
                block += line

def df_from_table_string(table_string, col_names = None):
    """ Return a pd.DataFrame from a multi-line table string."""
    from io import StringIO
    strIO = StringIO(table_string)
    return pd.read_table(strIO, header=None, names=col_names)
