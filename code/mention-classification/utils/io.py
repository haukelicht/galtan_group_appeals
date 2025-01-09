import os
import json

import zipfile
from io import TextIOWrapper

from typing import Any, List, Optional, Literal


def read_jsonlines(
        file: str, 
        nrows: Optional[int]=None, 
        skip: Optional[int]=None, 
        mode: Literal['r', 'rb']='r'
    ) -> List[Any]:
    """Read a JSONlines file

    Args:
        file (str): path to the file
        nrows (int, optional): number of rows to read
        skip (int, optional): number of rows to skip
        mode (str, optional): read mode ('r' or 'rb')

    Returns:
        List[Any]: list of data read from the file
    """
    # inspired by https://stackoverflow.com/a/11482347
    assert os.path.exists(file), f"File not found: {file}"
    if skip is None:
        skip = 0
    is_zip_file = zipfile.is_zipfile(file)
    data, issues = [], []
    with zip_open(file, mode=mode) if is_zip_file else open(file, mode) as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= skip + nrows:
                break
            if i < skip:
                continue
            # handle exception if parsing error json.JSONDecodeError
            try: 
                line = json.loads(line)
            except json.JSONDecodeError:
                issues.append(i)
                data.append(None)
            else:
                data.append(line)
    if issues:
        issues_str = ', '.join(map(str, issues[:6]))
        if len(issues) > 6:
            issues_str += ', ...'
        print(f"Warning: Failed to parse {len(issues)} lines: {issues_str} . Data for these lines is `None` in the output.")
    return data

def write_jsonlines(
        data: List, 
        file: str, 
        overwrite: bool=False, 
        append: bool=False
    ):
    """Write a list object to a JSONlines file

    Args:
        data (List): list of data to write
        file (str): path to the file
        overwrite (bool, optional): whether to overwrite the file if it exists
        append (bool, optional): whether to append to the file if it exists
    
    Raises:
        ValueError: if both `overwrite` and `append` are `True`
    """
    assert not os.path.exists(file) or overwrite or append, f"File already exists: {file}"
    mode = 'w' if overwrite else 'a'
    if overwrite and append:
       raise ValueError("Cannot overwrite and append at the same time")
    if append and not os.path.exists(file):
        mode = 'w'
    lines = [json.dumps(d) for d in data]
    with open(file, mode) as f:
        if mode == 'a': f.write('\n')
        f.write('\n'.join(lines))

