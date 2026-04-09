"""File system utility functions."""

import os
import glob


def create_folder(folder_path: str) -> None:
    """Create a folder if it does not already exist.

    Parameters
    ----------
    folder_path : str
        Path to the folder to create.
    """
    if os.path.exists(folder_path):
        print(f"Folder already exists: {folder_path}")
    else:
        os.mkdir(folder_path)
        print(f"Folder created: {folder_path}")


def get_most_recent_file(directory_pattern: str) -> str:
    """Return the most recently created file matching a glob pattern.

    Parameters
    ----------
    directory_pattern : str
        Glob pattern to match files (e.g., '/path/to/dir/*.csv').

    Returns
    -------
    str
        Path to the most recently created file.
    """
    return max(glob.glob(directory_pattern), key=os.path.getctime)
