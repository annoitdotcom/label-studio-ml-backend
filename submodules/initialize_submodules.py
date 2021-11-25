#!/usr/bin/env python
"""Initializes required submodules.
This script should be run only once after the models have been downloaded.
"""
import json
import os
import shutil
import subprocess
from pathlib import Path

curr_path = Path(__file__).resolve()
top_dir = curr_path.parent.parent


def init_submodules():
    """Makes python module from each git submodule."""
    with open(os.path.join(top_dir, "submodules", "gitmodules.json"), "r") as fp:
        gitmodules_json = json.load(fp)

    for name, value in gitmodules_json.items():
        url, sha = value["url"], value["sha1_hash"]
        print(f"Adding submodule {name} from {url}")
        submodule_path = top_dir / name

        print(output)
        print(f"Resetting submodule name to commit {sha}")
        try:
            output = subprocess.check_output(f"git reset --hard {sha}",
                                             stderr=subprocess.STDOUT, shell=True, cwd=submodule_path)
        except subprocess.CalledProcessError as cpe:
            raise RuntimeError(
                f"Command '{cpe.cmd}' return with error (code {cpe.returncode}): {cpe.output}")
        print(output)

        # Delete the git directory for the submodules since this script is meant to mainly be used to build the
        # container image.
        shutil.rmtree(submodule_path / ".git")

        # Make submodule directory as python module.
        with open(submodule_path / "__init__.py", "a"):
            print(f"Initializing submodule {name}.")


if __name__ == "__main__":
    init_submodules()
    print("Initialization finished.")
