"""
Script implementing a Database Manager for the SMM script
"""
import numpy as np
import numpy.typing as npt
from shutil import copyfile
from typing import List, Union
import logging
import os
import h5py

class H5Database:
    def __init__(self, db_file: str) -> None:
        logging.info(f"Loading Database: {db_file}")
        self._db_file: str = db_file
        self._prepend_path: str = os.path.dirname(db_file)
        self._content: List[str] = []
        with h5py.File(self._db_file, "a") as db:
            if "root" not in db.keys():
                db.create_group("root")
            print(db.keys())
            root = db["root"]
            self._content = [mat_name for mat_name in root.keys()]
        logging.debug(f"DB Content: {self._content}")

    def add_content(self, name: str, data: npt.NDArray) -> int:
        """
        Add content to the database
        Return (0, -1) on success or fail
        """
        if name in self._content:
            return -1
        with h5py.File(self._db_file, "r+") as db:
            root = db["root"]
            mat_group = root.create_group(name)
            mat_group.create_dataset("data", data.shape, data=data)
            mat_group.create_dataset("min", (data.shape[1],), data=np.min(data, axis=0))
            mat_group.create_dataset("max", (data.shape[1],), data=np.max(data, axis=0))
        self._content.append(name)
        return 0

    def rmv_content(self, name: Union[str, int]):
        rmv_material: str = name if isinstance(name, str) else self._content[name]
        rmv_index = self._content.index(rmv_material)
        logging.debug(f"Removing {rmv_material}::{rmv_index}")
        print(f"Removing {rmv_material}::{rmv_index}")
        with h5py.File(self._db_file, "r+") as db:
            root = db["root"]
            del root[rmv_material]
        self._content.pop(rmv_index)

    """ Properties """

    @property
    def content(self):
        return self._content


class Database:
    def __init__(self, db_file: str) -> None:
        """
        Load all contents in database to variable
        """
        logging.info(f"Loading Database '{db_file}'")
        self.db_file: str = db_file
        self.prepend_path: str = os.path.dirname(db_file)
        with open(self.db_file) as file:
            self.content: List[str] = [
                line for line in file.read().splitlines() if len(line) > 0
            ]
        logging.debug(f"{self.content=}")

    def add_content(self, name: str, data: npt.NDArray) -> None:
        """Add content to the Database"""
        with open(self.db_file, "a") as db_file:
            db_file.write(name + "\n")
        self.content.append(name)
        np.savetxt(os.path.join(self.prepend_path, name + ".txt"), data)
        logging.info(f"Added material '{name}' to Database")

    def rmv_content(self, name: str) -> int:
        """Remove material from the Database"""
        with open(self.db_file, "r") as db_file:
            lines: List[str] = db_file.readlines()
        with open(self.db_file, "w") as db_file:
            for line in lines:
                if line.strip("\n") != name:
                    db_file.write(line)
        os.remove(os.path.join(self.prepend_path, name + ".txt"))
        logging.info(f"Remove material '{name}' from Database")
        for index, mat_name in enumerate(self.content):
            if mat_name == name:
                self.content.pop(index)
        return 0

    def export_content(self, name: str, export_path: str) -> None:
        """
        Export content of DB material
        """
        copyfile(os.path.join(self.prepend_path, name + ".txt"), export_path)
        logging.info(f"Export material '{name}' from Database")

    def find_index(self, name: str) -> int:
        """
        Find index for a particular element in the database
        Return:
            index > 0: Index
            index < 0: No index found
        """
        try:
            index = self.content.index(name)
        except ValueError:
            index = -1
        return index

    def __getitem__(self, name: Union[str, int]) -> npt.NDArray:
        """Syntax to access a certain item in DB Database[index/name]"""
        logging.debug(f"Accessing {name}")
        if isinstance(name, int):
            return np.loadtxt(
                os.path.join(self.prepend_path, self.content[name] + ".txt")
            )
        elif isinstance(name, str):
            return np.loadtxt(os.path.join(self.prepend_path, name + ".txt"))


if __name__ == "__main__":
    db = Database("database")
    print(db.content)
