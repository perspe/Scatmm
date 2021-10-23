"""
Script implementing a Database Manager for the SMM script
"""
import numpy as np
from shutil import copyfile
import os


class Database():
    def __init__(self, db_file):
        """
        Load all contents in database to variable
        """
        self.db_file = db_file
        self.prepend_path: str = os.path.dirname(db_file)
        with open(self.db_file) as db_file:
            self.content = db_file.read().splitlines()

    def add_content(self, name, data):
        with open(self.db_file, "a") as db_file:
            db_file.write(name)
        self.content.append(name)
        np.savetxt(os.path.join(self.prepend_path, name + ".txt"), data)

    def rmv_content(self, name):
        with open(self.db_file, "r") as db_file:
            lines = db_file.readlines()
        with open(self.db_file, "w") as db_file:
            for line in lines:
                if line.strip("\n") != name:
                    db_file.write(line)
        os.remove(os.path.join(self.prepend_path, name + ".txt"))
        for index, mat_name in enumerate(self.content):
            if mat_name == name:
                self.content.pop(index)
        return 0

    def export_content(self, name, export_path):
        copyfile(os.path.join(self.prepend_path, name + ".txt"), export_path)

    def find_index(self, name):
        for index, db_name in enumerate(self.content):
            if db_name == name:
                return index

    def __getitem__(self, name):
        if type(name) == int:
            return np.loadtxt(
                os.path.join(self.prepend_path, self.content[name] + ".txt"))
        elif type(name) == str:
            return np.loadtxt(os.path.join(self.prepend_path, name + ".txt"))


if __name__ == "__main__":
    db = Database("database")
    print(db.content)
