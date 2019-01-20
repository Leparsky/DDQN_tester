import os
import csv
from datetime import datetime
def WritetoCsvFile(filename, list):
    sd=datetime.strftime(datetime.now(), "%d.%m.%Y %H:%M:%S")
    dir_path = os.path.dirname(filename)
    if dir_path == "":
        dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(dir_path + filename, "a") as output:
        wr = csv.writer(output, delimiter=';')
        wr.writerow([sd] + list)