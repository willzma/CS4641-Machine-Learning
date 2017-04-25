# This script was used to convert longitude values from continuous range (-180, 180) to discrete range (-9, 9)

import csv

# Input Latitude, Aircraft.Damage [4, 9]
# Output Longitude [5]
# 79293

with open("Aviation_Accident_Database_&_Synopses.csv", "r", encoding="utf8") as file:
    reader = csv.reader(file)
    csv_list = list(reader)
    header = csv_list[0]
    csv_list = csv_list[1:] #Ignore first row of data

for record in csv_list:
    record[2] = str(int(float(record[2]) / 20));

with open("Aviation_Accident_Database_&_Synopses1.csv", "w", encoding="utf8") as file:
    file.write(str(header).replace("'", "").replace("[", "").replace("]", "") + "\n")
    for record in csv_list:
        file.write(str(record).replace("'", "").replace("[", "").replace("]", "") + "\n")
