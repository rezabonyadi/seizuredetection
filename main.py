from epilepsydataprovider import SeizureDataRead



kaggle_address = "R:/projects/Seizure/Seizure_detection"
folders = ["Dog_1"]
data = SeizureDataRead.read_kaggle_2014(kaggle_address, folders[0], 0.8)

i = 0