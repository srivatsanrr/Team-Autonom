import os
import json

json_folder_path = "/home/ubuntu/Team-Autonom/"

json_files = [ x for x in os.listdir(json_folder_path) if x.endswith("json") ]
json_data = list()
for json_file in json_files:
    json_file_path = os.path.join(json_folder_path, json_file)
    with open (json_file_path, "r") as f:
        json_data.append(json.load(f))

print(json_data[0]['children'])

for data in json_data[0]['children']:
    print(data['maxcol'])



'''print(json_data[0]['children'][0]['maxcol'])
print(json_data[0]['children'][0]['mincol'])
print(json_data[0]['children'][0]['minrow'])
print(json_data[0]['children'][0]['maxrow'])'''
