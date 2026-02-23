import os

#function to rename file
def rename_file(star_name, full_directory_location, filename, gaia_id):
    og_path = os.path.join(full_directory_location,filename)
    if str(gaia_id) in og_path:
        new_path = os.path.join(full_directory_location, filename.replace(str(gaia_id), star_name))
        os.replace(og_path, new_path)

#function to rename directory and files inside it
def rename_directory(star_name, directory_location, gaia_id):
    # if directory_location is None:
    #     print(f'No directory location. Skipping rename')
    #     return
    #rename files inside it
    full_path = os.path.abspath(directory_location)
    for file in os.listdir(full_path):
        filename = os.fsdecode(file)
        rename_file(star_name, full_path, filename, str(gaia_id))
    if str(gaia_id) in full_path:
        new_path = full_path.replace(str(gaia_id), star_name) #for Windows
        #rename directory
        new_directory_location = os.rename(full_path, new_path)