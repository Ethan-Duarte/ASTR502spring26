#run with: python run_known_cluster.py
#used Gemini to debug
#imports
import Comove
import pandas as pd
import os
import shutil

#functions

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

# ## Define the inputs:
# ##An target star name searchable on simbad. If coordinates are used, it will just be used as the default results directory
data = pd.read_csv(r'C:\Users\Snigdha\Documents\college\2025-2026\ASTR502\Comove\ASTR502_Mega_Target_List - ASTR502_Mega_Target_List.csv')

#variables for relevant columns
hostname = data['hostname']
gaia_id = data['gaia_dr3_id']
ra = data['ra']
dec = data['dec']
rv = data['st_rv']
err_rv =  data['st_e_rv']

i = 981
targname = gaia_id[i]
print(hostname[i])
#print('RV: ' + str(rv[i]))

# ##Alternative is to use coordinates, default is [None,None] in which case the targname is used to get coordinates
#rd = [None,None]
rd = [ra[i], dec[i]]#['217.3920159', '39.7903991']
# print(rd)

# ##input target star radial velocity to calulate 3D space velocities
radvel= rv[i] ##km/s
err_radvel = err_rv[i]
# print(radvel)
# print(err_radvel)

# ##Neighbour velocity difference limit, and on sky search radius
vlim=5.0 ##km/s
srad=25.0 ##parsecs (spherical radius around target)

# Construct the folder name that Comove will create
# Check if it exists and delete it
old_folder = f"./{hostname[i]}_friends"
old_folder_with_id = f"./{gaia_id[i]}_friends"
if os.path.exists(old_folder):
    print(f"Cleaning up existing folder: {old_folder}")
    shutil.rmtree(old_folder)
elif os.path.exists(old_folder_with_id):
    print(f"Cleaning up existing folder: {old_folder_with_id}")
    shutil.rmtree(old_folder_with_id)

# ##This line runs the entire code. Set showplots=True to interactively plot, otherwise they are only saved as pngs
# ##Set verbose=True to see LOTS of print output
output_location = Comove.findfriends(str(targname),float(radvel),velocity_limit=vlim,search_radius=srad,radec=rd,output_directory=None,verbose=False,showplots=False)

#rename directory and files using hostname
rename_directory(hostname[i], output_location, gaia_id[i])

print('Part 1 Done!')

#Cuts


#calculate error