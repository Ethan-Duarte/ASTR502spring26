import Comove
import pandas as pd
import os
import shutil
from rename import *

data = pd.read_csv(r'C:\Users\Snigdha\Documents\college\2025-2026\ASTR502\Comove\ASTR502_Mega_Target_List - ASTR502_Mega_Target_List.csv')

#variables for relevant columns
hostname = data['hostname']
gaia_id = data['gaia_dr3_id']
ra = data['ra']
dec = data['dec']
rv = data['st_rv']
err_rv =  data['st_e_rv']

problem = []

for i in range(len(data[0])): 
    try:
        targname = gaia_id[i]
        print(hostname[i])

        # ##Alternative is to use coordinates, default is [None,None] in which case the targname is used to get coordinates
        #rd = [None,None]
        rd = [ra[i], dec[i]]#['217.3920159', '39.7903991']

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
            #Debugging:
            #print(f"Cleaning up existing folder: {old_folder}")
            #shutil.rmtree(old_folder)
            print( str(old_folder) +'Already Exists!')
        elif os.path.exists(old_folder_with_id):
            #Debugging:
            #print(f"Cleaning up existing folder: {old_folder_with_id}")
            #shutil.rmtree(old_folder_with_id)
            print( str(old_folder_with_id) +'Already Exists!')


        # ##This line runs the entire code. Set showplots=True to interactively plot, otherwise they are only saved as pngs
        # ##Set verbose=True to see LOTS of print output
        output_location = Comove.findfriends(str(targname),float(radvel),velocity_limit=vlim,search_radius=srad,radec=rd,output_directory=None,verbose=False,showplots=False)

        #rename directory and files using hostname
        rename_directory(hostname[i], output_location, gaia_id[i])
    except:
        print(f'problematic: {hostname[i]}')
        problem.append(i)