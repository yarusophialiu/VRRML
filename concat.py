import os
import shutil

# Define the source and target directories
source_bedroom = '/home/yl962/rds/hpc-work/VRR/Data/VRR_Patches/2024-09-14/bedroom'
target_bedroom = '/home/yl962/rds/hpc-work/VRR/Data/VRR_Patches/cleaned_patches/bedroom'

# Iterate through all subfolders in the source directory
for subfolder_name in os.listdir(source_bedroom):
    source_subfolder = os.path.join(source_bedroom, subfolder_name)

    if os.path.isdir(source_subfolder):
        # Check if the corresponding subfolder exists in the target directory
        target_subfolder = os.path.join(target_bedroom, subfolder_name)
        print(f'target_subfolder {target_subfolder}')
        
        os.makedirs(target_subfolder, exist_ok=True)
        if os.path.exists(target_subfolder) and os.path.isdir(target_subfolder):
        # if os.path.isdir(target_subfolder):
            print(f'Found matching subfolder: {target_subfolder}')

            # Move all the contents from the source subfolder to the target subfolder
            for filename in os.listdir(source_subfolder):
                source_file = os.path.join(source_subfolder, filename)
                target_file = os.path.join(target_subfolder, filename)
                print(f'source_file {source_file}')
                print(f'target_file {target_subfolder}')
#                shutil.move(source_file, target_subfolder)
                # Move the file or directory
#                if os.path.isfile(source_file):
#                    shutil.move(source_file, target_file)
#                elif os.path.isdir(source_file):
#                    shutil.move(source_file, target_subfolder)

            print(f'Moved contents from {source_subfolder} to {target_subfolder}')
        else:
            print(f'No matching subfolder for {subfolder_name} in target directory.')