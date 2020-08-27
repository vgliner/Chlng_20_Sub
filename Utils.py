import configparser
import os
import csv
import numpy as np
import scipy.signal
# import matplotlib.pyplot as plt

def split(source, dest_folder, write_size):
    # Make a destination folder if it doesn't exist yet
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
 
    partnum = 0
 
    # Open the source file in binary mode
    input_file = open(source, 'rb')
 
    while True:
        # Read a portion of the input file
        chunk = input_file.read(write_size)
 
        # End the loop if we have hit EOF
        if not chunk:
            break
 
        # Increment partnum
        partnum += 1
 
        # Create a new file name
        filename = os.path.join(dest_folder, 'part'+str(partnum)+'.prt')
 
        # Create a destination file
        dest_file = open(filename, 'wb')
 
        # Write to this portion of the destination file
        dest_file.write(chunk)
 
        # Explicitly close 
        dest_file.close()
     
    # Explicitly close
    input_file.close()
     
    # Return the number of files created by the split
    return partnum
 
 
def join(source_dir, dest_file, read_size):
    # Create a new destination file
    output_file = open(dest_file, 'wb')
     
    # Get a list of the file parts
    parts = [] 
    for file in os.listdir(source_dir):
        if file.endswith(".prt"):
            parts.append(file)
     
    # Sort them by name (remember that the order num is part of the file name)
    parts.sort()
 
    # Go through each portion one by one
    for file in parts:
         
        # Assemble the full path to the file
        path = os.path.join(source_dir, file)
         
        # Open the part
        input_file = open(path, 'rb')
         
        while True:
            # Read all bytes of the part
            bytes = input_file.read(read_size)
             
            # Break out of loop if we are at end of file
            if not bytes:
                break
                 
            # Write the bytes to the output file
            output_file.write(bytes)
             
        # Close the input file
        input_file.close()
         
    # Close the output file
    output_file.close()

def read_config_file():
    config = configparser.ConfigParser()
    config_file = os.path.join(os.getcwd(), 'config.ini')
    config.read(config_file)
    paths=[]
    paths.append(config['Database Path']['Path'])
    try:
        paths.append(config['Brazilian Database Path']['Data_Path'])
        paths.append(config['Brazilian Database Path']['Annotations_Path'])
        paths.append(config['Brazilian Database Path']['Annotations_Dict_Path'])
    except:
        print('No Brazilian database path')
    print(f'Database path: {paths[0]}')
    return paths


# %%% Main 
if __name__ == "__main__":
    print('Test utilities file')
    # read_config_file()
    split(r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Submission_new_format\classifier.pt',r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Submission_new_format',int(100e6))
    join(r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Submission_new_format', r'C:\Users\vgliner\OneDrive - JNJ\Desktop\Submission_new_format\classifier.pt', int(100e6))
    print('finished testing utilities file')
