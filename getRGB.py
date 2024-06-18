from PIL import ImageColor
import os
import argparse

def get_first_hex():
    # find the .hex file in the same directory as this script
    filename = ""
    for file in os.listdir():
        if file.endswith(".hex"):
            filename = file
            break
    return filename

def convert2rgb(filename):
    pallette_name = os.path.basename(filename).rstrip('.hex').upper()
    # replace all non-alphanumeric characters with underscores
    pallette_name = ''.join([c if c.isalnum() else '_' for c in pallette_name])
    # if pallette_name starts with a number, prepend a P_
    if pallette_name[0].isdigit():
        pallette_name = 'P_' + pallette_name

    pallette_path = os.path.join('pallettes', pallette_name + '.py')

    with open(filename, 'r') as f:
        with open(pallette_path, 'w') as f2:
            f2.write('import numpy as np\n\n')
            f2.write(f'{pallette_name} = np.array([\n')
            for line in f:
                # convert hex to rgb
                rgb = ImageColor.getrgb('#' + line.strip())
                f2.write(f'\t[{rgb[2]}, {rgb[1]}, {rgb[0]}],    # {line.strip()}\n')
            f2.write('])\n')

    return pallette_name

def import_in_pallettes(module_name):
    with open(os.path.join('pallettes', 'pallettes.py'), 'a') as f:
        f.write(f'from .{module_name} import *\n')

def main():
    # get filename by argparse
    parser = argparse.ArgumentParser(description='Convert .hex file to .txt file with RGB values')
    parser.add_argument('mode', type=str, help='the mode to run the script in')
    parser.add_argument('filename', type=str, help='the .hex file to convert')
    args = parser.parse_args()
    mode = args.mode
    filename = args.filename

    if filename == "":
        filename = get_first_hex()
    # else:
    #     # recursive search for the file in all subdirectories
    #     for root, dirs, files in os.walk('.'):
    #         if filename in files:
    #             filename = os.path.join(root, filename)
    #             print(f'Found {filename}')
    #             break

    pallette_name = convert2rgb(filename)

    if mode == 'import':
        import_in_pallettes(pallette_name)
    

if __name__ == '__main__':
    main()

