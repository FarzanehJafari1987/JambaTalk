import os 
import argparse
import pywavefront
import torch
import json
import numpy as np
import pickle 
from plyfile import PlyData, PlyElement
import matplotlib as plt


# generate your own template.pkl in /BIWI folder:
def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset", type=str, default="BIWI")
    parser.add_argument("--template_file", type=str, default="templates")
    parser.add_argument("--vertices", type=str, default="vertices_npy")
    args = parser.parse_args()

    # load vertices data from each of the character's obj file and save on the .pkl file
    template_file = os.path.join(args.dataset, args.template_file)
    objects = []
    for (root, dirs, file) in os.walk(template_file):
        for f in file:
            if '.obj' in f:
                objects.append(f[:2])

    print(objects)

    vertices_dict = {}
    for obj in objects:
        obj_path = os.path.join(template_file, obj+'.obj')
        vertices = torch.tensor(pywavefront.Wavefront(obj_path).vertices)
        print(vertices)

        vertices_dict[obj] = np.array(vertices)
        print(vertices_dict)

    
    # write this result to a dictionary and save it to template.pkl
    with open('BIWI/templates.pkl', 'wb') as file: 
        pickle.dump(vertices_dict, file, protocol=pickle.HIGHEST_PROTOCOL) 


if __name__ == "__main__": 
    main()