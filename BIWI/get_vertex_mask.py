import trimesh
import numpy as np
import pickle 

face_mesh = trimesh.load('BIWI/templates/F1.obj', process=False)
vertices = face_mesh.vertices
mask_vertices = []
indices = []
for k in range(vertices.shape[0]):
    vertex = vertices[k]
    if vertex[1] >= -0.08 and vertex[1] <= -0.035 and vertex[0] >= -0.035 and vertex[0] <= 0.035:
        mask_vertices.append(vertex)
        indices.append(k)
        
mask = trimesh.Trimesh(vertices=mask_vertices)
# mask.export('BIWI/mouth_indices.ply')
# np.save('BIWI/mouth_indices.npy', indices)

# write this result to a dictionary and save it to template.pkl
with open('BIWI/BIWI_lip.pkl', 'wb') as file: 
    pickle.dump(indices, file, protocol=pickle.HIGHEST_PROTOCOL) 
