import os
import time
import math
import pybel
import shutil
import pickle
import numpy as np
from pymol import cmd

class config:

    def __init__(self):

        self.mut_labels = dict({"I3H":"1.51_0.29","Z4H":"2.22_0.38","V5H":"-0.29_0.16","E8H":"-0.16_0.34","E13H":"-1.12_0.34","Z4T":"-0.01_0.75","Z4Y":"-0.55_0.78",})
        self.work_path = os.path.join("/public/home/yqyang/VEGF/FEP/MD")

def get_residue_reference():
    return {'LEU', 'MET', 'ILE', 'GLU', 'CYS', 'GLY', 'PHE', 'ASN', 'GLN', 'LYS', 'TYR', 'ARG', 'THR', 'PRO', 'VAL', 'ASP', 'ALA', 'TRP', 'HIS', 'SER', 'NLE', 'HSD', 'HSE', 'HSP'}        

def get_atoms_reference():
    return {'N.3':7, 'O.spc':8, 'O.t3p':8, 'C.3':6, 'O.co2':8, 'N.pl3':7, 'S.2':16, 'O.3':8, 'O.2':8, 'C.cat':6, 'P.3':15, 'C.2':6, 'C.ar':6, 'N.1':7, 'N.ar':7, 'C.1':6, 'S.o2':16, 'N.4':7, 'S.3':16, 'S.o':16, 'N.am':7, 'N.2':7}    

def split_pdb(pdb):

    with open(pdb) as f:
        f1 = f.readlines()

    MUT = open(str(pdb).replace(".pdb", "_l.pdb"), "w")
    PRO = open(str(pdb).replace(".pdb", "_r.pdb"), "w")

    for i in f1:
        if i.startswith("ATOM"):
            if i[66:76].strip() == "MUT":
                MUT.write(i)
            elif i[66:76].strip() == "VG1" or i[66:76].strip() == "VG2":
                PRO.write(i)

    MUT.close()
    PRO.close()
            
    mol = next(pybel.readfile("pdb", str(pdb).replace(".pdb", "_r.pdb")))
    output = pybel.Outputfile("mol2", str(pdb).replace(".pdb", "_r.mol2"), overwrite=True)
    output.write(mol)
    output.close()
    mol = next(pybel.readfile("pdb", str(pdb).replace(".pdb", "_l.pdb")))
    output = pybel.Outputfile("mol2", str(pdb).replace(".pdb", "_l.mol2"), overwrite=True)
    output.write(mol)
    output.close()

def get_single_snapshot(work_path, mut, v_e):

    shutil.copy(os.path.join(work_path, mut, "common", "complex.psf"), "complex.psf")
    shutil.copy(os.path.join(work_path, mut, "prod", "com-prodstep.dcd"), "com-prodstep.dcd")

    cmd.load("complex.psf")
    cmd.load_traj("com-prodstep.dcd")
    
    for i in range(500, 1000):
        cmd.save("{0}_{1}.pdb".format(mut, str(i+1)), state = i+1)
        split_pdb("{0}_{1}.pdb".format(mut, str(i+1)))

    cmd.delete("all")

    mean = float(v_e.split("_")[0])
    std = float(v_e.split("_")[1])

    r_lst = ["{0}_{1}_r.mol2".format(mut, str(i+1)) for i in range(500, 1000)]
    l_lst = ["{0}_{1}_l.mol2".format(mut, str(i+1)) for i in range(500, 1000)]
    label = np.random.normal(mean, std, (500))

    return r_lst, l_lst, label

def read_complexes(protein_list, ligand_list, labels_list):
    
    data, labels = dict(), dict()
    
    for idx, (p_file, l_file) in enumerate(zip(protein_list, ligand_list)):
        
        '''
        protein atoms type : Store SYMBL of each atom in list.
        protein atoms type index dict : Store index for nine atom types.
        protein atoms coords : The coordinates of each atom.
        protein bonds : Store each atom's bond in an undirected graph((start, end),(start, end)).        
        '''
        ligand_atoms_type, ligand_atoms_type_index_dict, ligand_atoms_coords, ligand_bonds = read_protein_file((open(l_file, "r")).readlines())
        
        protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = read_protein_file((open(p_file, "r")).readlines())

        '''
        Find all cases where the distance between the ligand and the protein atom is less than a certain threshold.
        interaction list -> key : (protein atom type, ligand atom type), 
        value : [[ligand atom index, protein atom index], [ligand atom index, protein atom index], ...]
        '''
        interaction_list = get_interaction(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords)
        
        ''' find graph '''
        graphs = r_radius_graph(interaction_list, ligand_bonds, ligand_atoms_type, protein_bonds, protein_atoms_type)
        
        protein_name = p_file.split("/")[-1].split(".")[0]
        ligand_name = l_file.split("/")[-1].split(".")[0]
        
        name = protein_name + "/" + ligand_name
        
        data[name] = graphs
        labels[name] = labels_list[idx]
        
    return data, labels

def read_protein_file(lines):
    
    atoms_reference_dict = get_atoms_reference()
    residue_reference_set = get_residue_reference()
    residue_list, atoms_dict, protein_atoms_type, protein_atoms_type_index_dict, protein_atoms_coords, protein_bonds = list(), dict(), list(), dict(), list(), dict()
    flag, index, atom_count = 0, 0, 0    

    while index < len(lines):
        
        ''' Extract information about the atom '''
        line = lines[index].strip()    
    
        if flag == 1:
            if "@<TRIPOS>BOND" not in line:
                line_list = line.split()
                
                ''' Extract SYMBL type of atom '''
                atoms_number = int(line_list[0])
                atom_type = line_list[5]
                residue_name = line_list[7].strip()[:3]    
                
                ''' 
                It works if it is not H(hydrogen).
                Save the coordinate information and atom type, and redefine the index.
                '''    
                if atom_type != "H" and residue_name in residue_reference_set and atom_type in atoms_reference_dict:    

                    protein_atoms_coords.append([line_list[2], line_list[3], line_list[4]])   # list
                    protein_atoms_type.append(atoms_reference_dict[atom_type])   # list
                    
                    if atoms_reference_dict[atom_type] in protein_atoms_type_index_dict: 
                        protein_atoms_type_index_dict[atoms_reference_dict[atom_type]].append(atom_count)
                        
                    else:    
                        protein_atoms_type_index_dict[atoms_reference_dict[atom_type]] = [atom_count]                 # protein_atoms_type_index_dict: keys: atom_type_number   values: [new_positions]
                        
                    atoms_dict[atoms_number] = atom_count
                    atom_count += 1                
    
        elif flag == 2:
            ''' Extract informaction about bond. '''
            
            line_list = line.split()    

            start_atom = int(line_list[1]) 
            end_atom = int(line_list[2])
            bond_type = line_list[3]
            
            if start_atom in atoms_dict and end_atom in atoms_dict:
                protein_bonds[(atoms_dict[start_atom], atoms_dict[end_atom])] = bond_type
                protein_bonds[(atoms_dict[end_atom], atoms_dict[start_atom])] = bond_type    
    
        if "@<TRIPOS>ATOM" in line:
            flag = 1
        elif "@<TRIPOS>BOND" in line:
            flag = 2
        
        index += 1
        
    return protein_atoms_type, protein_atoms_type_index_dict, np.array(protein_atoms_coords, dtype = np.float32), protein_bonds        

'''
The protein and ligand atom pairs that exist with a certain distance are stored in
16 different types.
'''        
def get_interaction(ligand_atoms_type_index_dict, ligand_atoms_coords, protein_atoms_type_index_dict, protein_atoms_coords):
        
        interaction_list = make_interaction_dict()        
        
        for i in protein_atoms_type_index_dict.keys():
                
                protein_atom_list = protein_atoms_type_index_dict[i]
                protein_atom_list_coords = protein_atoms_coords[protein_atom_list]
                
                for j in ligand_atoms_type_index_dict.keys():
                        
                        ligand_atom_list = ligand_atoms_type_index_dict[j]
                        ligand_atom_list_coords = ligand_atoms_coords[ligand_atom_list]
                        
                        interaction_ligand_atoms_index, interaction_protein_atoms_index, interaction_distance_matrix = cal_distance(ligand_atom_list_coords, protein_atom_list_coords) 

                        for ligand_, protein_ in zip(interaction_ligand_atoms_index, interaction_protein_atoms_index):
                                interaction_list[(i,j)].append((ligand_atom_list[ligand_], protein_atom_list[protein_]))
                        
        return interaction_list

def distance_x_y_z(coord_a, coord_b):
    return math.sqrt((float(coord_b[0])-float(coord_a[0]))**2 + (float(coord_b[1])-float(coord_a[1]))**2 + (float(coord_b[2])-float(coord_a[2]))**2)

'''
Find protein and ligand atom pairs within a certain distance.
'''        
def cal_distance(ligand_coords, protein_coords):
        
        threshold = 12.

        lig_pro_matrix = []
        for i in ligand_coords:
            tmp = []
            for j in protein_coords:
                tmp.append(distance_x_y_z(i, j))
            lig_pro_matrix.append(tmp)

        distance = np.array(lig_pro_matrix, dtype=np.float64)
        rows, cols = np.where(distance <= threshold)
                
        return rows, cols, distance

'''
Initialize a dictionary to store 36 interactions (protein-ligand pairs).
'''        
def make_interaction_dict():
        
        interaction_list = dict()
        
        protein_atoms_list = [6, 7, 8, 16]
        ligand_atoms_list = [6, 7, 8, 16]
        
        for i in protein_atoms_list:
            for j in ligand_atoms_list:
                interaction_list[(i,j)] = list()
                        
        return interaction_list

def r_radius_graph(interaction_list, ligand_bonds, ligand_atoms_type, protein_bonds, protein_atoms_type):    

    '''
    undirect, atom = [atom1, atom2, ...] 
    bond dict => start : [end1, end2, ...]
    '''
    graphs = dict()
    ligand_bond_start_end, protein_bond_start_end = make_bond_dict(ligand_bonds, protein_bonds)
    
    for i in interaction_list.keys():
        
        sub_interaction_list = interaction_list[i]
        graph_list = list()
        
        ''' Extract one protein-ligand interaction(pair) '''
        for val in sub_interaction_list:
            
            sub = dict()
            
            ''' find ligand descriptor ''' 
            cur_ligand_atom_index, cur_ligand_atom = val[0], ligand_atoms_type[val[0]]
            
            visited_ligand, step = set(), 0
            visited_ligand.add(cur_ligand_atom_index)
            
            ''' Find the one-step neighborhoods of the current atom. '''
            next_ligand_atom_list, visited_ligand = get_next_atom(cur_ligand_atom_index, ligand_bond_start_end, visited_ligand)
            
            for next_ligand_atom_index in next_ligand_atom_list:
                bond = ligand_bonds[(cur_ligand_atom_index, next_ligand_atom_index)]
                next_ligand_atom = ligand_atoms_type[next_ligand_atom_index]
                
                if ((1, cur_ligand_atom, next_ligand_atom, bond)) in sub:
                    sub[(1, cur_ligand_atom, next_ligand_atom, bond)] += 1
                else:
                    sub[(1, cur_ligand_atom, next_ligand_atom, bond)] = 1
                    
            ''' find protein descriptor ''' 
            cur_protein_atom_index, cur_protein_atom = val[1], protein_atoms_type[val[1]]
            
            visited_protein, step = set(), 0
            visited_protein.add(cur_protein_atom_index)
            
            ''' Find the one-step neighborhoods of the current atom. '''
            next_protein_atom_list, visited_protein = get_next_atom(cur_protein_atom_index, protein_bond_start_end, visited_protein)
            
            for next_protein_atom_index in next_protein_atom_list:
                bond = protein_bonds[(cur_protein_atom_index, next_protein_atom_index)]
                next_protein_atom = protein_atoms_type[next_protein_atom_index]
                
                if ((0, cur_protein_atom, next_protein_atom, bond)) in sub: 
                    sub[(0, cur_protein_atom, next_protein_atom, bond)] += 1
                else:
                    sub[(0, cur_protein_atom, next_protein_atom, bond)] = 1                
            
            graph_list.append(sub)
        
        graphs[i] = graph_list        
    
    return graphs

'''
Find one-step neighborhoods of the current atom.
'''    
def get_next_atom(current_atom, bond_dict, visited_atoms):

    next_atoms_list = bond_dict[current_atom]
    next_atoms_list = list(set(next_atoms_list) -  visited_atoms)    
    
    if len(next_atoms_list) == 0:
        return [], visited_atoms
    else:
        for i in next_atoms_list:
            visited_atoms.add(i)
            
        return next_atoms_list, visited_atoms

'''
Stores bonds between atoms in a dictionary.
'''    
def make_bond_dict(ligand_bonds, protein_bonds):
    
    ligand_start_end, protein_start_end = dict(), dict()
    
    for i in ligand_bonds.keys():
        start = i[0]
        end = i[1]
        
        if start in ligand_start_end:
            ligand_start_end[start].append(end)
            
        else:    
            ligand_start_end[start] = [end]
            
    for i in protein_bonds.keys():    
        start = i[0]
        end = i[1]
    
        if start in protein_start_end:
            protein_start_end[start].append(end)
        else:
            protein_start_end[start] = [end]
            
    return ligand_start_end, protein_start_end

def remove_files(mut):

    os.remove("complex.psf")
    os.remove("com-prodstep.dcd")
    for i in range(500, 1000):
        os.remove("{0}_{1}.pdb".format(mut, str(i+1)))
        os.remove("{0}_{1}_r.pdb".format(mut, str(i+1)))
        os.remove("{0}_{1}_r.mol2".format(mut, str(i+1)))
        os.remove("{0}_{1}_l.pdb".format(mut, str(i+1)))
        os.remove("{0}_{1}_l.mol2".format(mut, str(i+1)))

def run():
    
    start = time.time()

    settings = config()
    for i in settings.mut_labels.keys():
        protein_list, ligand_list, labels_list = get_single_snapshot(settings.work_path, i, settings.mut_labels[i])
        graphs_dict, labels = read_complexes(protein_list, ligand_list, labels_list)
        with open("Descriptors_" + i + ".pkl", "wb") as f:
            pickle.dump((graphs_dict, labels), f) 
        remove_files(i)
        
    end = time.time()
    runtime_h = (end - start) / 3600
    print("run " + str(runtime_h) + " h.")

def main():
    run()
    
if __name__=="__main__":
    main() 