import numpy as np 
import pandas as pd 
import argparse 
from typing import List, Dict, Tuple
from numpy.typing import NDArray
import plotly.express as px

parser = argparse.ArgumentParser("Asdf")
parser.add_argument("-f", "--file", required=True, help="Path to the MOLCAS .log file")
parser.add_argument("--RASSI", action="store_true", default=False, help="Plot RASSI energies.")
parser.add_argument("--SO-RASSI", action="store_true", default=False, help="Plot SO-Rassi energies.")
    
args = parser.parse_args()

with open(args.file, 'r') as f:
    cont = f.readlines()

def extract_distances(filecont: List[str]) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:

    distances = []
    distances = [float(i.strip().split()[-1]) for i in cont if 'EVAL DIST' in i and '$DIST' not in i]
    distances = [float(f"{i:.5f}") for i in distances]
    distance_indices = [j for j,i in enumerate(filecont) if 'EVAL DIST' in i and '$DIST' not in i]
    
    return np.array(distances, dtype=np.float32), np.array(distance_indices, dtype=np.int32)

def assess_n_roots(filecont: List[str]) -> Dict:

    energy_tipe_calc_dict: Dict[str, int] = {}

    currtype = ''

    for i in cont:
        if '::' in i and "RASSI" in i:
            calctype = i.strip().split()[1]
            if calctype != currtype and calctype not in energy_tipe_calc_dict.keys():
                energy_tipe_calc_dict.update({calctype : 1})
                currtype = calctype
            elif calctype == currtype and calctype in energy_tipe_calc_dict.keys():
                 energy_tipe_calc_dict[calctype] = energy_tipe_calc_dict[calctype] + 1 
            elif calctype != currtype and calctype in energy_tipe_calc_dict.keys():
                break 

    return(energy_tipe_calc_dict)

def parse_character(strslice):
   character_starts = []
   character_ends = []
   for index, line in enumerate(strslice):
       if 'printout of CI-coefficients ' in line:
           character_starts.append(index) 

   for charstart in character_starts:
       for index, line in enumerate(strslice[charstart:]):
           if line.strip().split() == []:
               character_ends.append(index-1)
               break

   character_strings = []
   for cs, ce in zip(character_starts, character_ends):
       local_chr_str = ''
       for line in strslice[cs+3:cs+ce]:
           splstr = line.strip().split()
           chr_whgt_str = f'{" ".join(splstr[1:-3])}: {splstr[-1]}\n'
           local_chr_str += chr_whgt_str
#       print(local_chr_str)
       character_strings.append(local_chr_str)

   return character_strings
 
def parse_parents(str_slice): 
    for index, line in enumerate(str_slice):
        if ' SO State  Total energy (au)           Spin-free states, spin, and weights' in line:       
            start_index = index
            break 

    for index, line in enumerate(str_slice[start_index:]):
        if line.strip().split() == []:
            end_index = index - 1
            break

    # print(f'the start index ix {start_index}')
    # print(f'the end index ix {end_index}')

    parent_states = []

    for line in str_slice[start_index + 2 : start_index + end_index]:
        local_linestrip = line.strip().split()
        relevant_info = local_linestrip[2:]
        # print(relevant_info)
        rassi_states = [i for j,i in enumerate(relevant_info) if j % 3 == 0]
        js = [i for j,i in enumerate(relevant_info) if j % 3 == 1]
        weights = [i for j,i in enumerate(relevant_info) if j % 3 == 2]
        #  print(rassi_states)
        #  print(js)
        #  print(weights)
        relevant_parent_states = ''
        for r, j, w in zip(rassi_states, js, weights): 
            if float(w) > 0.001:
                relevant_parent_states += f'{r}  {w}  {j}\n'
        parent_states.append(relevant_parent_states)
    # print(parent_states)
    return(parent_states)
        

def join_rassi_with_so(string_value_list_rassi, string_value_list_sorassi):
    joined_rassis = []
    for rassistr in string_value_list_sorassi:
        finalstr = ''
        for minirassi_state in rassistr.split('\n')[:-1]:
            print(minirassi_state)
            finalstr += minirassi_state
            finalstr += '    Where the Rassi state is defined by:\n        '
            print(minirassi_state.strip().split())
            finalstr += string_value_list_rassi[int(minirassi_state.strip().split()[0])-1].replace('\n', '\n        ')[:-8]
        joined_rassis.append(finalstr)
    return(joined_rassis)

dists, dist_indices = extract_distances(cont)
n_roots_dict = assess_n_roots(cont)

# print(cont[dist_indices[1] : dist_indices[2]])

a = parse_character(cont[dist_indices[1] : dist_indices[2]])
b = parse_parents(cont[dist_indices[1] : dist_indices[2]])

print(a)
print(b)

join_rassi_with_so(a, b)
print('\n\n\n', join_rassi_with_so(a, b))

rows_list = []

for index, distance in enumerate(dists[:-1]):
    start_slice = dist_indices[index]
    end_slice = dist_indices[index+1]
     
    for key, expected_roots in n_roots_dict.items():
        found_roots = 0

        string_value_list_rassi = parse_character(cont[start_slice : end_slice])
        string_value_list_sorassi = parse_parents(cont[start_slice : end_slice])
        
        sorassi_joint = join_rassi_with_so(string_value_list_rassi, string_value_list_sorassi)
        string_value = '' 

        for line in cont[start_slice : start_slice + end_slice]:
            parts = line.strip().split() 
            
            if '::' in line and key == parts[1]:
                found_roots += 1  
                energy = float(parts[-1])
 
                if key == 'RASSI':
                    string_value = string_value_list_rassi[found_roots-1]

                if key == 'SO-RASSI':
                    string_value = sorassi_joint[found_roots-1]

                rows_list.append({
                    'dist': distance,
                    'type': key,
                    'root': found_roots,
                    'value': energy,
                    'string': string_value
                })

                if found_roots == expected_roots:
                    break

column_names = ['dist', 'type', 'root', 'value', 'string']
df = pd.DataFrame(rows_list, columns=column_names)

target_mask = df['type'].isin(['RASSI', 'SO-RASSI'])

group_mins = df.loc[target_mask].groupby('type')['value'].transform('min')

df.loc[target_mask, 'value'] = df.loc[target_mask, 'value'] - group_mins

df['value'] = df['value'] * 27.2114 

# print(df)                

print(df[(df['root'] == 1) & (df['type'] == 'RASSI')].head())
print(df[(df['root'] == 1) & (df['type'] == 'SO-RASSI')].head())

if args.RASSI and 'RASSI' in n_roots_dict.keys():
    
    df_rassi = df[df['type'] == 'RASSI']
    df_rassi['string'] = df_rassi['string'].str.replace('\n', '<br>', regex=False)
    fig = px.line(
        df_rassi, 
        x="dist", 
        y="value", 
        color="root",
        hover_data=["string"],
        title="RASSI with character curves"
    )
    
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    fig.show()



if args.SO_RASSI and 'SO-RASSI' in n_roots_dict.keys():
    df_sorassi = df[df['type'] == 'SO-RASSI']
    df_sorassi['string'] = df_sorassi['string'].str.replace('\n', '<br>', regex=False)
    fig = px.line(
        df_sorassi, 
        x="dist", 
        y="value", 
        color="root",
        hover_data=["string"],
        title="SO-RASSI with character curves"
    )
    
    fig.update_traces(
        hovertemplate="%{customdata[0]}<extra></extra>"
    )
    
    fig.show()

