'''This module writes the data to a file'''

import time

def write_to_file(name, data:dict):
    '''This function writes the all_faces_recognized to a file'''
    #get current time
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
    high_probability = ''
    low_probability = ''
    #sort data based on dict values
    data = dict(sorted(data.items(), key=lambda item: item[1], reverse=True))   
    with open(name,'w') as f:
        str = f'-----\nAttendance At: {current_time}\n\n-----\n'
        for key, value in data.items():
            if value >= 3:
                high_probability += f'{key} - {value}\n'
            else:
                low_probability += f'{key} - {value}\n'
        str += f'High Probability:\n{high_probability}\n\nLow Probability (Can be Ignored):\n{low_probability}\n------'
        f.write(f'{str}\n')