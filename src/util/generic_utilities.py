'''This module writes the data to a file'''

def write_to_file(data):
    '''This function writes the all_faces_recognized to a file'''
    with open('all_faces_recognized.txt','w') as f:
        f.write(f'{data}\n')