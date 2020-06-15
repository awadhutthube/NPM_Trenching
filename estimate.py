import numpy as np

def get_slice_points(cloud_array, n_slices = 5):
    # slices = np.split(cloud_array, n_slices, axis = 0)
    max_val = np.amax(cloud_array[:,0])
    min_val = np.amin(cloud_array[:,0])
    jump = (max_val - min_val)/n_slices
    slice_points = [min_val+i*jump for i in range(n_slices+1)]
    return slice_points

def get_section(cloud_array, low_bound, high_bound):
    section = cloud_array[cloud_array[:,0] >= low_bound]
    section = section[section[:,0] < high_bound]
    return section

def project_section(section):
    y_co = (section[:,1]*1000).round().astype('int')
    z_co = (section[:,2]*1000).round().astype('int')
    s_map = np.zeros(150,300).astype('float')
    s_map[z_co, y_co] = 1
    return s_map

