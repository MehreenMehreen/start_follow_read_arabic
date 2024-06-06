import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}


def warp_image(img, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 12)
    w_mesh_std = kwargs.get('w_mesh_std', 1.5)

    h_mesh_interval = kwargs.get('h_mesh_interval', 12)
    h_mesh_std = kwargs.get('h_mesh_std', 1.5)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        ############################################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2:
            color = 0
        else:
            color = np.array([0,0,255])
        for s in source:
            img[int(s[0]):int(s[0])+1,:] = color
            img[:,int(s[1]):int(s[1])+1] = color

    # Perturb source control points
    
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] = destination[:,0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:,1] = destination[:,1] + random_state.normal(0.0, w_mesh_std, size=source_shape)
    
    
    # Warp image
    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]
    warped = cv2.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(255,255,255))

    return warped

# https://stackoverflow.com/questions/60609607/how-to-create-this-barrel-radial-distortion-with-python-opencv
# Also see x_u and y_u at https://en.wikipedia.org/wiki/Distortion_(optics)
#img: input image
# barrel distortion
# adjust k_1 and k_2 to achieve the required distortion
def barrel_distortion(img, k_1=0.2, k_2=0.05):

    #img = input_img
    h,w = img.shape[:2]
    # distorted img boundary
    
    x,y = np.meshgrid(np.float32(np.arange(w)),np.float32(np.arange(h))) # meshgrid for interpolation mapping
    # center and scale the grid for radius calculation (distance from center of image)
    x_c = w/2 
    y_c = h/2 
    x = x - x_c
    y = y - y_c
    x = x/x_c
    y = y/y_c

    radius_sq = (x**2 + y**2) # distance sq from the center of image
    m_r = 1 + k_1*radius_sq + k_2*radius_sq**2 # radial distortion model

    # apply the model 
    x = x * m_r 
    y = y * m_r

    # reset all the shifting
    x = x*x_c + x_c
    y = y*y_c + y_c

    
    result = cv2.remap(img, x, y, cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=0)
 #   result = result[x_lims[0]:x_lims[1]+1, y_lims[0]:y_lims[1]+1]
    result = resize_height(result, h)
    return result

#(x,y) are coord of source
#(a,b) are coord of dst
#(x,y) maps to (x+ycos(theta)+ysin(theta))
#(a,b) maps to (a-ycos(theta),b/sin(theta))
def my_distort_rotate(src, theta):
    
    if theta == np.pi/2:
        return src
    if theta == 0:
        return np.zeros(src.shape)
    
    # If theta is zero then img reduces to a single line
    # Img same when theta = 90
    
    src_h, src_w = src.shape[:2]
    dst_h, dst_w = np.abs(int(src_h*np.sin(theta))), np.abs(int(src_w*np.cos(theta)))
    
    map_x,map_y = np.meshgrid(np.float32(np.arange(src_w)),np.float32(np.arange(src_h))) # meshgrid for interpolation mapping
    
    x_c = src_w/2 
    y_c = src_h/2 
    # w.r.t center
    map_x = map_x - x_c
    map_y = map_y - y_c
    
    
    # w.r.t. center of img box/rect
    map_y = map_y/np.sin(theta) 
    map_x = map_x - map_y*np.cos(theta)
    
    # w.r.t upper left
    map_y = map_y + y_c
    map_x = map_x + x_c
    
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=0)
    # crop above and below
    to_crop = (src_h - dst_h)//2
    
    dst = dst[to_crop:, :]
    dst = dst[0:dst.shape[0]-to_crop,:]
    dst = resize_height(dst, src_h)
    return dst


# original coord are (x,y)
# mapped coord are (a,b)
# (a,b) = (rcos(theta)-r+x, rsin(theta))
# theta varies from theta_max to -theta_max
# when origin is center of box/image y varies from src_h to -src_h
# so (x,y) = ( a - r*cos(theta) + r,theta*h/theta_max
# When origin is center of box
# max(a,b) = (src_w/2,r*sin(theta_max))
# min_a = -src_w/2+rcos(theta_max)-r  
def my_distort_arc_left(src, theta_max=np.pi/6, r=30):
    src_h, src_w = src.shape[:2]
    max_dst_y = int(r*np.sin(theta_max))
    max_dst_x = int(src_w/2-r*np.cos(theta_max)+r + src_w/2)
   
    map_x,map_y = np.meshgrid(np.float32(np.arange(max_dst_x)),np.float32(np.arange(2*max_dst_y))) # meshgrid for interpolation mapping
    
    x_c = src_w/2 
    y_c = src_h/2 
    # w.r.t center
    map_x = map_x - (src_w/2-r*np.cos(theta_max)+r)
    map_y = map_y - max_dst_y


    # w.r.t. center of img box/rect
    theta = np.arcsin(map_y/r)
    map_y = theta*src_h/2/theta_max
    map_x = map_x - r*np.cos(theta) + r
    
    # w.r.t. upper left
    map_y = map_y + y_c
    map_x = map_x + x_c    
    
    # map values
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=0)
    dst = resize_height(dst, src_h)
    return dst

# original coord are (x,y)
# mapped coord are (a,b)
# (a,b) = (rcos(theta)-r+x, rsin(theta))
# theta varies from theta_max to -theta_max
# when origin is center of box/image y varies from src_h to -src_h
# so (x,y) = ( a - r*cos(theta) + r,theta*h/theta_max
# When origin is center of box
# max(a,b) = (src_w/2,r*sin(theta_max))
# min_a = -src_w/2+rcos(theta_max)-r  
def my_distort_arc_right(src, theta_max=np.pi/2, r=30):
    src_h, src_w = src.shape[:2]
    max_dst_y = int(r*np.sin(theta_max))
    max_dst_x = int(src_w/2-r*np.cos(theta_max)+r + src_w/2)
    
    map_x,map_y = np.meshgrid(np.float32(np.arange(max_dst_x)),np.float32(np.arange(2*max_dst_y))) # meshgrid for interpolation mapping
    
    x_c = src_w/2 
    y_c = src_h/2 
    # w.r.t center
    map_x = map_x - (src_w/2)
    map_y = map_y - max_dst_y
    


    # w.r.t. center of img box/rect
    theta = np.arcsin(map_y/r)
    map_y = theta*src_h/2/theta_max
    map_x = map_x + r*np.cos(theta) - r
    
    # w.r.t. upper left
    map_y = map_y + y_c
    map_x = map_x + x_c    
    
    # map values
    dst = cv2.remap(src, map_x, map_y, cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue=0)
    dst = resize_height(dst, src_h)
    return dst
    

# method_index is for testing
def apply_random_transform(img, seed=0, method_index=None):
    method_name = ['distort_w', 'distort_h', 'barrel', 'arc_left', 'arc_right', 'rotate_right', 'rotate_left', 'Nothing']
    
    h, w = img.shape[:2]
    
    if seed!=0:
        np.random.seed(seed)
    if method_index == None:
        selected_method = np.random.randint(len(method_name))
    else:
        selected_method = method_index
    
    #print('method:', method_name[selected_method])
    
    if method_name[selected_method] == 'distort_w':
        return warp_image(img, w_mesh_std = np.random.uniform(2,2.6))
    if method_name[selected_method] == 'distort_h':
        return warp_image(img, h_mesh_std = np.random.uniform(2,2.6))
    if method_name[selected_method] == 'barrel':
        return barrel_distortion(img, k_1=np.random.uniform(.01, .05), k_2=np.random.uniform(.01,.05))
    if method_name[selected_method] == 'arc_left':
        if w < 70:
            return img
        return my_distort_arc_left(img, theta_max=np.random.uniform(np.pi/3, np.pi/4), r=np.random.uniform(20,30))
    if method_name[selected_method] == 'arc_right':
        if w < 70:
            return img
        return my_distort_arc_right(img, theta_max=np.random.uniform(np.pi/3, np.pi/4), r=np.random.uniform(20,30))
    if method_name[selected_method] == "rotate_right":
        return my_distort_rotate(img, np.random.uniform(np.pi/2+.3, 5*np.pi/6))
    if method_name[selected_method] == "rotate_left":
        return my_distort_rotate(img, np.random.uniform(np.pi/6, np.pi/2-.3))
    if method_name[selected_method] == "Nothing":
        #print("Nothing distortion")
        return img
    print("No distortion, something wrong")
    return img
        
                                    
def resize_height(img, new_ht):
    h, w = img.shape[:2]
    new_width = int(w/h*new_ht)
    return cv2.resize(img, (new_width, new_ht))