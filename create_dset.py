from skimage import io

def read_from_tif(path: str):
    # Read tif file
    data = io.imread(path) # N x H x W
    return data

def  write_to_path(data, path: str, size: int = None, mode: str = 'gray'):
    # Write to path
    import cv2
    import os

    if not os.path.exists(path):
        os.makedirs(path)
    
    for i, image in enumerate(data):
        image = cv2.resize(image, (size, size)) if size else image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if mode == 'color' else image
        img_path = path + str(i) + '.png'
        cv2.imwrite(img_path, image)
    
    return None

def create_dset(read_path: str, write_path: str, size: int = None, mode: str = 'gray'):
    # Read from path
    data = read_from_tif(read_path)
    write_to_path(data, write_path, size, mode)
    return None

if __name__ == '__main__':
    read_path = '/home/utku/Downloads/PTY_pristine_raw.tif'
    size = 512
    mode = 'gray'
    write_path  = f'./data_{size}/'
    create_dset(read_path, write_path, size)

