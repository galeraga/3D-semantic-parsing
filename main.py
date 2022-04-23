# http://www.open3d.org/docs/latest/introduction.html
# Pay attention to Open3D-Viewer App http://www.open3d.org/docs/latest/introduction.html#open3d-viewer-app
# and the Open3D-ML http://www.open3d.org/docs/latest/introduction.html#open3d-ml
# pip install open3d
import open3d as o3d

# Set the sample path HERE:
POINT_CLOUD_DATA_PATH = "/Users/jgalera/datasets/S3DIS"
TEST_PC = "/Area_1/office_1/office_1.txt"

def RGB_normalization(f):
    """
    Takes the input file and calculates the RGB normalization
    """

    # Keep the original dataset file intact and create 
    # a new file with normalized RGB values
    tgt_file = f.split('.')[0] + '_rgb_norm.txt'

    normalized = ''
    with open(f) as src:
        with open(tgt_file, "w") as tgt:
            for l in src:
                # Convert the str to list for easier manipulation
                x, y, z, r, g, b = l.split()
                r = float(r)/255
                g = float(g)/255
                b = float(b)/255

                # Back to str again
                normalized += ' '.join([str(x), str(y), str(z), 
                    '{:.8s}'.format(str(r)), 
                    '{:.8s}'.format(str(g)), 
                    '{:.8s}'.format(str(b)), 
                    '\n'])        
            
            tgt.write(normalized)
        print(tgt)
    
    return tgt_file


# To quickly test o3d
if __name__ == "__main__":
    """
    dataset = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)
    o3d.visualization.draw(pcd)
    """

    # Two minor issues when working with S3DIS dataset:
    # - Open3D does NOT support TXT file extension, so we have to specify 
    #   the xyzrgb format (check supported file extensions here: 
    #   http://www.open3d.org/docs/latest/tutorial/Basic/file_io.html) 
    # - When working with xyzrgb format, each line contains [x, y, z, r, g, b], 
    #   where r, g, b are in floats of range [0, 1]
    #   So we need to normalize the RGB values from the S3DIS dataset in order 
    #   to allow Open3D to display them

    pcd_RGB_normalized = RGB_normalization(POINT_CLOUD_DATA_PATH + TEST_PC)
    pcd = o3d.io.read_point_cloud(pcd_RGB_normalized, format='xyzrgb')
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd])
