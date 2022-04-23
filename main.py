# pip install open3d
import open3d as o3d

if __name__ == "__main__":
    dataset = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)
    o3d.visualization.draw(pcd)