import bpy
import json
import os

# -----------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------
# Update this to match your file path on Windows:
JSON_PATH = r"C:\Users\davis\OneDrive\Desktop\datasets\presentation\kmeans_data.json"

iteration_gap = 5   # Frames to skip between each iteration (e.g., iteration 0 → frame0, iteration1 → frame5, etc.)
sphere_radius = 0.1 # Default radius for data point spheres
centroid_radius = 0.2

# Color map for up to 4 clusters: Red, Green, Blue, Yellow
CLUSTER_COLORS = {
    0: (1.0, 0.0, 0.0, 1.0),  # RGBA Red
    1: (0.0, 1.0, 0.0, 1.0),  # Green
    2: (0.0, 0.0, 1.0, 1.0),  # Blue
    3: (1.0, 1.0, 0.0, 1.0),  # Yellow
}

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def create_or_get_material_for_cluster(cluster_id):
    """
    Create a new material or return existing one for the given cluster_id,
    using CLUSTER_COLORS for the color.
    """
    mat_name = f"Cluster_{cluster_id}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]
    
    mat = bpy.data.materials.new(name=mat_name)
    color = CLUSTER_COLORS.get(cluster_id, (1, 1, 1, 1))  # white if unknown cluster
    mat.diffuse_color = color  # For Blender 2.8+/Eevee/Cycles
    return mat

def create_or_get_sphere(obj_name, radius):
    """
    If an object named 'obj_name' exists, return it.
    Otherwise, create a new UV sphere at origin with given radius.
    """
    if obj_name in bpy.data.objects:
        return bpy.data.objects[obj_name]
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=(0,0,0))
        sphere = bpy.context.object
        sphere.name = obj_name
        return sphere

def set_keyframe_interpolation(interpolation_mode='CONSTANT'):
    """
    (Optional) Utility to set the interpolation mode for all location f-curves
    in the scene. Common modes: 'LINEAR', 'CONSTANT', 'BEZIER' (default).
    """
    for action in bpy.data.actions:
        for fcurve in action.fcurves:
            if fcurve.data_path.endswith("location"):
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = interpolation_mode

# -----------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------
def main():
    # 1) Load the JSON data
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    # 2) For each iteration, set the scene frame, place points
    for iteration_data in data:
        iteration = iteration_data["iteration"]
        points = iteration_data.get("points", [])
        centroids = iteration_data.get("centroids", [])

        # e.g. iteration 0 -> frame 0, iteration1 -> frame 5, etc.
        frame_num = iteration * iteration_gap

        # Switch to that frame
        bpy.context.scene.frame_set(frame_num)

        # 3) Process data points
        for p in points:
            p_id = p["id"]
            x = p["x"]
            y = p["y"]
            z = p["z"]
            cluster_id = p.get("cluster", 0)

            # sphere name, e.g. "point_0"
            sphere_name = f"point_{p_id}"
            sphere = create_or_get_sphere(sphere_name, sphere_radius)

            # Move the sphere
            sphere.location = (x, y, z)
            sphere.keyframe_insert(data_path="location", frame=frame_num)

            # Apply a material based on the cluster
            mat = create_or_get_material_for_cluster(cluster_id)
            sphere.data.materials.clear()
            sphere.data.materials.append(mat)

        # 4) Process centroids
        for c in centroids:
            c_id = c["id"]
            cx = c["x"]
            cy = c["y"]
            cz = c["z"]
            centroid_name = f"centroid_{c_id}"
            cent = create_or_get_sphere(centroid_name, centroid_radius)
            
            cent.location = (cx, cy, cz)
            cent.keyframe_insert(data_path="location", frame=frame_num)

            # If you want a special color for centroids, do something like:
            # mat_cent = create_or_get_material_for_cluster(-1)
            # cent.data.materials.clear()
            # cent.data.materials.append(mat_cent)

    print("Data import and keyframing complete!")

    # 5) (Optional) Adjust interpolation to 'CONSTANT' if you want discrete jumps
    # set_keyframe_interpolation('CONSTANT')

# -----------------------------------------------------------
# RUN SCRIPT
# -----------------------------------------------------------
main()

