import subprocess

def osm_to_sumo(osm_file_path, sumo_net_file_path):
    # Convert .osm to SUMO .net.xml using netconvert
    netconvert_command = f"netconvert --osm-files {osm_file_path} -o {sumo_net_file_path}"
    try:
        subprocess.run(netconvert_command, shell=True, check=True)
        print(f"Conversion complete. The SUMO network file is saved at {sumo_net_file_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")




# Example usage
osm_file_path = 'maps/map1.osm'
sumo_net_file_path = 'maps/map1.net.xml'
osm_to_sumo(osm_file_path, sumo_net_file_path)
