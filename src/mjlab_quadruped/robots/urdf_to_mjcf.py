import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

def rpy_to_quat(roll: float, pitch: float, yaw: float) -> str:
    # Convert roll, pitch, yaw (in radians) to quaternion [w, x, y, z]
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return f"{w} {x} {y} {z}"

def parse_rpy(rpy_str: str) -> Tuple[float, float, float]:
    # Parse RPY string and return tuple
    values = [float(x) for x in rpy_str.split()]
    return values[0], values[1], values[2]

class URDFToMuJoCoConverter:
    def __init__(self, urdf_file: str):
        self.urdf_file = urdf_file
        self.tree = ET.parse(urdf_file)
        self.root = self.tree.getroot()
        self.links = {}
        self.joints = {}
        self.parent_map = {}
        self.parse_urdf()

    def parse_urdf(self):
        # Build maps for quick lookup
        for link in self.root.findall('link'):
            link_name = link.get('name')
            self.links[link_name] = link

        for joint in self.root.findall('joint'):
            joint_name = joint.get('name')
            self.joints[joint_name] = joint
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            self.parent_map[child] = (parent, joint)

    def find_root_link(self) -> str:
        # Root link is one that has no parent
        for link_name in self.links.keys():
            if link_name not in self.parent_map:
                return link_name
        return None

    def get_child_joints(self, parent_link: str) -> List[str]:
        # Find all joints where this link is the parent
        children = []
        for child_link, (parent, joint) in self.parent_map.items():
            if parent == parent_link:
                children.append(joint.get('name'))
        return children

    def extract_origin(self, element) -> Tuple[str, str]:
        # Extract position and rotation from origin element
        origin = element.find('origin')
        if origin is None:
            return "0 0 0", "0 0 0"
        
        pos = origin.get('xyz', '0 0 0')
        rot = origin.get('rpy', '0 0 0')
        return pos, rot

    def extract_inertial(self, link_element) -> Optional[ET.Element]:
        # Extract and convert inertial element
        inertial = link_element.find('inertial')
        if inertial is None:
            return None

        inertial_elem = ET.Element('inertial')
        
        mass = inertial.find('mass')
        if mass is not None:
            inertial_elem.set('mass', mass.get('value'))
        
        origin = inertial.find('origin')
        if origin is not None:
            pos_str = origin.get('xyz', '0 0 0')
            inertial_elem.set('pos', pos_str)
            
            # Convert RPY to quaternion for inertial
            rpy_str = origin.get('rpy', '0 0 0')
            roll, pitch, yaw = parse_rpy(rpy_str)
            if roll != 0 or pitch != 0 or yaw != 0:
                quat = rpy_to_quat(roll, pitch, yaw)
                inertial_elem.set('quat', quat)
        
        inertia = inertial.find('inertia')
        if inertia is not None:
            ixx = inertia.get('ixx', '0')
            iyy = inertia.get('iyy', '0')
            izz = inertia.get('izz', '0')
            inertial_elem.set('diaginertia', f"{ixx} {iyy} {izz}")
        
        return inertial_elem

    def extract_geometry(self, link_element) -> List[ET.Element]:
        # Extract visual and collision geometry
        geoms = []
        
        # Process visual elements
        for visual in link_element.findall('visual'):
            geom = self._create_geom_from_geometry(visual, 'visual')
            if geom is not None:
                geoms.append(geom)
        
        # Process collision elements
        for collision in link_element.findall('collision'):
            geom = self._create_geom_from_geometry(collision, 'collision')
            if geom is not None:
                if not geoms:
                    geoms.append(geom)
        
        return geoms

    def _create_geom_from_geometry(self, parent_elem, geom_type: str) -> Optional[ET.Element]:
        # Create MuJoCo geom element from URDF visual/collision
        geometry = parent_elem.find('geometry')
        if geometry is None:
            return None

        pos, rot = self.extract_origin(parent_elem)
        geom = ET.Element('geom')
        
        # Determine geometry type
        mesh = geometry.find('mesh')
        box = geometry.find('box')
        sphere = geometry.find('sphere')
        cylinder = geometry.find('cylinder')

        if mesh is not None:
            geom.set('type', 'mesh')
            mesh_file = mesh.get('filename')
            if mesh_file:
                mesh_name = Path(mesh_file).stem
                geom.set('mesh', mesh_name)
        elif box is not None:
            geom.set('type', 'box')
            size = box.get('size', '1 1 1')
            sizes = size.split()
            halfsize = ' '.join([str(float(s)/2) for s in sizes])
            geom.set('size', halfsize)
        elif sphere is not None:
            geom.set('type', 'sphere')
            radius = sphere.get('radius', '1')
            geom.set('size', radius)
        elif cylinder is not None:
            geom.set('type', 'cylinder')
            radius = cylinder.get('radius', '1')
            length = cylinder.get('length', '1')
            geom.set('size', f"{radius} {length/2}")

        if pos != "0 0 0":
            geom.set('pos', pos)
        if rot != "0 0 0":
            roll, pitch, yaw = parse_rpy(rot)
            quat = rpy_to_quat(roll, pitch, yaw)
            geom.set('quat', quat)

        if geom_type == 'collision':
            geom.set('contype', '0')
            geom.set('conaffinity', '0')

        return geom

    def convert_joint(self, joint_name: str) -> Tuple[ET.Element, str, str]:
        # Convert URDF joint to MuJoCo joint
        # Returns: (joint_element, joint_pos, joint_quat)
        joint_elem = self.joints[joint_name]
        
        mj_joint = ET.Element('joint')
        mj_joint.set('name', joint_name)
        
        joint_type = joint_elem.get('type')
        if joint_type == 'continuous':
            mj_joint.set('type', 'hinge')
        elif joint_type == 'revolute':
            mj_joint.set('type', 'hinge')
        elif joint_type == 'prismatic':
            mj_joint.set('type', 'slide')
        elif joint_type == 'fixed':
            mj_joint.set('type', 'fixed')
        
        # Get origin
        pos, rot = self.extract_origin(joint_elem)
        
        # Get axis (in parent frame)
        axis_elem = joint_elem.find('axis')
        if axis_elem is not None:
            axis_str = axis_elem.get('xyz', '0 0 1')
            mj_joint.set('axis', axis_str)
        
        # Handle limits
        limit = joint_elem.find('limit')
        if limit is not None and joint_type != 'continuous':
            lower = limit.get('lower', '-1')
            upper = limit.get('upper', '1')
            mj_joint.set('range', f"{lower} {upper}")
        
        return mj_joint, pos, rot

    def build_body_tree(self, link_name: str) -> ET.Element:
        # Recursively build MuJoCo body tree
        link_elem = self.links[link_name]
        
        body = ET.Element('body')
        body.set('name', link_name)
        
        # Add inertial properties
        inertial = self.extract_inertial(link_elem)
        if inertial is not None:
            body.append(inertial)
        
        # Add geometry
        geoms = self.extract_geometry(link_elem)
        for geom in geoms:
            body.append(geom)
        
        # Add child joints and bodies
        for joint_name in self.get_child_joints(link_name):
            joint_elem = self.joints[joint_name]
            child_link = joint_elem.find('child').get('link')
            
            mj_joint, joint_pos, joint_rot = self.convert_joint(joint_name)
            
            # Create intermediate body for joint frame
            joint_frame = ET.Element('body')
            joint_frame.set('name', f"{joint_name}_frame")
            
            if joint_pos != "0 0 0":
                joint_frame.set('pos', joint_pos)
            if joint_rot != "0 0 0":
                roll, pitch, yaw = parse_rpy(joint_rot)
                quat = rpy_to_quat(roll, pitch, yaw)
                joint_frame.set('quat', quat)
            
            # Add joint at frame origin
            mj_joint.set('pos', '0 0 0')
            joint_frame.append(mj_joint)
            
            # Create child body (unrotated, at joint frame origin)
            child_body = ET.Element('body')
            child_body.set('name', child_link)
            child_body.set('pos', '0 0 0')
            
            # Add child content
            child_link_elem = self.links[child_link]
            child_inertial = self.extract_inertial(child_link_elem)
            if child_inertial is not None:
                child_body.append(child_inertial)
            
            child_geoms = self.extract_geometry(child_link_elem)
            for geom in child_geoms:
                child_body.append(geom)
            
            # Recursively add grandchildren
            for grandchild_joint_name in self.get_child_joints(child_link):
                grandchild = self._build_child_recursively(grandchild_joint_name)
                child_body.append(grandchild)
            
            joint_frame.append(child_body)
            body.append(joint_frame)
        
        return body

    def _build_child_recursively(self, joint_name: str) -> ET.Element:
        # Helper to build child bodies recursively
        joint_elem = self.joints[joint_name]
        child_link = joint_elem.find('child').get('link')
        
        mj_joint, joint_pos, joint_rot = self.convert_joint(joint_name)
        
        # Create joint frame
        joint_frame = ET.Element('body')
        joint_frame.set('name', f"{joint_name}_frame")
        
        if joint_pos != "0 0 0":
            joint_frame.set('pos', joint_pos)
        if joint_rot != "0 0 0":
            roll, pitch, yaw = parse_rpy(joint_rot)
            quat = rpy_to_quat(roll, pitch, yaw)
            joint_frame.set('quat', quat)
        
        # Add joint
        mj_joint.set('pos', '0 0 0')
        joint_frame.append(mj_joint)
        
        # Create child body
        child_body = ET.Element('body')
        child_body.set('name', child_link)
        child_body.set('pos', '0 0 0')
        
        # Add child content
        child_link_elem = self.links[child_link]
        child_inertial = self.extract_inertial(child_link_elem)
        if child_inertial is not None:
            child_body.append(child_inertial)
        
        child_geoms = self.extract_geometry(child_link_elem)
        for geom in child_geoms:
            child_body.append(geom)
        
        # Recursively add further children
        for grandchild_joint_name in self.get_child_joints(child_link):
            grandchild = self._build_child_recursively(grandchild_joint_name)
            child_body.append(grandchild)
        
        joint_frame.append(child_body)
        return joint_frame

    def convert(self) -> ET.Element:
        # Main conversion
        mujoco = ET.Element('mujoco')
        mujoco.set('model', self.root.get('name', 'model'))
        
        compiler = ET.SubElement(mujoco, 'compiler')
        compiler.set('angle', 'radian')
        compiler.set('inertiafromgeom', 'true')
        
        option = ET.SubElement(mujoco, 'option')
        option.set('timestep', '0.002')
        option.set('gravity', '0 0 -9.81')
        
        asset = ET.SubElement(mujoco, 'asset')
        
        # Add mesh references
        mesh_set = set()
        for link in self.root.findall('link'):
            for geom_type in ['visual', 'collision']:
                for elem in link.findall(geom_type):
                    geometry = elem.find('geometry')
                    if geometry is not None:
                        mesh = geometry.find('mesh')
                        if mesh is not None:
                            mesh_file = mesh.get('filename')
                            if mesh_file and mesh_file not in mesh_set:
                                mesh_elem = ET.SubElement(asset, 'mesh')
                                mesh_name = Path(mesh_file).stem
                                mesh_elem.set('name', mesh_name)
                                mesh_elem.set('file', mesh_file)
                                mesh_set.add(mesh_file)
        
        worldbody = ET.SubElement(mujoco, 'worldbody')
        
        root_link = self.find_root_link()
        if root_link:
            root_body = self.build_body_tree(root_link)
            worldbody.append(root_body)
        
        return mujoco

    def prettify(self, elem) -> str:
        rough_string = ET.tostring(elem, encoding='unicode')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    def save_to_file(self, output_file: str):
        mujoco_elem = self.convert()
        xml_str = self.prettify(mujoco_elem)
        
        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            lines = xml_str.split('\n')[1:]
            f.write('\n'.join(lines))
        
        print(f"Converted MuJoCo XML saved to: {output_file}")


def main():

    parser = argparse.ArgumentParser(description="URDF to XML converter")

    parser.add_argument("--urdf", required=True, type=str, help="Path to URDF file")
    parser.add_argument("--xml", default=None, type=str, help="Output MJCF file")

    args = parser.parse_args()

    output_path = args.xml if args.xml else args.urdf.replace(".urdf", ".xml")

    converter = URDFToMuJoCoConverter(args.urdf)
    converter.save_to_file(output_path)


if __name__ == '__main__':
    main()
