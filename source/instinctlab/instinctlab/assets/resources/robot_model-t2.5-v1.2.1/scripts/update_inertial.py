import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import re
class Origin:
    def __init__(self):
        self._px = 0.0
        self._py = 0.0
        self._pz = 0.0
        self._roll = 0.0
        self._pitch = 0.0
        self._yaw = 0.0
    
    @property
    def px(self) -> float:
        return self._px
    
    @property
    def py(self) -> float:
        return self._py
    
    @property
    def pz(self) -> float:
        return self._pz
    
    @property
    def roll(self) -> float:
        return self._roll
    
    @property
    def pitch(self) -> float:
        return self._pitch
    
    @property
    def yaw(self) -> float:
        return self._yaw

class Inertia:
    def __init__(self):
        self._ixx = 0.0
        self._ixy = 0.0
        self._ixz = 0.0
        self._iyy = 0.0
        self._iyz = 0.0
        self._izz = 0.0
    
    @property
    def ixx(self) -> float:
        return self._ixx
    
    @property
    def ixy(self) -> float:
        return self._ixy
    
    @property
    def ixz(self) -> float:
        return self._ixz
    
    @property
    def iyy(self) -> float:
        return self._iyy
    
    @property
    def iyz(self) -> float:
        return self._iyz
    
    @property
    def izz(self) -> float:
        return self._izz

class Properties:
    def __init__(self):
        self._name = ""
        self._mass = 0.0
        self.origin = Origin()
        self.inertia = Inertia()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def mass(self) -> float:
        return self._mass

class PropertiesParser:
    def __init__(self):
        self.all_properties = []
    
    def parse_file(self, file_path):
        """解析整个文件，识别独立一行的组件名称作为头"""
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        current_component_name = None
        current_component_lines = []
        
        for line in lines:
            line = line.strip()
            
            # 检查是否是独立一行的组件名称
            # 匹配类似 "left_hip_pitch" 这样的格式（单独一行）
            if self.is_component_header(line):
                # 如果已经有在处理的组件，先保存它
                if current_component_name and current_component_lines:
                    properties = self.parse_component(current_component_name, current_component_lines)
                    if properties:
                        self.all_properties.append(properties)
                
                # 开始新的组件
                current_component_name = line
                current_component_lines = []
            elif current_component_name is not None:
                # 如果不是空行，添加到当前组件的行中
                if line:
                    current_component_lines.append(line)
        
        # 处理最后一个组件
        if current_component_name and current_component_lines:
            properties = self.parse_component(current_component_name, current_component_lines)
            if properties:
                self.all_properties.append(properties)
        
        return self.all_properties
    
    def is_component_header(self, line):
        """判断一行是否是组件名称头"""
        if not line:
            return False
        
        # 匹配两种模式：
        # 1. 类似 "left_hip_pitch", "right_knee_roll" 这样的下划线命名
        # 2. 类似 "torso", "head", "base" 这样的单独单词
        pattern1 = r'^[a-z]+(_[a-z]+)+$'  # 下划线命名
        pattern2 = r'^[a-z]+$'            # 单独单词
        
        return re.match(pattern1, line) is not None or re.match(pattern2, line) is not None
    
    def parse_component(self, component_name, lines):
        """解析单个组件的属性"""
        properties = Properties()
        properties._name = component_name
        
        text = '\n'.join(lines)
        
        # 解析质量
        mass_match = re.search(r'质量\s*=\s*([0-9.]+)\s*千克', text)
        if mass_match:
            properties._mass = float(mass_match.group(1))
        
        # 解析重心坐标
        px_match = re.search(r'X\s*=\s*([-0-9.]+)', text)
        py_match = re.search(r'Y\s*=\s*([-0-9.]+)', text)
        pz_match = re.search(r'Z\s*=\s*([-0-9.]+)', text)
        
        if px_match:
            properties.origin._px = float(px_match.group(1))
        if py_match:
            properties.origin._py = float(py_match.group(1))
        if pz_match:
            properties.origin._pz = float(pz_match.group(1))
        
        # 解析惯性张量（由输出坐标系决定的部分）
        inertia_data = self.extract_inertia_tensor(text)
        if inertia_data:
            properties.inertia._ixx = inertia_data.get('ixx', 0.0)
            properties.inertia._ixy = inertia_data.get('ixy', 0.0)
            properties.inertia._ixz = inertia_data.get('ixz', 0.0)
            properties.inertia._iyy = inertia_data.get('iyy', 0.0)
            properties.inertia._iyz = inertia_data.get('iyz', 0.0)
            properties.inertia._izz = inertia_data.get('izz', 0.0)
        
        return properties
    
    def extract_inertia_tensor(self, text):
        """从文本中提取惯性张量数据"""
        inertia_data = {}
        
        # 查找"由输出座标系决定"的部分
        output_coord_section = re.search(r'由输出座标系决定[^I]*(Ixx\s*=\s*[0-9.-]+.*?Izz\s*=\s*[0-9.-]+)', text, re.DOTALL)
        
        if output_coord_section:
            inertia_text = output_coord_section.group(1)
            
            patterns = {
                'ixx': r'Ixx\s*=\s*([0-9.-]+)',
                'ixy': r'Ixy\s*=\s*([0-9.-]+)', 
                'ixz': r'Ixz\s*=\s*([0-9.-]+)',
                'iyy': r'Iyy\s*=\s*([0-9.-]+)',
                'iyz': r'Iyz\s*=\s*([0-9.-]+)',
                'izz': r'Izz\s*=\s*([0-9.-]+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, inertia_text)
                if match:
                    inertia_data[key] = float(match.group(1))
        
        return inertia_data
def main():
    # 创建解析器
    parser = PropertiesParser()
    
    # 解析文件
    properties_list = parser.parse_file('output.txt')
    
    # 打印结果验证
    print(f"共找到 {len(properties_list)} 个组件:")
    print("=" * 60)
    
    for prop in properties_list:
        print(f"组件名称: {prop.name}")
        print(f"质量: {prop.mass} kg")
        print(f"重心坐标: ({prop.origin.px:.6f}, {prop.origin.py:.6f}, {prop.origin.pz:.6f})")
        print(f"惯性张量:")
        print(f"  Ixx: {prop.inertia.ixx:.6f}")
        print(f"  Ixy: {prop.inertia.ixy:.6f}")
        print(f"  Ixz: {prop.inertia.ixz:.6f}")
        print(f"  Iyy: {prop.inertia.iyy:.6f}")
        print(f"  Iyz: {prop.inertia.iyz:.6f}")
        print(f"  Izz: {prop.inertia.izz:.6f}")
        print("-" * 50)


class URDFUpdater:
    def __init__(self, properties_list):
        """
        初始化URDF更新器
        
        Args:
            properties_list: 从PropertiesParser解析得到的属性列表
        """
        self.properties_list = properties_list
        # 创建名称映射：去掉"_link"后缀的link名称到Properties对象的映射
        self.name_to_properties = {}
        for prop in properties_list:
            # 将属性名称映射到link名称（添加_link后缀）
            if prop.name == "pelvis":
                link_name = "pelvis"
            else:
                link_name = prop.name + "_link"
            self.name_to_properties[link_name] = prop
    
    def update_urdf_file(self, urdf_file_path, output_file_path=None):
        """
        更新URDF文件中的惯量信息
        
        Args:
            urdf_file_path: 输入的URDF文件路径
            output_file_path: 输出的URDF文件路径，如果为None则覆盖原文件
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 解析URDF文件
            tree = ET.parse(urdf_file_path)
            root = tree.getroot()
            
            # 更新所有匹配的link
            updated_links = self._update_links(root)
            
            # 保存文件
            if output_file_path is None:
                output_file_path = urdf_file_path
            
            # 保持XML格式
            tree.write(output_file_path, encoding='utf-8', xml_declaration=True)
            
            print(f"成功更新URDF文件: {output_file_path}")
            print(f"更新了 {updated_links} 个link的惯量信息")
            
            # 打印更新详情
            self._print_update_summary()
            
            return True
            
        except Exception as e:
            print(f"更新URDF文件失败: {e}")
            return False
    
    def _update_links(self, root):
        """更新所有匹配的link"""
        updated_count = 0
        
        for link in root.findall('link'):
            link_name = link.get('name')
            if link_name in self.name_to_properties:
                if self._update_link_inertial(link, self.name_to_properties[link_name]):
                    updated_count += 1
                    print(f"✓ 已更新link: {link_name}")
                else:
                    print(f"✗ 更新link失败: {link_name}")
            else:
                print(f"- 未找到匹配的属性: {link_name}")
        
        return updated_count
    
    def _update_link_inertial(self, link_element, properties):
        """
        更新单个link的inertial元素
        
        Args:
            link_element: link的XML元素
            properties: Properties对象
        
        Returns:
            bool: 更新是否成功
        """
        try:
            # 查找或创建inertial元素
            inertial = link_element.find('inertial')
            if inertial is None:
                inertial = ET.SubElement(link_element, 'inertial')
            
            # 更新origin
            origin_elem = inertial.find('origin')
            if origin_elem is None:
                origin_elem = ET.SubElement(inertial, 'origin')
            
            # 设置origin的xyz属性（注意：URDF中的坐标单位是米）
            origin_elem.set('xyz', f"{properties.origin.px} {properties.origin.py} {properties.origin.pz}")
            # 设置rpy属性（如果没有欧拉角信息，保持默认值）
            origin_elem.set('rpy', "0 0 0")
            
            # 更新mass
            mass_elem = inertial.find('mass')
            if mass_elem is None:
                mass_elem = ET.SubElement(inertial, 'mass')
            mass_elem.set('value', str(properties.mass))
            
            # 更新inertia
            inertia_elem = inertial.find('inertia')
            if inertia_elem is None:
                inertia_elem = ET.SubElement(inertial, 'inertia')
            
            # 设置惯性张量
            inertia_elem.set('ixx', str(properties.inertia.ixx))
            inertia_elem.set('ixy', str(properties.inertia.ixy))
            inertia_elem.set('ixz', str(properties.inertia.ixz))
            inertia_elem.set('iyy', str(properties.inertia.iyy))
            inertia_elem.set('iyz', str(properties.inertia.iyz))
            inertia_elem.set('izz', str(properties.inertia.izz))
            
            return True
            
        except Exception as e:
            print(f"更新link inertial失败: {e}")
            return False
    
    def _print_update_summary(self):
        """打印更新摘要"""
        print("\n" + "="*60)
        print("更新摘要:")
        print("="*60)
        
        # for link_name, prop in self.name_to_properties.items():
        #     print(f"Link: {link_name}")
        #     print(f"  质量: {prop.mass:.6f} kg")
        #     print(f"  重心: ({prop.origin.px:.6f}, {prop.origin.py:.6f}, {prop.origin.pz:.6f})")
        #     print(f"  惯性张量:")
        #     print(f"    ixx: {prop.inertia.ixx:.6f}")
        #     print(f"    ixy: {prop.inertia.ixy:.6f}")
        #     print(f"    ixz: {prop.inertia.ixz:.6f}")
        #     print(f"    iyy: {prop.inertia.iyy:.6f}")
        #     print(f"    iyz: {prop.inertia.iyz:.6f}")
        #     print(f"    izz: {prop.inertia.izz:.6f}")
        #     print()

def main():
    """
    主函数：读取属性文件并更新URDF
    """
    # 1. 读取属性文件
    parser1 = argparse.ArgumentParser(description="Update script example.")
    parser1.add_argument("-t", "--txt", type=str, required=True, help="Input txt")
    parser1.add_argument("-i", "--input", type=str, required=True, help="Input value")
    parser1.add_argument("-o", "--output", type=str, required=True, help="Output value")
    args = parser1.parse_args()

    parser = PropertiesParser()
    properties_list = parser.parse_file(args.txt)

    print(f"从属性文件中读取到 {len(properties_list)} 个组件")
    
    # 2. 创建URDF更新器
    updater = URDFUpdater(properties_list)
    
    # 3. 更新URDF文件
    urdf_file = args.input  # 从命令行参数获取URDF文件路径
    output_file = args.output  # 从命令行参数获取输出文件路径

    success = updater.update_urdf_file(urdf_file, output_file)
    
    if success:
        print(f"\n✅ URDF文件更新完成!")
        print(f"输入文件: {urdf_file}")
        print(f"输出文件: {output_file}")
    else:
        print(f"\n❌ URDF文件更新失败!")

if __name__ == "__main__":
    main()