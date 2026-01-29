import xml.etree.ElementTree as ET
import pandas as pd

def simple_urdf_to_csv(urdf_file, csv_file):
    """
    简化版：直接将URDF关节限位导出到CSV
    
    Args:
        urdf_file (str): 输入URDF文件路径
        csv_file (str): 输出CSV文件路径
    """
    # 解析URDF
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    joints_data = []
    
    for joint in root.findall('joint'):
        joint_data = {
            'name': joint.get('name', 'unknown'),
            'type': joint.get('type', 'unknown')
        }
        
        limit = joint.find('limit')
        if limit is not None:
            joint_data.update({
                'lower': limit.get('lower', 'None'),
                'upper': limit.get('upper', 'None'),
                'effort': limit.get('effort', 'None'),
                'velocity': limit.get('velocity', 'None')
            })
        else:
            joint_data.update({
                'lower': 'None',
                'upper': 'None',
                'effort': 'None',
                'velocity': 'None'
            })
        
        joints_data.append(joint_data)
    
    # 创建DataFrame并保存
    df = pd.DataFrame(joints_data)
    df.to_csv(csv_file, index=False)
    print(f"成功导出 {len(joints_data)} 个关节信息到 {csv_file}")

# 使用示例
if __name__ == "__main__":
    # 直接指定文件路径
    urdf_path = "x2.urdf"  # 替换为你的URDF文件路径
    csv_path = "x2.csv"  # 输出的CSV文件路径
    
    simple_urdf_to_csv(urdf_path, csv_path)