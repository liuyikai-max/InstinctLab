#!/usr/bin/env python3
"""
URDF质量分析工具
读取URDF文件中的关节质量信息并导出到CSV
"""

import xml.etree.ElementTree as ET
import csv
import os
import sys
import argparse

# 定义关节组
JOINT_GROUPS = {
    "left_leg": [
        'left_hip_pitch_joint',      # 0
        'left_hip_roll_joint',       # 1
        'left_hip_yaw_joint',        # 2
        'left_knee_joint',           # 3
        'left_ankle_pitch_joint',    # 4
        'left_ankle_roll_joint',     # 5
    ],
    "right_leg": [
        'right_hip_pitch_joint',      # 7
        'right_hip_roll_joint',      # 7
        'right_hip_yaw_joint',       # 8
        'right_knee_joint',          # 9
        'right_ankle_pitch_joint',   # 10
        'right_ankle_roll_joint',    # 11
    ],
    "left_arm": [
        'left_shoulder_pitch_joint',
        'left_shoulder_roll_joint',
        'left_shoulder_yaw_joint',
        'left_elbow_joint',
        'left_wrist_yaw_joint',
        'left_wrist_pitch_joint',
        'left_wrist_roll_joint',
    ],
    "right_arm": [
        'right_shoulder_pitch_joint',
        'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint',
        'right_elbow_joint',
        'right_wrist_yaw_joint',
        'right_wrist_pitch_joint',
        'right_wrist_roll_joint',
    ],
    "head": [
        'head_pitch_joint',
        'head_yaw_joint',
    ],
    "waist": [
        'pelvis',
        'waist_yaw_joint',
        'waist_pitch_joint',
        # 'torso_link',
    ]
}

def parse_urdf_mass(urdf_file):
    """
    解析URDF文件，提取每个关节的质量信息
    
    Args:
        urdf_file (str): URDF文件路径
        
    Returns:
        tuple: (关节质量列表, 总质量, 关节组质量字典)
    """
    try:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"URDF文件解析错误: {e}")
        return [], 0, {}
    except FileNotFoundError:
        print(f"找不到URDF文件: {urdf_file}")
        return [], 0, {}
    
    mass_data = []
    total_mass = 0.0
    link_mass_dict = {}  # 用于存储每个link的质量
    
    # 遍历所有link元素
    for link in root.findall('link'):
        link_name = link.get('name', 'unknown')
        mass = 0.0
        inertia_info = "No"
        com = [0.0, 0.0, 0.0]  # 质心位置
        
        # 查找inertial元素
        inertial = link.find('inertial')
        if inertial is not None:
            # 查找mass元素
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                mass = float(mass_elem.get('value', 0))
                total_mass += mass
                inertia_info = "Yes"
            
            # 查找质心位置
            origin = inertial.find('origin')
            if origin is not None:
                xyz = origin.get('xyz', '0 0 0')
                com = [float(x) for x in xyz.split()]
        
        mass_data.append({
            'link_name': link_name,
            'mass': mass,
            'has_inertial': inertia_info,
            'com_x': com[0],
            'com_y': com[1],
            'com_z': com[2]
        })
        
        # 存储到字典中供后续使用
        link_mass_dict[link_name] = mass
    
    # 计算关节组质量
    group_masses = calculate_group_masses(link_mass_dict)
    
    return mass_data, total_mass, group_masses

def calculate_group_masses(link_mass_dict):
    """
    计算各个关节组的总质量
    
    Args:
        link_mass_dict (dict): link名称到质量的映射
        
    Returns:
        dict: 关节组名称到总质量的映射
    """
    group_masses = {}
    
    for group_name, joint_list in JOINT_GROUPS.items():
        group_total = 0.0
        found_joints = []
        missing_joints = []
        
        for joint_name in joint_list:
            # 尝试找到对应的link（通常link名称与joint名称相关）
            # 这里假设link名称与joint名称相同或类似
            link_found = False
            
            # 尝试精确匹配
            if joint_name in link_mass_dict:
                group_total += link_mass_dict[joint_name]
                found_joints.append(joint_name)
                link_found = True
            else:
                # 尝试模糊匹配：将_joint替换为_link
                link_name = joint_name.replace('_joint', '_link')
                if link_name in link_mass_dict:
                    group_total += link_mass_dict[link_name]
                    found_joints.append(f"{joint_name} -> {link_name}")
                    link_found = True
                else:
                    # 尝试其他可能的命名模式
                    for link_key in link_mass_dict.keys():
                        if joint_name in link_key or link_key in joint_name:
                            group_total += link_mass_dict[link_key]
                            found_joints.append(f"{joint_name} -> {link_key}")
                            link_found = True
                            break
            
            if not link_found:
                missing_joints.append(joint_name)
        
        group_masses[group_name] = {
            'total_mass': group_total,
            'found_joints': found_joints,
            'missing_joints': missing_joints
        }
    
    return group_masses

def export_to_csv(mass_data, total_mass, group_masses, output_file):
    """
    将质量数据导出到CSV文件
    
    Args:
        mass_data (list): 质量数据列表
        total_mass (float): 总质量
        group_masses (dict): 关节组质量信息
        output_file (str): 输出CSV文件路径
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['link_name', 'mass', 'has_inertial', 'com_x', 'com_y', 'com_z']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for data in mass_data:
                writer.writerow(data)
            
            # 写入总质量信息
            csvfile.write(f"\n总质量,{total_mass:.6f}\n")
            
            # 写入关节组质量信息
            csvfile.write("\n关节组质量汇总\n")
            csvfile.write("组名称,总质量(kg),找到的关节,缺失的关节\n")
            
            for group_name, group_info in group_masses.items():
                found_str = "; ".join(group_info['found_joints']) if group_info['found_joints'] else "无"
                missing_str = "; ".join(group_info['missing_joints']) if group_info['missing_joints'] else "无"
                csvfile.write(f"{group_name},{group_info['total_mass']:.6f},{found_str},{missing_str}\n")
            
        print(f"质量数据已导出到: {output_file}")
        print(f"总质量: {total_mass:.6f} kg")
        
    except Exception as e:
        print(f"导出CSV文件时出错: {e}")

def print_summary(mass_data, total_mass, group_masses):
    """
    打印质量汇总信息
    
    Args:
        mass_data (list): 质量数据列表
        total_mass (float): 总质量
        group_masses (dict): 关节组质量信息
    """
    print("\n" + "="*80)
    print("URDF质量分析结果")
    print("="*80)
    
    print(f"{'关节名称':<30} {'质量(kg)':<12} {'有无惯性参数':<15} {'质心位置(x,y,z)':<20}")
    print("-"*80)
    
    for data in mass_data:
        com_str = f"({data['com_x']:.3f}, {data['com_y']:.3f}, {data['com_z']:.3f})"
        print(f"{data['link_name']:<30} {data['mass']:<12.6f} {data['has_inertial']:<15} {com_str:<20}")
    
    print("-"*80)
    print(f"{'总质量':<30} {total_mass:<12.6f} {'kg':<15}")
    
    # 打印关节组质量信息
    print("\n" + "="*80)
    print("关节组质量汇总")
    print("="*80)
    
    for group_name, group_info in group_masses.items():
        print(f"\n{group_name}: {group_info['total_mass']:.6f} kg")
        
        if group_info['found_joints']:
            print("  找到的关节:")
            for joint_info in group_info['found_joints']:
                print(f"    - {joint_info}")
        
        if group_info['missing_joints']:
            print("  缺失的关节:")
            for joint_name in group_info['missing_joints']:
                print(f"    - {joint_name}")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='URDF质量分析工具')
    parser.add_argument('--urdf_file', help='URDF文件路径')
    parser.add_argument('-o', '--output', default='urdf_mass_analysis.csv', 
                       help='输出CSV文件路径 (默认: urdf_mass_analysis.csv)')
    parser.add_argument('--no-csv', action='store_true', 
                       help='不导出CSV文件，仅显示结果')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.urdf_file):
        print(f"错误: 文件 '{args.urdf_file}' 不存在")
        sys.exit(1)
    
    # 解析URDF文件
    print(f"正在解析URDF文件: {args.urdf_file}")
    mass_data, total_mass, group_masses = parse_urdf_mass(args.urdf_file)
    
    if not mass_data:
        print("未找到任何质量数据，请检查URDF文件格式")
        sys.exit(1)
    
    # 显示汇总信息
    print_summary(mass_data, total_mass, group_masses)
    
    # 导出到CSV
    if not args.no_csv:
        export_to_csv(mass_data, total_mass, group_masses, args.output)

if __name__ == "__main__":
    main()