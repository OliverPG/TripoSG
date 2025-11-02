#!/usr/bin/env python3
"""
多视角图像生成脚本
生成一个立方体和一个球前后并排放置的三视图和透视图
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image, ImageDraw

def create_cube_and_sphere_structure(size=512, output_dir="multiview_images"):
    """创建立方体和球前后并排放置的多视角图像"""
    print("🎨 创建立方体+球结构的多视角图像...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    images = []
    image_paths = []
    
    # 1. 前视图（正视图）
    img_front = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img_front)
    
    # 绘制立方体（左侧）
    cube_x1, cube_y1 = size//4, size//3
    cube_x2, cube_y2 = size//2 - size//8, 2*size//3
    draw.rectangle([cube_x1, cube_y1, cube_x2, cube_y2], 
                   fill='lightblue', outline='blue', width=3)
    
    # 绘制球体（右侧，稍微靠后）
    sphere_center_x = 3*size//4
    sphere_center_y = size//2
    sphere_radius = size//6
    draw.ellipse([sphere_center_x-sphere_radius, sphere_center_y-sphere_radius,
                  sphere_center_x+sphere_radius, sphere_center_y+sphere_radius],
                 fill='lightcoral', outline='red', width=3)
    
    # 添加阴影效果增强立体感
    draw.rectangle([cube_x1+5, cube_y1+5, cube_x2+5, cube_y2+5], 
                   fill='#e0e0e0', outline='#a0a0a0', width=1)
    draw.ellipse([sphere_center_x-sphere_radius+3, sphere_center_y-sphere_radius+3,
                  sphere_center_x+sphere_radius+3, sphere_center_y+sphere_radius+3],
                 fill='#f0f0f0', outline='#c0c0c0', width=1)
    
    draw.text((size//2-30, size//8), "Front View", fill='black', font_size=20)
    images.append(img_front)
    
    # 2. 侧视图（右视图）
    img_side = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img_side)
    
    # 绘制立方体侧面（前方物体）
    cube_side_x1, cube_side_y1 = size//3, size//3
    cube_side_x2, cube_side_y2 = 2*size//3, 2*size//3
    draw.rectangle([cube_side_x1, cube_side_y1, cube_side_x2, cube_side_y2],
                   fill='lightgreen', outline='green', width=3)
    
    # 绘制球体侧面（后方物体，稍微小一些）
    sphere_side_center_x = 2*size//3 + size//12
    sphere_side_center_y = size//2
    sphere_side_radius = size//8
    draw.ellipse([sphere_side_center_x-sphere_side_radius, sphere_side_center_y-sphere_side_radius,
                  sphere_side_center_x+sphere_side_radius, sphere_side_center_y+sphere_side_radius],
                 fill='lightyellow', outline='orange', width=3)
    
    # 添加阴影效果
    draw.rectangle([cube_side_x1+4, cube_side_y1+4, cube_side_x2+4, cube_side_y2+4],
                   fill='#e8e8e8', outline='#b0b0b0', width=1)
    draw.ellipse([sphere_side_center_x-sphere_side_radius+2, sphere_side_center_y-sphere_side_radius+2,
                  sphere_side_center_x+sphere_side_radius+2, sphere_side_center_y+sphere_side_radius+2],
                 fill='#f8f8f8', outline='#d0d0d0', width=1)
    
    draw.text((size//2-25, size//8), "Side View", fill='black', font_size=20)
    images.append(img_side)
    
    # 3. 俯视图（顶视图）
    img_top = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img_top)
    
    # 绘制立方体顶部（前方物体）
    cube_top_x1, cube_top_y1 = size//4, size//4
    cube_top_x2, cube_top_y2 = size//2, size//2
    draw.rectangle([cube_top_x1, cube_top_y1, cube_top_x2, cube_top_y2],
                   fill='lightpink', outline='purple', width=3)
    
    # 绘制球体顶部（后方物体）
    sphere_top_center_x = 3*size//4
    sphere_top_center_y = size//2
    sphere_top_radius = size//8
    draw.ellipse([sphere_top_center_x-sphere_top_radius, sphere_top_center_y-sphere_top_radius,
                  sphere_top_center_x+sphere_top_radius, sphere_top_center_y+sphere_top_radius],
                 fill='lightcyan', outline='teal', width=3)
    
    # 添加阴影效果
    draw.rectangle([cube_top_x1+3, cube_top_y1+3, cube_top_x2+3, cube_top_y2+3],
                   fill='#f0f0f0', outline='#c0c0c0', width=1)
    draw.ellipse([sphere_top_center_x-sphere_top_radius+2, sphere_top_center_y-sphere_top_radius+2,
                  sphere_top_center_x+sphere_top_radius+2, sphere_top_center_y+sphere_top_radius+2],
                 fill='#f8f8f8', outline='#d8d8d8', width=1)
    
    draw.text((size//2-25, size//8), "Top View", fill='black', font_size=20)
    images.append(img_top)
    
    # 4. 透视图（45度角视角）
    img_perspective = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(img_perspective)
    
    # 绘制透视效果的立方体（前方）
    cube_perspective = [
        (size//3, size//3),        # 左上
        (size//2, size//4),        # 右上（透视）
        (size//2, 2*size//3),      # 右下
        (size//3, 2*size//3)       # 左下
    ]
    draw.polygon(cube_perspective, fill='lightblue', outline='blue', width=3)
    
    # 绘制透视效果的球体（后方）
    sphere_perspective_center_x = 2*size//3
    sphere_perspective_center_y = size//2
    sphere_perspective_radius = size//7
    # 椭圆模拟透视效果
    draw.ellipse([sphere_perspective_center_x-sphere_perspective_radius, 
                  sphere_perspective_center_y-sphere_perspective_radius//2,
                  sphere_perspective_center_x+sphere_perspective_radius, 
                  sphere_perspective_center_y+sphere_perspective_radius//2],
                 fill='lightcoral', outline='red', width=3)
    
    # 添加深度阴影效果
    shadow_points = [
        (size//3+10, size//3+10),
        (size//2+8, size//4+8),
        (size//2+8, 2*size//3+8),
        (size//3+10, 2*size//3+10)
    ]
    draw.polygon(shadow_points, fill='#d0d0d0', outline='#a0a0a0', width=1)
    
    draw.text((size//2-35, size//8), "Perspective View", fill='black', font_size=20)
    images.append(img_perspective)
    
    # 保存所有图像
    view_names = ["front", "side", "top", "perspective"]
    for i, (img, view_name) in enumerate(zip(images, view_names)):
        filename = f"{output_dir}/cube_sphere_{view_name}.png"
        img.save(filename)
        image_paths.append(filename)
        print(f"✅ 保存{view_name}视图: {filename}")
    
    print(f"🎯 生成完成！共创建{len(images)}个视角图像")
    return image_paths

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成立方体+球结构的多视角图像')
    parser.add_argument('--output-dir', default='multiview_images', 
                       help='图像保存目录（默认：multiview_images）')
    parser.add_argument('--image-size', type=int, default=512,
                       help='图像尺寸（默认：512）')
    
    args = parser.parse_args()
    
    # 生成多视角图像
    image_paths = create_cube_and_sphere_structure(
        size=args.image_size, 
        output_dir=args.output_dir
    )
    
    print(f"\n📁 所有图像已保存到: {args.output_dir}")
    print("📋 图像列表:")
    for path in image_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    main()