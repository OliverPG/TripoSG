#!/usr/bin/env python3
"""
从指定路径读取多张图片生成3D结构
"""
import os
import sys
import argparse
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
import torch
import numpy as np
from PIL import Image
import trimesh

def resize_image(image, max_size=512):
    """调整图像尺寸以优化内存使用"""
    width, height = image.size
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        print(f"调整图像尺寸: {width}x{height} -> {new_width}x{new_height}")
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def create_multiview_fusion(image_paths, max_width=1024):
    """创建多视角融合图像（网格布局）"""
    print("🖼️  创建多视角融合图像...")
    
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img = resize_image(img, 256)
            images.append(img)
            print(f"✅ 加载图像: {os.path.basename(img_path)}")
        else:
            print(f"❌ 图像文件不存在: {img_path}")
            return None
    
    if not images:
        print("❌ 没有有效的图像文件")
        return None
    
    # 计算网格布局
    num_images = len(images)
    if num_images <= 4:
        grid_cols = 2
        grid_rows = (num_images + 1) // 2
    else:
        grid_cols = 3
        grid_rows = (num_images + 2) // 3
    
    # 计算最终图像尺寸
    img_width, img_height = images[0].size
    total_width = img_width * grid_cols
    total_height = img_height * grid_rows
    
    # 如果总尺寸过大，进行缩放
    if total_width > max_width:
        scale_factor = max_width / total_width
        new_width = int(total_width * scale_factor)
        new_height = int(total_height * scale_factor)
        
        print(f"融合图像尺寸过大，进行缩放: {total_width}x{total_height} -> {new_width}x{new_height}")
        
        # 缩放所有图像
        resized_images = []
        for img in images:
            new_img_width = int(img.width * scale_factor)
            new_img_height = int(img.height * scale_factor)
            resized_img = img.resize((new_img_width, new_img_height), Image.Resampling.LANCZOS)
            resized_images.append(resized_img)
        
        images = resized_images
        img_width, img_height = images[0].size
        total_width = img_width * grid_cols
        total_height = img_height * grid_rows
    
    # 创建网格布局的融合图像
    fused_image = Image.new('RGB', (total_width, total_height), color='white')
    
    # 按网格布局粘贴图像
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        x_offset = col * img_width
        y_offset = row * img_height
        fused_image.paste(img, (x_offset, y_offset))
    
    print(f"✅ 网格融合图像完成，尺寸: {fused_image.size}，布局: {grid_rows}x{grid_cols}")
    return fused_image

def run_pipeline_optimized(pipe, image, device, params):
    """优化的管道运行函数"""
    pipe_params = {
        'image': image,
        'num_inference_steps': params['num_inference_steps'],
        'guidance_scale': params['guidance_scale'],
        'num_tokens': params['num_tokens'],
        'generator': torch.Generator(device=device).manual_seed(42)
    }
    
    if params.get('use_flash_decoder', True):
        pipe_params['use_flash_decoder'] = True
        pipe_params['flash_octree_depth'] = params.get('flash_octree_depth', 6)
    else:
        pipe_params['use_flash_decoder'] = False
        pipe_params['dense_octree_depth'] = params.get('dense_octree_depth', 6)
        pipe_params['hierarchical_octree_depth'] = params.get('hierarchical_octree_depth', 7)
    
    return pipe(**pipe_params)

def run_pipeline_with_timing(pipe, image, device, params):
    """带计时功能的管道运行函数"""
    start_time = time.time()
    result = run_pipeline_optimized(pipe, image, device, params)
    end_time = time.time()
    
    execution_time = end_time - start_time
    return result, execution_time

def save_optimized_result(result, filename):
    """保存优化版结果"""
    if hasattr(result, 'meshes') and result.meshes:
        mesh = result.meshes[0]
    else:
        mesh = trimesh.Trimesh(result.samples[0][0].astype(np.float32), 
                              np.ascontiguousarray(result.samples[0][1]))
    
    mesh.export(filename)
    print(f"✅ 网格已保存: {filename}")
    print(f"  顶点数: {len(mesh.vertices)}")
    print(f"  面数: {len(mesh.faces)}")
    return mesh

def generate_3d_from_images(image_paths, output_file="multiview_output.glb"):
    """从图像路径列表生成3D结构"""
    print("🚀 初始化3D生成器...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        pipe = TripoSGPipeline.from_pretrained(
            "pretrained_weights/TripoSG"
        ).to(device)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 检查图像文件存在性
    valid_paths = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            valid_paths.append(img_path)
        else:
            print(f"⚠️  图像文件不存在: {img_path}")
    
    if not valid_paths:
        print("❌ 没有有效的图像文件")
        return
    
    print(f"✅ 使用 {len(valid_paths)} 个视角生成3D模型...")
    
    try:
        # 创建多视角融合图像
        fused_image = create_multiview_fusion(valid_paths, max_width=512)
        if fused_image is None:
            return
        
        # 保存融合图像用于调试
        fused_image.save("multiview_fused_current.jpg")
        print("✅ 多视角图像融合完成")
        
        # 使用优化参数
        params = {
            'num_inference_steps': 25,
            'guidance_scale': 7.0,
            'num_tokens': 1024,
            'use_flash_decoder': True,
            'flash_octree_depth': 6,
        }
        
        result, execution_time = run_pipeline_with_timing(pipe, fused_image, device, params)
        
        print(f"✅ 推理完成！耗时: {execution_time:.2f}秒")
        
        # 保存结果
        save_optimized_result(result, output_file)
        print(f"🎯 3D生成完成！结果保存到: {output_file}")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从指定路径读取图片生成3D结构')
    parser.add_argument('--image-dir', help='包含多视角图像的目录路径')
    parser.add_argument('--image-paths', nargs='+', help='指定图像文件路径列表')
    parser.add_argument('--output', default='multiview_output.glb',
                       help='输出3D模型文件名（默认：multiview_output.glb）')
    
    args = parser.parse_args()
    
    # 获取图像路径列表
    image_paths = []
    
    if args.image_paths:
        # 使用指定的图像路径列表
        image_paths = args.image_paths
        print(f"📋 使用指定的图像列表: {len(image_paths)} 个文件")
        
    elif args.image_dir:
        # 从目录中读取所有PNG和JPG图像
        if os.path.exists(args.image_dir):
            for filename in os.listdir(args.image_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(args.image_dir, filename))
            print(f"📁 从目录读取图像: {len(image_paths)} 个文件")
        else:
            print(f"❌ 目录不存在: {args.image_dir}")
            return
    else:
        print("❌ 请指定 --image-dir 或 --image-paths 参数")
        return
    
    if not image_paths:
        print("❌ 没有找到有效的图像文件")
        return
    
    # 生成3D结构
    generate_3d_from_images(image_paths, args.output)

if __name__ == "__main__":
    main()