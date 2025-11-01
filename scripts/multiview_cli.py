#!/usr/bin/env python3
import argparse
import os
import sys
import glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_view_inference import MultiViewTripoSG
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
import torch

def main():
    parser = argparse.ArgumentParser(description="多视角3D模型生成工具")
    
    # 必需参数
    parser.add_argument("--image-dir", type=str, required=True,
                       help="包含多视角图像的目录")
    parser.add_argument("--view-angles", type=str, required=True,
                       help="视角角度列表，用逗号分隔，如：0,45,90,135,180,225,270,315")
    
    # 可选参数
    parser.add_argument("--output", type=str, default="./multiview_output.glb",
                       help="输出文件路径")
    parser.add_argument("--steps", type=int, default=50,
                       help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=7.0,
                       help="引导尺度")
    parser.add_argument("--faces", type=int, default=2000,
                       help="目标面数")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子")
    
    args = parser.parse_args()
    
    # 处理视角角度
    view_angles = [float(angle.strip()) for angle in args.view_angles.split(",")]
    
    # 获取图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
    
    image_paths.sort()  # 确保顺序一致
    
    if len(image_paths) != len(view_angles):
        print(f"错误：图像数量({len(image_paths)})与视角数量({len(view_angles)})不匹配")
        return
    
    # 初始化模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = TripoSGPipeline.from_pretrained(
        "pretrained_weights/TripoSG",
        torch_dtype=torch.float16
    ).to(device)
    
    # 生成模型
    generator = MultiViewTripoSG(pipe)
    mesh = generator.generate_from_multiview(
        image_paths=image_paths,
        view_angles=view_angles,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        faces=args.faces,
        seed=args.seed
    )
    
    # 保存结果
    mesh.export(args.output)
    print(f"多视角3D模型已保存至: {args.output}")

if __name__ == "__main__":
    main()