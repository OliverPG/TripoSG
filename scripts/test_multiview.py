#!/usr/bin/env python3
"""
多视角3D生成测试脚本
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_view_inference import MultiViewTripoSG
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
import torch

def test_multiview_generation():
    """测试多视角生成功能"""
    print("初始化多视角3D生成器...")
    
    # 初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = TripoSGPipeline.from_pretrained(
        "pretrained_weights/TripoSG",
        torch_dtype=torch.float16
    ).to(device)
    
    generator = MultiViewTripoSG(pipe)
    
    # 测试数据（使用示例图像）
    image_paths = [
        "assets/example_data/hjswed.png",  # 前视图
        "assets/example_data/iuvbww.png",  # 侧视图
        "assets/example_data/jkghed.png",  # 顶视图
    ]
    view_angles = [0, 90, 180]  # 对应的视角角度
    
    print(f"使用 {len(image_paths)} 个视角生成3D模型...")
    
    try:
        mesh = generator.generate_from_multiview(
            image_paths=image_paths,
            view_angles=view_angles,
            num_inference_steps=30,  # 测试时使用较少的步数
            guidance_scale=7.0,
            faces=1000,
            seed=42
        )
        
        # 保存结果
        mesh.export("test_multiview_output.glb")
        print("✅ 多视角生成测试成功！")
        print(f"生成网格信息：")
        print(f"  - 顶点数: {len(mesh.vertices)}")
        print(f"  - 面数: {len(mesh.faces)}")
        print(f"  - 文件已保存: test_multiview_output.glb")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_multiview_generation()