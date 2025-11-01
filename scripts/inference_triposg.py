import argparse
import os
import sys
from glob import glob
from typing import Any, Union

import numpy as np
import torch
import trimesh
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline
from image_process import prepare_image
from briarmbg import BriaRMBG

import pymeshlab


@torch.no_grad()
def run_triposg(
    pipe: Any,
    image_input: Union[str, Image.Image],
    rmbg_net: Any,
    seed: int,
    num_inference_steps: int = 20,
    guidance_scale: float = 7.0,
    faces: int = -1,
) -> trimesh.Scene:

    img_pil = prepare_image(image_input, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net)

    outputs = pipe(
        image=img_pil,
        generator=torch.Generator(device=pipe.device).manual_seed(seed),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).samples[0]
    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))

    if faces > 0:
        mesh = simplify_mesh(mesh, faces)

    return mesh

def mesh_to_pymesh(vertices, faces):
    mesh = pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    return ms

def pymesh_to_trimesh(mesh):
    verts = mesh.vertex_matrix()#.tolist()
    faces = mesh.face_matrix()#.tolist()
    return trimesh.Trimesh(vertices=verts, faces=faces)  #, vID, fID

def simplify_mesh(mesh: trimesh.Trimesh, n_faces):
    if mesh.faces.shape[0] > n_faces:
        ms = mesh_to_pymesh(mesh.vertices, mesh.faces)
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum = n_faces)
        return pymesh_to_trimesh(ms.current_mesh())
    else:
        return mesh

if __name__ == "__main__":
    device = "cuda"
    dtype = torch.float16

    parser = argparse.ArgumentParser(
        description="TripoSG 3D模型生成工具 - 从单张图像生成3D网格模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
参数调优指南:
=============

精度与速度平衡:
- 高质量模式: --num-inference-steps 100 --guidance-scale 7.0 --faces 5000
- 平衡模式: --num-inference-steps 50 --guidance-scale 7.0 --faces 2000  
- 快速模式: --num-inference-steps 25 --guidance-scale 5.0 --faces 1000

内存使用说明:
- 推理步骤越多，显存占用越高
- 面数设置过高可能导致显存不足
- 建议在8GB以上显存的GPU上运行

示例用法:
python inference_triposg.py --image-input input.jpg --faces 2000 --num-inference-steps 50
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--image-input", 
        type=str, 
        required=True,
        help="输入图像路径 (支持格式: JPG, PNG, BMP等)"
    )
    
    # 输出参数
    parser.add_argument(
        "--output-path", 
        type=str, 
        default="./output.glb",
        help="输出3D模型文件路径 (默认: ./output.glb)"
    )
    
    # 随机种子参数
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="随机种子，用于结果复现 (取值范围: 0-4294967295, 默认: 42)"
    )
    
    # 推理步骤参数
    parser.add_argument(
        "--num-inference-steps", 
        type=int, 
        default=20,
        choices=range(10, 201, 10),  # 10-200，步长为10
        metavar="[10-200]",
        help="""扩散模型的推理步骤数
取值范围: 10-200 (推荐: 25-100)
对精度的影响: 
  - 步骤越多，生成质量越高，细节更丰富
  - 步骤过少可能导致模型不完整或质量较差
对速度的影响:
  - 每增加10步，推理时间增加约20-30%
  - 50步约需1-2分钟，100步约需2-4分钟
推荐设置:
  - 高质量: 80-100步
  - 平衡: 40-60步  
  - 快速: 20-30步
        """
    )
    
    # 引导尺度参数
    parser.add_argument(
        "--guidance-scale", 
        type=float, 
        default=7.0,
        metavar="[1.0-20.0]",
        help="""分类器自由引导尺度 (CFG scale)
取值范围: 1.0-20.0 (推荐: 5.0-10.0)
对精度的影响:
  - 值越高，模型更忠实于输入图像
  - 值过低可能导致模型与输入图像相关性弱
  - 值过高可能导致模型过度锐化或失真
对速度的影响:
  - 对推理速度影响较小
  - 主要影响生成质量而非速度
推荐设置:
  - 标准质量: 7.0-8.0
  - 高保真: 9.0-10.0
  - 创意生成: 5.0-6.0
        """
    )
    
    # 面数参数
    parser.add_argument(
        "--faces", 
        type=int, 
        default=-1,
        metavar="[100-10000]",
        help="""输出网格的目标面数
取值范围: 100-10000 (默认: -1 表示不简化)
对精度的影响:
  - 面数越多，模型细节越丰富，文件越大
  - 面数过少可能导致细节丢失
对速度的影响:
  - 面数设置对推理速度影响较小
  - 主要影响后处理和文件大小
文件大小参考:
  - 1000面: 约100-500KB
  - 5000面: 约1-5MB  
  - 10000面: 约5-10MB
推荐设置:
  - 实时应用: 1000-2000面
  - 高质量渲染: 5000-8000面
  - 存档质量: 8000-10000面
        """
    )
    
    args = parser.parse_args()

    # 参数验证
    if args.num_inference_steps < 10 or args.num_inference_steps > 200:
        print("警告: 推理步骤数应在10-200范围内，使用默认值20")
        args.num_inference_steps = 50
        
    if args.guidance_scale < 1.0 or args.guidance_scale > 20.0:
        print("警告: 引导尺度应在1.0-20.0范围内，使用默认值7.0")
        args.guidance_scale = 7.0
        
    if args.faces != -1 and (args.faces < 100 or args.faces > 10000):
        print("警告: 面数应在100-10000范围内，使用默认值(不简化)")
        args.faces = -1

    # download pretrained weights
    triposg_weights_dir = "pretrained_weights/TripoSG"
    rmbg_weights_dir = "pretrained_weights/RMBG-1.4"
    snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=triposg_weights_dir)
    snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_weights_dir)

    # init rmbg model for background removal
    rmbg_net = BriaRMBG.from_pretrained(rmbg_weights_dir).to(device)
    rmbg_net.eval() 

    # init tripoSG pipeline
    pipe: TripoSGPipeline = TripoSGPipeline.from_pretrained(triposg_weights_dir).to(device, dtype)

    # run inference
    run_triposg(
        pipe,
        image_input=args.image_input,
        rmbg_net=rmbg_net,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        faces=args.faces,
    ).export(args.output_path)
    print(f"Mesh saved to {args.output_path}")