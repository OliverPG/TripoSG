import sys
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from triposg.pipelines.pipeline_triposg import TripoSGPipeline
import trimesh

class MultiViewTripoSG:
    def __init__(self, pipe: TripoSGPipeline):
        self.pipe = pipe
        
    def encode_multiview_images(self, image_paths: List[str], view_angles: List[float]):
        """编码多视角图像特征"""
        image_embeddings = []
        
        for img_path, angle in zip(image_paths, view_angles):
            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            
            # 使用DINOv2编码图像特征
            image_embeds, _ = self.pipe.encode_image(
                image, 
                device=self.pipe.device,
                num_images_per_prompt=1
            )
            
            # 添加视角信息（角度编码）
            angle_embedding = self._encode_view_angle(angle)
            # 将角度编码扩展到与图像嵌入相同的维度
            expanded_angle_embedding = angle_embedding.expand_as(image_embeds)
            combined_embedding = torch.cat([image_embeds, expanded_angle_embedding], dim=-1)
            
            image_embeddings.append(combined_embedding)
        
        # 融合多视角特征
        fused_embedding = self._fuse_multiview_embeddings(image_embeddings)
        return fused_embedding
    
    def _encode_view_angle(self, angle: float):
        """编码视角角度信息"""
        # 使用正弦余弦编码
        angle_rad = torch.tensor([angle * np.pi / 180.0])
        encoding = torch.cat([
            torch.sin(angle_rad),
            torch.cos(angle_rad)
        ]).unsqueeze(0).unsqueeze(0)
        return encoding.to(self.pipe.device)
    
    def _fuse_multiview_embeddings(self, embeddings: List[torch.Tensor]):
        """融合多视角特征"""
        # 简单加权平均
        if len(embeddings) == 0:
            return None
            
        # 计算每个视角的权重（可根据视角质量调整）
        weights = torch.softmax(torch.ones(len(embeddings)), dim=0)
        fused = sum(w * emb for w, emb in zip(weights, embeddings))
        return fused
    
    def generate_from_multiview(self, 
                              image_paths: List[str], 
                              view_angles: List[float],
                              num_inference_steps: int = 50,
                              guidance_scale: float = 7.0,
                              faces: int = -1,
                              seed: int = 42) -> trimesh.Trimesh:
        """基于多视角图像生成3D模型"""
        # 编码多视角特征
        fused_embedding = self.encode_multiview_images(image_paths, view_angles)
        
        # 设置随机种子
        generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        
        # 调用修改后的pipeline
        outputs = self.pipe(
            image=None,  # 不使用单张图像
            multiview_embeddings=fused_embedding,
            use_multiview=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).samples[0]
        
        mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
        
        # 网格简化（可选）
        if faces > 0:
            mesh = self.simplify_mesh(mesh, faces)
            
        return mesh
    
    def simplify_mesh(self, mesh: trimesh.Trimesh, n_faces: int):
        """简化网格面数"""
        if mesh.faces.shape[0] > n_faces:
            # 使用pymeshlab进行简化
            try:
                import pymeshlab
                ms = pymeshlab.MeshSet()
                ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
                simplified_mesh = ms.current_mesh()
                return trimesh.Trimesh(vertices=simplified_mesh.vertex_matrix(), 
                                     faces=simplified_mesh.face_matrix())
            except ImportError:
                print("pymeshlab未安装，跳过网格简化")
                return mesh
        return mesh

def main():
    """多视角推理主函数"""
    # 初始化管道
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    # 加载预训练模型
    pipe = TripoSGPipeline.from_pretrained(
        "pretrained_weights/TripoSG",
        torch_dtype=dtype
    ).to(device)
    
    # 创建多视角生成器
    multiview_generator = MultiViewTripoSG(pipe)
    
    # 示例：使用8个视角
    image_paths = [
        "view_0.jpg", "view_45.jpg", "view_90.jpg", "view_135.jpg",
        "view_180.jpg", "view_225.jpg", "view_270.jpg", "view_315.jpg"
    ]
    view_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # 生成3D模型
    mesh = multiview_generator.generate_from_multiview(
        image_paths=image_paths,
        view_angles=view_angles,
        num_inference_steps=50,
        guidance_scale=7.0,
        faces=2000,
        seed=42
    )
    
    # 保存模型
    mesh.export("multiview_output.glb")
    print("多视角3D模型生成完成，保存为 multiview_output.glb")

if __name__ == "__main__":
    main()