@echo off
echo ===============================================
echo TripoSG 示例配置脚本
echo ===============================================
echo.

echo 1. 快速测试模式 (约1-2分钟)
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 25 --guidance-scale 6.0 --faces 1000
echo.

echo 2. 平衡质量模式 (约2-3分钟)  
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 50 --guidance-scale 7.0 --faces 3000
echo.

echo 3. 高质量模式 (约3-5分钟)
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 80 --guidance-scale 8.5 --faces 6000
echo.

echo 4. 最高质量模式 (约5-8分钟)
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 100 --guidance-scale 9.0 --faces 8000
echo.

echo 参数说明:
echo --num-inference-steps: 推理步骤数 (20-100, 影响质量/速度)
echo --guidance-scale: 引导尺度 (5.0-10.0, 影响忠实度)  
echo --faces: 目标面数 (1000-8000, 影响细节/文件大小)
echo.

pause