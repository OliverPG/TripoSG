@echo off
echo ===============================================
echo TripoSG ʾ�����ýű�
echo ===============================================
echo.

echo 1. ���ٲ���ģʽ (Լ1-2����)
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 25 --guidance-scale 6.0 --faces 1000
echo.

echo 2. ƽ������ģʽ (Լ2-3����)  
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 50 --guidance-scale 7.0 --faces 3000
echo.

echo 3. ������ģʽ (Լ3-5����)
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 80 --guidance-scale 8.5 --faces 6000
echo.

echo 4. �������ģʽ (Լ5-8����)
echo python inference_triposg.py --image-input input.jpg --num-inference-steps 100 --guidance-scale 9.0 --faces 8000
echo.

echo ����˵��:
echo --num-inference-steps: �������� (20-100, Ӱ������/�ٶ�)
echo --guidance-scale: �����߶� (5.0-10.0, Ӱ����ʵ��)  
echo --faces: Ŀ������ (1000-8000, Ӱ��ϸ��/�ļ���С)
echo.

pause