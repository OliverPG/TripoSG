@REM set HF_ENDPOINT=https://hf-mirror.com
@REM python -m scripts.inference_triposg --image-input assets/example_data/hjswed.png --faces 100 --output-path ./output.glb
@REM python -m scripts.test_multiview_fixed_v3
@REM python -m scripts.test_multiview_optimized
@REM python -m scripts.test_multiview_comparison --generate-test
python -m scripts.test_multiview_comparison --generate-test --skip-single-view
