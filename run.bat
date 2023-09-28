@REM set CUDA_VISIBLE_DEVICES=""
set PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.9,max_split_size_mb:512
".venv/Scripts/python.exe" infer-web.py --port 7897