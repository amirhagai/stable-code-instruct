# stable-code-instruct

installation steps - 

1. clone the repo
2. cd stable-code-instruct
3. docker build -t transformers docker/
4. docker run --shm-size=5g -it -p 7860:7860 --gpus all --name huggingface_container -v ./stable_code_ai:/app transformers
5. conda activate HF
5. python /app/infer.py

