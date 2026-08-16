# gemma-at-the-edge
testing openweight google gemma models on edge devices

"google/gemma-4/transformers/gemma-4-26b-a4b"

https://www.kaggle.com/models/google/gemma-4

https://gemma-llm.readthedocs.io/en/latest/checkpoints.html


```python
import kagglehub

kagglehub.login()

weights_dir = kagglehub.model_download("google/gemma-4/transformers/gemma-4-26b-a4b")
```

PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 uv run python main.py
