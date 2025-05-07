FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt')" && \
    python -m spacy download en_core_web_sm

COPY . .

RUN chmod +x run_pipeline.sh

ENV PYTHONPATH=/app

CMD ["/bin/bash"]
