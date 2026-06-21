FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# For MODE=nnunet, also:
# RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
#  && pip install --no-cache-dir nnunetv2
COPY . .
ENV MODE=nnunet \
    nnUNet_raw=/app/data/nnUNet_raw \
    nnUNet_preprocessed=/app/data/nnUNet_preprocessed \
    nnUNet_results=/app/data/nnUNet_results
EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]
