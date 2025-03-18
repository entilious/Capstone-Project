FROM python:3.9
LABEL authors="jaish"

# Set the working directory
WORKDIR /webapp/backend

# Install system dependencies
RUN apt update && apt install -y curl

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Pull DeepSeek-R1:7B model during build
RUN ollama serve & sleep 5 && ollama pull phi3.5 # deepseek-r1:7b

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application into the container
COPY . .

# Expose necessary ports
#Ollama runs on 11434 by default
EXPOSE 8000 11434

# Start Ollama in the background, then start Uvicorn
CMD ["bash", "-c", "ollama serve & sleep 5 && uvicorn backend:app --host 0.0.0.0 --port 8000"]

