# Use Python 3.10 as base image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files to the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (FastAPI default port)
EXPOSE 7860

# Start FastAPI server with uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
