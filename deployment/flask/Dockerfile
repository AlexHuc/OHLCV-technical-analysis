# Use lightweight base image
FROM python:3.11-slim

# Install Pipenv
RUN pip install pipenv

# Set working directory
WORKDIR /app

# Copy Pipenv files first for dependency caching
COPY ["deployment/flask/Pipfile", "deployment/flask/Pipfile.lock", "./"]

# Install dependencies
RUN pipenv install --deploy --system

# Copy models and app code
RUN mkdir -p ./models
COPY ["deployment/flask/predict.py", "./"]
COPY ["models/horizontal_unet.pth", "./models/"]
COPY ["models/unet_ray.pth", "./models/"]

# Expose port
EXPOSE 9696

# Run the Flask app with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
