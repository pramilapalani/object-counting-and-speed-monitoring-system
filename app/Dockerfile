# Use an official lightweight Python image.
FROM python:3.9-slim

# Install system dependencies including libGL and libglib.
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container.
WORKDIR /app

# Copy requirements and install dependencies.
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code.
COPY . /app

# Expose port 5000 for the Flask app.
EXPOSE 5000

# Set environment variable for Flask.
ENV FLASK_APP=app.py

# Run the Flask server.
CMD ["flask", "run", "--host=0.0.0.0"]
