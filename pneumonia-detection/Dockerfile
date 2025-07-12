# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Make port 8501 available to the world outside this container (for Streamlit)
EXPOSE 8501

# Make port 8000 available (for FastAPI, if used)
EXPOSE 8000

# Define environment variable
ENV NAME World

# Run app.py when the container launches
# The command is split into a list to avoid shell interpretation issues.
# Use this command to run the Streamlit app:
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# To run the FastAPI service instead, you would build the image with a different CMD
# or override it at runtime. For example:
# CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
