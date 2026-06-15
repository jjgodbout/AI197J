# Use the official Python 3.10 image as a base image
FROM nvcr.io/nvidia/pytorch:23.06-py3

# Set the working directory in the container
WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Install Jupyter
RUN pip install jupyter


# Copy the entire current directory contents into the container at /app
COPY . /app
# Expose the Streamlit default port
EXPOSE 8501 8888

# Use a script or process manager to run multiple commands
# This is a simple example; for production, consider using a process manager
CMD streamlit run app.py -- server .host=0.0.0.0. & jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''

