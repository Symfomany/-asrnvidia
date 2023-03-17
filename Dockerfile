# Use the official Python 3.9 image
# FROM nvidia/cuda:11.4.0-base-ubuntu20.04
FROM python:3.9


RUN apt update
RUN apt-get install -y python3 python3-pip


# Set the working directory to /code
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./requirements.txt /code/requirements.txt

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


# Install requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME

CMD ["uvicorn", "model:app", "--host", "0.0.0.0", "--port", "8000"]