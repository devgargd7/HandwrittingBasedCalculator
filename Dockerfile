FROM python:3.10.2
RUN mkdir /main
WORKDIR /main
ADD app .
RUN python -m pip install --upgrade pip
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install -r Requirements.txt
CMD ["python","app.py"]