FROM aukesch/speechmatching
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip --no-cache-dir install -r requirements.txt
COPY . /app
CMD ["python3", "main.py"]

