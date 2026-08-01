FROM apache/airflow:2.10.5

# Copy both configuration files into the container image build context
COPY requirements.txt /requirements.txt
#COPY constraints.txt /constraints.txt

# Install packages using the local constraint file
RUN pip install --no-cache-dir -r /requirements.txt  #--constraint /constraints.txt
