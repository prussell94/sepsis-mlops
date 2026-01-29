# 1. Use a slim Python base to keep the container small
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. COPY only the requirements file first (Crucial for Caching)
COPY requirements.txt .

# 4. INSTALL the dependencies
# --no-cache-dir keeps the image lean by not saving the installer files
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY your actual script (Only happens after libraries are installed)
# COPY preprocess.py .
COPY . .

# 6. Define the command to run when the container starts
ENTRYPOINT ["python", "preprocess.py"]