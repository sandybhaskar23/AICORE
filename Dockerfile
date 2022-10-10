# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9.12

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
#["wget", "-q","-O","-","https://dl-ssl.google.com/linux/linux_signing_key.pub", "|" , "apt-key", "add","-"]
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'  
#["sh","-c","\'echo", " \'deb \[arch=amd64\] http://dl.google.com/linux/chrome/deb/ stable main\'", ">>"," /etc/apt/sources.list.d/google-chrome.list\'"]
RUN apt-get -y update
#["apt-get","-y","update"]
RUN apt-get install -y google-chrome-stable

# ["apt-get","install","-y","google-chrome-stable"]
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip
#["wget","-O","/tmp/chromedriver.zip", "http://chromedriver.storage.googleapis.com/", "`curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE`/chromedriver_linux64.zip"]
RUN apt-get install -yqq unzip
#["apt-get","install","-yqq","unzip"]
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/
#["unzip","/tmp/chromedriver.zip" , "chromedriver", "-d", "/usr/local/bin/"]



# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

WORKDIR /app
COPY . /app

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser



# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "DataCollection/MyCancerGenome/mycancergenome_scraper.py"]
