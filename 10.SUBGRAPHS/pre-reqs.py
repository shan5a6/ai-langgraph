pip install playwright
playwright install
playwright install-deps
sudo apt-get update
sudo apt-get install -y \
    libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxdamage1 libasound2 \
    libgbm1 libpangocairo-1.0-0 libcups2 libxcomposite1 libxrandr2 libxrender1 \
    libgtk-3-0 libnss3 libdrm2 libxss1 libxtst6

# 1. Install the core newspaper library
pip install newspaper3k

# 2. Patch the dependency error you saw previously
pip install lxml[html_clean]

# 3. Download language model data required by newspaper3k
python3 -m nltk.downloader popular

