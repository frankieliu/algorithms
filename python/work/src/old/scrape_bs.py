import requests
from bs4 import BeautifulSoup
import re
from collections import defaultdict

jobs = [200606785]
headings = [("pay", "Pay \& Benefits"),
            ("pref", "Preferred Qualifications"),
            ("min", "Minimum Qualifications"),
            ("desc", "Description"),
            ("sum", "Summary"),
            ("title", "Search"),
            ("end", "Apple employees")]
regex = "|".join(f"(?P<{name}>{value})" for name,value in headings)

for job in jobs:
    url = f"https://jobs.apple.com/en-us/details/{job}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text content
    text = soup.get_text(separator='\n', strip=True)

    search = False

    # Optionally trim or clean up the text
    col = {}
    prev = None
    saved = []
    for line in text.splitlines():
        line = line.strip()
        # looking for headings
        match_ = re.match(regex, line)
        if match_:
            # don't save prev
            if prev:
                col[prev] = saved
                saved = []
            prev = match_.lastgroup
        saved.append(line)
    if len(saved) > 0:
        col[prev] = saved
    print(col)

