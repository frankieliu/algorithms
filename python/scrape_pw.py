from playwright.sync_api import sync_playwright

def get_page_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Or use p.firefox / p.webkit
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')  # Wait for the page to finish loading

        # Get full visible text content
        content = page.inner_text('body')

        browser.close()
        return content

url = "https://example.com"
url = "https://jobs.apple.com/en-us/details/200605190/senior-machine-learning-engineer?team=SFTWR"
text = get_page_text(url)
print(text)  # Preview first 1000 charactersfrom playwright.sync_api import sync_playwright

def get_page_text(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)  # Or use p.firefox / p.webkit
        page = browser.new_page()
        page.goto(url, wait_until='networkidle')  # Wait for the page to finish loading

        # Get full visible text content
        content = page.inner_text('body')

        browser.close()
        return content

url = "https://example.com"
text = get_page_text(url)
print(text[:1000])  # Preview first 1000 charactersu
print(text[:1000])  # Preview first 1000 characters