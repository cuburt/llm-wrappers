import csv
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader

# Function to scrape code snippets from a given URL and store them in a CSV file
def scrape_and_store_code_snippets(url, csv_writer):
    # Hypothetical function to load the page content using RecursiveUrlLoader
    page_content = RecursiveUrlLoader.load(url)
    # Check if there are code snippets
    code_snippets = page_content.find_all('code')
    if code_snippets:
        for snippet in code_snippets:
            # Extract text from code snippet
            snippet_text = snippet.get_text()
            csv_writer.writerow([snippet_text])


# Recursive function to crawl the given URL and its subpages
def crawl_and_store(url, csv_filename):
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # Scrape code snippets from the main page
        scrape_and_store_code_snippets(url, writer)
        # Hypothetical function to load the page content using RecursiveUrlLoader
        page_content = RecursiveUrlLoader.load(url)
        # Find links to subpages
        subpage_links = page_content.find_all('a', href=True)
        for link in subpage_links:
            subpage_url = link['href']
            # Check if the link is to a subpage
            if subpage_url.startswith(url):
                # Recursively crawl subpages
                crawl_and_store(subpage_url, csv_filename)


# Main function
def main():
    url = "https://help.hcltechsw.com/dom_designer/11.0.1/basic/H_DUMMY_TOPIC_HEAD.html"
    csv_filename = "code_snippets.csv"
    crawl_and_store(url, csv_filename)


if __name__ == "__main__":
    main()