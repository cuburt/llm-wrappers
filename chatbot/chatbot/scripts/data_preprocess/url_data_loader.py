# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: somya.upadhyay, cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
import os
import shutil
from bs4 import BeautifulSoup 
from selenium import webdriver 
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import requests
from scripts.log import logger
from pathlib import Path
import platform
import tempfile
import csv
from langchain_community.document_loaders.unstructured import UnstructuredBaseLoader
from typing import List, Any
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse
import nltk
from collections import Counter
from urllib.parse import urljoin


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


def most_common_base_url(urls):
    most_common_base_url = None
    # Parse the URLs to extract the base URLs
    base_urls = [f"{urlparse(url).scheme}://{urlparse(url).netloc}" for url in urls if urlparse(url).scheme and urlparse(url).netloc]

    # Count the occurrences of each base URL
    base_url_counts = Counter(base_urls)

    # Find the most common base URL
    if base_url_counts.most_common(1):
        most_common_base_url, _ = base_url_counts.most_common(1)[0]

    return most_common_base_url


class UnstructuredHtmlStringLoader(UnstructuredBaseLoader):
    '''
    Uses unstructured to load a string
    Source of the string, for metadata purposes, can be passed in by the caller
    '''
    def __init__(
        self, content: str, source: str = None, mode: str = "single",
        **unstructured_kwargs: Any
    ):
        self.content = content
        self.source = source
        super().__init__(mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:
        from unstructured.partition.html import partition_html

        return partition_html(text=self.content, **self.unstructured_kwargs)

    def _get_metadata(self) -> dict:
        return {"source": self.source} if self.source else {}


class UrlDataLoader:
    def __init__(self, root_url_to_scrape, depth, is_dynamic, allowed_base_urls):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('averaged_perceptron_tagger_eng')
        self.root_url_to_scrape = root_url_to_scrape
        self.base_url = f"{urlparse(root_url_to_scrape).scheme}://{urlparse(root_url_to_scrape).netloc}"
        self.depth = depth
        self.is_dynamic = is_dynamic
        allowed_base_urls.append(self.base_url)
        self.allowed_base_urls = allowed_base_urls
        self.all_urls = set()

    def remove_suffix_by_separator(self,text, separator,min):
        """
        Removes the suffix from a string based on a specified separator.
        Args:
            text (str): The input string.
            separator (str): The separator to split the string.
        Returns:
            str: The modified string with the suffix removed.
        """
        parts = text.split(separator)
        if len(parts) > min:
            return separator.join(parts[:-1])
        else:
            return text
        
    def scrape_urls_from_html(self, html_content, parent_url):
        response = requests.get(parent_url)
        response.raise_for_status()
        soup = BeautifulSoup(html_content, "html5lib")
        # urls = []
        # Find all anchor tags (a) which contain href attribute
        anchor_tags = soup.find_all('a', href=True)
        urls = [urljoin(parent_url, tag['href']) for tag in anchor_tags]
        # for tag in anchor_tags:
        #     parent_url_t = parent_url
        #     logger.info(f"Scraping: {tag['href']} for urls")
        #     # Extract URLs from anchor tags
        #     if 'href' in tag and tag['href'].startswith("?"):
        #         if '?' in parent_url_t:
        #             parent_url_t = self.remove_suffix_by_separator(parent_url_t, '?', 1)
        #             url = self.remove_suffix_by_separator(parent_url_t, '?', 1) + tag['href']
        #         else:
        #             url = parent_url_t + ['href']
        #         logger.info(url)
        #         urls.append(url)
        #     elif 'href' in tag and tag['href'].startswith("#"):
        #         url = parent_url_t + tag['href']
        #         logger.info(url)
        #         urls.append(url)
        #     elif 'href' in tag and not tag['href'].startswith("https://") and not tag['href'].startswith('#'):
        #         while tag['href'].startswith("../"):
        #             tag['href'] = tag['href'][3:]
        #             parent_url_t = self.remove_suffix_by_separator(parent_url_t, '/', 3)
        #         url = self.remove_suffix_by_separator(parent_url_t, '/', 3)+'/' + tag['href']
        #         logger.info(url)
        #         urls.append(url)
        #     else:
        #         logger.info(tag['href'])
        #         urls.append(tag['href'])

        return urls

    def scrape_urls_from_xml(self, html_content, parent_url):
        response = requests.get(parent_url)
        response.raise_for_status()
        logger.info("Reverting BS4 constructor to XML...")
        soup = BeautifulSoup(html_content, features="xml")

        # Find all anchor tags (a) which contain href attribute
        anchor_tags = soup.find_all('loc')
        urls = [tag.text for tag in anchor_tags if tag.text]

        return urls

    @staticmethod
    def get_html_content_js(url, skip_waiting_for_text=False):
        try:
            chrome_options = Options()
            # chrome_options.add_argument("--ignore-certificate-errors")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--headless")
            # chrome_options.add_argument("--remote-debugging-port=9222")
            # chrome_options.add_argument("--disable-gpu")
            # chrome_options.add_argument("--window-size=1920,1080")
            if platform.system() == 'Windows':
                driver = webdriver.Chrome(options=chrome_options)
                # driver.implicitly_wait(60)
            else:
                driver = webdriver.Chrome(options=chrome_options,
                                          service=Service("/usr/bin/chromedriver-linux64/chromedriver"))

            driver.get(url)

            if not skip_waiting_for_text:
                wait = WebDriverWait(driver, 15, 1)
                element = wait.until(EC.presence_of_element_located((By.XPATH, "//body//p")))

            else:
                wait = WebDriverWait(driver, 10)
                element = wait.until(EC.presence_of_element_located((By.XPATH, "html")))

            page_source, text = driver.page_source, element.text if element else None
            driver.quit()

            return page_source, text
        except Exception as e:
            raise Exception(str(e))

    @staticmethod
    def get_html_content(url, bool):
        try:
            # Send a GET request to fetch the HTML content
            response = requests.get(url)
            # Raise an exception for HTTP errors
            response.raise_for_status()
            return response.text, response.text
            
        except Exception as e:
            logger.error(e)
            return None

    def download_and_save_to_csv(self, url_list, csv_path):
        PROJECT_ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)
        if platform.system() == 'Windows':
            csv_path = csv_path.replace('/', '\\')
        csv_path = os.path.join(PROJECT_ROOT_DIR, csv_path, "data.csv")
        temp_dir = tempfile.gettempdir()

        # Create the directory for the CSV file if it doesn't exist
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))
            logger.info(f"data.csv created at {csv_path}")

        temp_csv_path = os.path.join(temp_dir, os.path.basename(csv_path))

        with open(temp_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['page_content', 'source'])  # Write the header
            rows = []
            for url in url_list:
                try:
                    # # Send a GET request to fetch the HTML content
                    # response = requests.get(url)
                    #
                    # # Raise an exception for HTTP errors
                    # response.raise_for_status()

                    if self.is_dynamic:
                        page_content, _ = self.get_html_content_js(url, True)
                    else:
                        page_content, _ = self.get_html_content(url, True)

                    loader = UnstructuredHtmlStringLoader(content=page_content, source=url)
                    html = loader.load()
                    assert len(html) == 1, "Sorry, multiple HTMLs cannot be extracted from the same URL."
                    assert html[0].page_content or html[0].page_content != "", "No page_content scraped from html."
                    # Write the HTML content and the source URL to the CSV file
                    rows.append([html[0].page_content.strip(), html[0].metadata['source']])
                    logger.info(f"HTML content from {url} downloaded and saved to CSV")

                except Exception as e:
                    logger.error(f"Error downloading HTML content from {url}: {e}")
            csv_writer.writerows(rows)
        shutil.move(temp_csv_path, csv_path)
        logger.info(f"CSV stored at: {csv_path}\n FileSize: {convert_bytes(Path(csv_path).stat().st_size)}")


    def scraper(self, url_to_scrape):
        scraped_urls=[]
        # Fetch HTML content of the page
        try:
            # old_base_url=self.root_url_to_scrape
            # base_url=self.remove_suffix_by_separator(old_base_url,"/",4)
            # while base_url!= old_base_url:
            #     old_base_url= base_url
            #     base_url=self.remove_suffix_by_separator(old_base_url,"/", 4)
            if url_to_scrape.startswith(self.base_url) and 'sitemap.xml' not in url_to_scrape:

                if self.is_dynamic:
                    html_content, _ = self.get_html_content_js(url_to_scrape, True)
                else:
                    html_content, _ = self.get_html_content(url_to_scrape, True)

                if html_content:
                    # Scrape URLs from the HTML content
                    scraped_urls = self.scrape_urls_from_html(html_content, url_to_scrape)

            elif 'sitemap.xml' in url_to_scrape:

                html_content, _ = self.get_html_content(url_to_scrape, True)
                if html_content:
                    # Scrape URLs from the HTML content
                    scraped_urls = self.scrape_urls_from_xml(html_content, url_to_scrape)

        except Exception as e:
            logger.error(str(e))         
        return scraped_urls


    def recursive_scraper(self, url, depth, max_depth, visited_urls=None):
        logger.info(f"URL: {url}")
        if visited_urls is None:
            visited_urls = set()

        if depth < max_depth:
            new_base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}" if urlparse(url).scheme and urlparse(url).netloc else self.base_url
            logger.info(f"BASE URL: {new_base_url}")
            if new_base_url and new_base_url != self.base_url and new_base_url in self.allowed_base_urls:
                logger.info(f"CHANGING BASE URL TO: {new_base_url}")
                self.base_url = new_base_url
            urls = self.scraper(url)
            self.all_urls.update(urls)
            logger.info(f"URL COUNT: {len(self.all_urls)}")
            for u in urls:
                if u not in visited_urls:
                    visited_urls.add(u)
                    self.recursive_scraper(u, depth + 1, max_depth, visited_urls)
        
        if depth == max_depth or not self.all_urls:
            # Check if we've reached max depth or there are no more URLs to scrape
            urls = self.scraper(url)
            self.all_urls.update(urls)
            logger.info(f"URL COUNT: {len(self.all_urls)}")

        return self.all_urls
