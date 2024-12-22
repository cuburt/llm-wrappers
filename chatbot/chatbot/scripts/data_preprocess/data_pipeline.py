# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:03:59 2024

@author: somya.upadhyay, cuburt.balanon

@project: XAI

@input:

@output:

@des
"""
from scripts.data_preprocess.get_data import GetData
from scripts.log import logger
from scripts.data_preprocess.url_data_loader import UrlDataLoader


class DataPipeline: # Renamed object according to what it does.
    def __init__(self, config):
        self.get_obj = GetData(config)
        url_config = config["url_config"]
        if url_config["url_boolean"]:
            urls_to_scrape = url_config["url_address"]
            assert urls_to_scrape, "No URLS provided. Disable scraping from config or add URLs to fix this error."
            for url_to_scrape in urls_to_scrape:
                try:
                    if url_to_scrape['is_active']:
                        loader_obj = UrlDataLoader(url_to_scrape["source"],
                                                   url_to_scrape["depth"],
                                                   is_dynamic=url_to_scrape["is_dynamic"],
                                                   allowed_base_urls=url_to_scrape["allowed_base_urls"])
                        all_urls = (loader_obj.recursive_scraper(loader_obj.root_url_to_scrape, 0, loader_obj.depth))
                        logger.info(f"FINAL URL COUNT: {len(all_urls)}")
                        loader_obj.download_and_save_to_csv(all_urls, url_to_scrape["destination"])
                except Exception as e:
                    logger.warning(str(e))
            
    def run(self, data_path):
        text_data = None
        try:
            text_data = self.get_obj(data_path)
            # self.loaded_data.extend(text_data)

        except AssertionError as ae:
            # Log, pass, break.
            logger.error(str(ae))

        except Exception as e:
            logger.warning(str(e))

        if not text_data:
            logger.warning("Loading latest vectorstore...")

        return text_data
