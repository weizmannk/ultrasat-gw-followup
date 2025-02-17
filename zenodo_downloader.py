# coding: utf-8
"""
---------------------------------------------------------------------------------------------------
                                    ABOUT
@author         : Ramodgwendé Weizmann KIENDREBEOGO
@email          : kiend.weizman7@gmail.com / weizmann.kiendrebeogo@oca.eu
@repo           : https://github.com/weizmannk/ultrasat-gw-followup.git
@createdOn      : December 2023
@description    : A tool for interacting with the Zenodo API, facilitating the download of files based on DOI.
                This class provides functionality to retrieve the latest version DOI associated with a provided
                permanent DOI, and subsequently download the corresponding file from Zenodo.

                You can easily download another dataset from Zenodo by replacing the permanent_doi
                  with a new one.

usage :  python3 zenodo_downloader.py
---------------------------------------------------------------------------------------------------
"""

from tqdm.auto import tqdm
import re
import requests

# If the link to the data is required manually, it can be accessed at : https://doi.org/10.5281/zenodo.14142969

permanent_doi = "14142969"
file_name = "runs_SNR-10.zip"


class ZenodoDownloader:

    """A class to interact with the Zenodo API and download files based on DOI.

    :param permanent_doi: Permanent DOI for a Zenodo record
    :type permanent_doi: str

    :param file_name: Name of the file to be downloaded
    :type file_name: str
    """

    def __init__(self, permanent_doi, file_name):
        self.permanent_doi = permanent_doi
        self.file_name = file_name
        self.headers = {
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Host": "zenodo.org",
            "Connection": "keep-alive",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }
        self.latest_doi = self.get_latest_zenodo_doi()

    def get_latest_zenodo_doi(self):

        """Retrieves the latest version DOI associated with the provided permanent DOI.

        :return: Latest version DOI
        :rtype: str
        """
        r = requests.get(
            f"https://zenodo.org/record/{self.permanent_doi}",
            allow_redirects=True,
            headers=self.headers,
        )
        try:
            data = r.text.split("10.5281/zenodo.")[1]
            doi = re.findall(r"^\d+", data)[0]
        except Exception as e:
            raise ValueError(f"Could not find latest DOI: {str(e)}")
        return doi

    def download_zenodo_data(self):

        """Downloads the file from Zenodo based on the provided DOI and file name.

        :return: None
        """

        try:
            # Fetching the Zenodo record metadata using the DOI
            r = requests.get(
                f"https://zenodo.org/api/records/{self.latest_doi}",
                headers=self.headers,
            )
            r.raise_for_status()

            # Extracting file URL(s) from the Zenodo record metadata
            record_data = r.json()
            files = record_data["files"]

            # Select file to download; default to first file if specific name not found
            file_to_download = next(
                (f for f in files if f["key"] == self.file_name), files[0]
            )

            file_url = file_to_download["links"]["self"]
            response = requests.get(file_url, stream=True)
            response.raise_for_status()  # Raise HTTP errors if any

            file_size = int(response.headers.get("content-length", 0))
            with open(self.file_name, "wb") as file:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, desc="Downloading"
                ) as progress:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)
                        progress.update(len(chunk))

            print(f"File downloaded successfully to {self.file_name}.")
        except requests.RequestException as e:
            print(f"Error: {e}")


zenodo_downloader = ZenodoDownloader(permanent_doi, file_name)
zenodo_downloader.download_zenodo_data()
