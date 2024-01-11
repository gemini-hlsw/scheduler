# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
#
# Adaptation from the following repository, distributed under the MIT license:
# https://github.com/ndrplz/google-drive-downloader
#
# MIT License
#
# Copyright (c) 2017 Andrea Palazzi (modified by AURA)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import os.path
import requests
from abc import ABC
from typing import Final


# This is just a collection of static methods grouped together, so make it abstract.
class GoogleDriveDownloader(ABC):
    """
    Minimal class to download shared files from Google Drive.
    """
    _timeout: Final[float] = 9.0

    _CHUNK_SIZE: Final[int] = 32768
    _DOWNLOAD_URL: Final[str] = 'https://docs.google.com/uc?export=download'

    @staticmethod
    def download_file(file_id, dest_path, overwrite=False):
        """
        Downloads a shared file from Google Drive into a given folder.
        Optionally unzips it.

        Parameters
        ----------
        file_id: str
            the file identifier.
            You can obtain it from the sharable link.
        dest_path: str
            the destination where to save the downloaded file.
            Must be a path (for example: './downloaded_file.txt')
        overwrite: bool
            optional, if True forces re-download and overwrite.
        Returns
        -------
        None
        """

        destination_directory = os.path.dirname(dest_path)
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        if not os.path.exists(dest_path) or overwrite:
            session = requests.Session()
            params = {'id': file_id}
            response = session.get(GoogleDriveDownloader._DOWNLOAD_URL,
                                   params=params,
                                   stream=True,
                                   timeout=GoogleDriveDownloader._timeout)
            GoogleDriveDownloader._save_response_content(response, dest_path)

    @staticmethod
    def _save_response_content(response, destination):
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(GoogleDriveDownloader._CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
