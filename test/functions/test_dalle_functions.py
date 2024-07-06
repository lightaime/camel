# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
import base64
import os

import pytest
from PIL import Image

from camel.functions.dalle_functions import (
    base64_to_image,
    image_path_to_base64,
    image_to_base64,
)


@pytest.fixture(scope="function")
def test_image_path():
    # Prepare the path for the test image
    test_image_path = 'test_image.png'
    # Create a test image file (or ensure it exists for the test)
    with open(test_image_path, 'wb') as f:
        f.write(b'This is a test image placeholder content.')

    # Provide the path to the temp file for the test to use
    yield test_image_path

    # Teardown: Clean up the temporary file
    try:
        os.remove(test_image_path)
    except OSError as e:
        print(f"Error: {e.strerror}")


def test_base64_to_image_valid():
    valid_base64_string = "iVBORw0KGgoAAAANSUhEUgAAAAUA\
                            AAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO\
                            9TXL0Y4OHwAAAABJRU5ErkJggg=="
    if valid_base64_string.startswith('data:image/png;base64,'):
        valid_base64_string = valid_base64_string[22:]

    image = base64_to_image(valid_base64_string)

    assert isinstance(
        image, Image.Image
    ), "The function should return a PIL Image object"


def test_base64_to_image_invalid():
    invalid_base64_string = "invalid_base64_string"

    image = base64_to_image(invalid_base64_string)

    # Check response is None
    assert (
        image is None
    ), "The function should return None for an invalid base64 string"


def test_image_path_to_base64(test_image_path):
    # Obtain the Base64 encoded string of the test image
    base64_encoded_string = image_path_to_base64(test_image_path)
    # Decode the Base64 string back to binary data for validation
    decoded_binary_data = base64.b64decode(base64_encoded_string)
    # Verify if the decoded data matches the original binary content of
    # the test image
    with open(test_image_path, 'rb') as f:
        original_binary_data = f.read()

    assert (
        decoded_binary_data == original_binary_data
    ), "The Base64 encoded string does not match the original image content."


def test_image_to_base64_with_invalid_input():
    # Intentionally pass a non-image object to simulate an error condition
    invalid_input = "This is not an image"

    # Expect the function to return None when an error occurs during encoding
    result = image_to_base64(invalid_input)

    # Assert that the function correctly returns None for invalid inputs
    assert result is None
