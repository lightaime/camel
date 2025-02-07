# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# @mengkang: Unit Test for video toolkit
import pytest
from dotenv import load_dotenv

from camel.toolkits import VideoAnalysisToolkit

load_dotenv()


@pytest.fixture
def video_toolkit():
    return VideoAnalysisToolkit()


def test_video_1(video_toolkit):
    video_path = "https://www.youtube.com/watch?v=L1vXCYZAYYM"
    question = "what is the highest number of bird species to be on camera \
        simultaneously? Please answer with only the number."

    res = video_toolkit.ask_question_about_video(video_path, question)
    assert res == "3"


def test_video_2(video_toolkit):
    video_path = "https://www.youtube.com/watch?v=1htKBjuUWec"
    question = "What does Teal'c say in response to the question \
    \"Isn't that hot?\" Please answer with the exact words or phrase."

    res = video_toolkit.ask_question_about_video(video_path, question)
    assert "extremely" in res.lower()


def test_video_3(video_toolkit):
    video_path = "https://www.youtube.com/watch?v=2Njmx-UuU3M"
    question = "What species of bird is featured in the video? Please answer\
    with exactly the name of the species without any additional text."

    res = video_toolkit.ask_question_about_video(video_path, question)
    assert res.lower() == "rockhopper penguin"
