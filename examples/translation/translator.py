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
import argparse
import codecs
import json
import multiprocessing
import os
import os.path as osp
import warnings

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.generators import SystemMessageGenerator
from camel.models import ModelFactory
from camel.types import (
    ModelPlatformType,
    ModelType,
    RoleType,
    TaskType,
)

warnings.filterwarnings("ignore")

language_list = [
    "arabic",
    "chinese",
    "french",
    "german",
    "hindi",
    "italian",
    "japanese",
    "korean",
    "russian",
    "spanish",
]

parser = argparse.ArgumentParser(description='Arguments for translation.')
parser.add_argument(
    '--directory_path',
    type=str,
    help='Directory that contains original json files',
    default='../camel_data/ai_society',
)
parser.add_argument(
    '--save_directory_path',
    type=str,
    help='Directory to save translated files',
    default='../camel_data/ai_society_translated',
)
parser.add_argument(
    '--single',
    action='store_true',
    help='Run translator in a non-parallel way.',
)
parser.add_argument(
    '--stream',
    action='store_true',
    help='Set OpenAI GPT model with the stream mode.',
)
parser.add_argument(
    '--language',
    type=str,
    help='Language you want to translated to. '
    'Notice that this is not used in the parallel mode, '
    'which uses SLURM_ARRAY_TASK_ID to indicate the '
    'language to be translated.',
    choices=language_list,
    default='arabic',
)


def translate_content(
    args: argparse.Namespace, file_path: str, language: str
) -> None:
    # Extract file name from the .json file path to be translated
    file_name = osp.splitext(osp.basename(file_path))[0]

    if not osp.exists(args.save_directory_path):
        os.makedirs(args.save_directory_path)

    save_lang_director_path = osp.join(args.save_directory_path, language)
    if not osp.exists(save_lang_director_path):
        os.makedirs(save_lang_director_path)

    # Check that file_name.json does not exist in the save directory
    save_path = osp.join(save_lang_director_path, f'{file_name}.json')
    if osp.exists(save_path):
        return

    # Load the json file
    with open(file_path, "r") as json_file:
        json_data = json.load(json_file)

    # Translate the content of each message in the json
    for i in range(json_data['num_messages']):
        msg_i_content = (
            "Sentence to translate: " + json_data[f"message_{i+1}"]["content"]
        )

        sys_msg_generator = SystemMessageGenerator(
            task_type=TaskType.TRANSLATION
        )

        assistant_sys_msg = sys_msg_generator.from_dict(
            meta_dict=dict(language=language.capitalize()),
            role_tuple=('Language Translator', RoleType.ASSISTANT),
        )

        if not args.stream:
            model_config = ChatGPTConfig(stream=False)
        else:
            model_config = ChatGPTConfig(stream=True)

        model = ModelFactory.create(
            model_platform=ModelPlatformType.DEFAULT,
            model_type=ModelType.DEFAULT,
            model_config=model_config,
        )

        assistant_agent = ChatAgent(
            system_message=assistant_sys_msg,
            model=model,
        )

        user_msg = msg_i_content

        assistant_response = assistant_agent.step(user_msg)
        assistant_msg = assistant_response.msg

        json_data[f"message_{i+1}"]["content"] = assistant_msg.content.text

    with codecs.open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def main(args: argparse.Namespace) -> None:
    if not args.single:
        # Get the language to translate based on Slurm array index
        slum_id_env = "SLURM_ARRAY_TASK_ID"
        try:
            language_index = int(os.environ[slum_id_env])
        except KeyError:
            print(f"{slum_id_env} not found. Defaulting to 0 (i.e Arabic)")
            # Default to Arabic, you can change to any other language
            language_index = 0
        # List of languages to translate to
        language_list = [
            "arabic",
            "chinese",
            "french",
            "german",
            "hindi",
            "italian",
            "japanese",
            "korean",
            "russian",
            "spanish",
        ]
        language = language_list[language_index]
    else:
        language = args.language

    # Get list of all .json files paths
    json_file_paths = []

    for filename in os.listdir(args.directory_path):
        if filename.endswith(".json"):
            file_path = osp.join(args.directory_path, filename)
            json_file_paths.append(file_path)

    if not args.single:
        pool = multiprocessing.Pool()
        # Apply parallel translation to all .json files
        for file_path in json_file_paths:
            pool.apply_async(
                translate_content, args=(args, file_path, language)
            )
        pool.close()
        pool.join()
    else:
        for file_path in json_file_paths:
            translate_content(args, file_path, language)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args=args)
