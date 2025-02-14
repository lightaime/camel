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
import os
from typing import Any, List, Optional, Union

from openai import AsyncOpenAI, OpenAI, _legacy_response

from camel.types import AudioModelType, VoiceType


class OpenAIAudioModels:
    r"""Provides access to OpenAI's Text-to-Speech (TTS) and Speech_to_Text
    (STT) models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        r"""Initialize an instance of OpenAI."""
        self._url = url or os.environ.get("OPENAI_API_BASE_URL")
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(
            timeout=120,
            max_retries=3,
            base_url=self._url,
            api_key=self._api_key,
        )
        self._async_client = AsyncOpenAI(
            timeout=120,
            max_retries=3,
            base_url=self._url,
            api_key=self._api_key,
        )

    def text_to_speech(
        self,
        input: str,
        model_type: AudioModelType = AudioModelType.TTS_1,
        voice: VoiceType = VoiceType.ALLOY,
        storage_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[
        List[_legacy_response.HttpxBinaryResponseContent],
        _legacy_response.HttpxBinaryResponseContent,
    ]:
        r"""Convert text to speech using OpenAI's TTS model. This method
        converts the given input text to speech using the specified model and
        voice.

        Args:
            input (str): The text to be converted to speech.
            model_type (AudioModelType, optional): The TTS model to use.
                Defaults to `AudioModelType.TTS_1`.
            voice (VoiceType, optional): The voice to be used for generating
                speech. Defaults to `VoiceType.ALLOY`.
            storage_path (str, optional): The local path to store the
                generated speech file if provided, defaults to `None`.
            **kwargs (Any): Extra kwargs passed to the TTS API.

        Returns:
            Union[List[_legacy_response.HttpxBinaryResponseContent],
                _legacy_response.HttpxBinaryResponseContent]: List of response
                content object from OpenAI if input charaters more than 4096,
                single response content if input charaters less than 4096.

        Raises:
            Exception: If there's an error during the TTS API call.
        """
        try:
            # Model only support at most 4096 characters one time.
            max_chunk_size = 4095
            audio_chunks = []
            chunk_index = 0
            if len(input) > max_chunk_size:
                while input:
                    if len(input) <= max_chunk_size:
                        chunk = input
                        input = ''
                    else:
                        # Find the nearest period before the chunk size limit
                        while input[max_chunk_size - 1] != '.':
                            max_chunk_size -= 1

                        chunk = input[:max_chunk_size]
                        input = input[max_chunk_size:].lstrip()

                    response = self._client.audio.speech.create(
                        model=model_type.value,
                        voice=voice.value,
                        input=chunk,
                        **kwargs,
                    )
                    if storage_path:
                        try:
                            # Create a new storage path for each chunk
                            file_name, file_extension = os.path.splitext(
                                storage_path
                            )
                            new_storage_path = (
                                f"{file_name}_{chunk_index}{file_extension}"
                            )
                            response.write_to_file(new_storage_path)
                            chunk_index += 1
                        except Exception as e:
                            raise Exception(
                                "Error during writing the file"
                            ) from e

                    audio_chunks.append(response)
                return audio_chunks

            else:
                response = self._client.audio.speech.create(
                    model=model_type.value,
                    voice=voice.value,
                    input=input,
                    **kwargs,
                )

            if storage_path:
                try:
                    response.write_to_file(storage_path)
                except Exception as e:
                    raise Exception("Error during write the file") from e

            return response

        except Exception as e:
            raise Exception("Error during TTS API call") from e

    def _split_audio(
        self, audio_file_path: str, chunk_size_mb: int = 24
    ) -> list:
        r"""Split the audio file into smaller chunks. Since the Whisper API
        only supports files that are less than 25 MB.

        Args:
            audio_file_path (str): Path to the input audio file.
            chunk_size_mb (int, optional): Size of each chunk in megabytes.
                Defaults to `24`.

        Returns:
            list: List of paths to the split audio files.
        """
        from pydub import AudioSegment

        audio = AudioSegment.from_file(audio_file_path)
        audio_format = os.path.splitext(audio_file_path)[1][1:].lower()

        # Calculate chunk size in bytes
        chunk_size_bytes = chunk_size_mb * 1024 * 1024

        # Number of chunks needed
        num_chunks = os.path.getsize(audio_file_path) // chunk_size_bytes + 1

        # Create a directory to store the chunks
        output_dir = os.path.splitext(audio_file_path)[0] + "_chunks"
        os.makedirs(output_dir, exist_ok=True)

        # Get audio chunk len in milliseconds
        chunk_size_milliseconds = len(audio) // (num_chunks)

        # Split the audio into chunks
        split_files = []
        for i in range(num_chunks):
            start = i * chunk_size_milliseconds
            end = (i + 1) * chunk_size_milliseconds
            if i + 1 == num_chunks:
                chunk = audio[start:]
            else:
                chunk = audio[start:end]
            # Create new chunk path
            chunk_path = os.path.join(output_dir, f"chunk_{i}.{audio_format}")
            chunk.export(chunk_path, format=audio_format)
            split_files.append(chunk_path)
        return split_files

    def speech_to_text(
        self,
        audio_file_path: str,
        translate_into_english: bool = False,
        **kwargs: Any,
    ) -> str:
        r"""Convert speech audio to text.

        Args:
            audio_file_path (str): The audio file path, supporting one of
                these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or
                webm.
            translate_into_english (bool, optional): Whether to translate the
                speech into English. Defaults to `False`.
            **kwargs (Any): Extra keyword arguments passed to the
                Speech-to-Text (STT) API.

        Returns:
            str: The output text.

        Raises:
            ValueError: If the audio file format is not supported.
            Exception: If there's an error during the STT API call.
        """
        supported_formats = [
            "flac",
            "mp3",
            "mp4",
            "mpeg",
            "mpga",
            "m4a",
            "ogg",
            "wav",
            "webm",
        ]
        file_format = audio_file_path.split(".")[-1].lower()

        if file_format not in supported_formats:
            raise ValueError(f"Unsupported audio file format: {file_format}")
        try:
            if os.path.getsize(audio_file_path) > 24 * 1024 * 1024:
                # Split audio into chunks
                audio_chunks = self._split_audio(audio_file_path)
                texts = []
                for chunk_path in audio_chunks:
                    audio_data = open(chunk_path, "rb")
                    if translate_into_english:
                        translation = self._client.audio.translations.create(
                            model="whisper-1", file=audio_data, **kwargs
                        )
                        texts.append(translation.text)
                    else:
                        transcription = (
                            self._client.audio.transcriptions.create(
                                model="whisper-1", file=audio_data, **kwargs
                            )
                        )
                        texts.append(transcription.text)
                    os.remove(chunk_path)  # Delete temporary chunk file
                return " ".join(texts)
            else:
                # Process the entire audio file
                audio_data = open(audio_file_path, "rb")

                if translate_into_english:
                    translation = self._client.audio.translations.create(
                        model="whisper-1", file=audio_data, **kwargs
                    )
                    return translation.text
                else:
                    transcription = self._client.audio.transcriptions.create(
                        model="whisper-1", file=audio_data, **kwargs
                    )
                    return transcription.text
        except Exception as e:
            raise Exception("Error during STT API call") from e
