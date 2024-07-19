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
import re


class NewWorkforceConf:
    def __init__(self, role: str, system: str, description: str):
        self.role = role
        self.system = system
        self.description = description


def parse_create_wf_resp(response: str) -> NewWorkforceConf:
    r"""Parses the response of the new workforce creation from the manager
    agent."""
    config = re.search(r"(<workforce>.*</workforce>)", response, re.DOTALL)
    if config is None:
        raise ValueError("No workforce configuration found in the response.")
    config_raw = config.group(1)

    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(config_raw)
        workforce_info = {child.tag: child.text for child in root}
    except Exception as e:
        raise ValueError(f"Failed to parse workforce configuration: {e}")

    if (
        "role" not in workforce_info
        or "system" not in workforce_info
        or "description" not in workforce_info
    ):
        raise ValueError("Missing required fields in workforce configuration.")

    return NewWorkforceConf(
        role=workforce_info["role"] or "",
        system=workforce_info["system"] or "",
        description=workforce_info["description"] or "",
    )


def parse_assign_task_resp(response: str) -> str:
    r"""Parses the response of the task assignment from the manager agent."""
    assignee_id = re.search(r"<id>(.*)</id>", response)
    if assignee_id is None:
        raise ValueError("No assignee found in the response.")
    return assignee_id.group(1)
