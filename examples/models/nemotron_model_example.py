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

from camel.models import NemotronModel
from camel.types import ModelType

nemotron = NemotronModel(model_type=ModelType.NVIDIA_NEMOTRON_340B_REWARD)

message = [
    {"role": "user", "content": "I am going to Paris, what should I see?"},
    {
        "role": "assistant",
        "content": "Ah, Paris, the City of Light! There are so "
        "many amazing things to see and do in this beautiful city ...",
    },
]

ans = nemotron._run(message)
print(ans)
'''
===============================================================================
ChatCompletion(id='4668ad22-1dec-4df4-ba92-97ffa5fbd16d', choices=[Choice
(finish_reason='length', index=0, logprobs=ChoiceLogprobs(content=
[ChatCompletionTokenLogprob(token='helpfulness', bytes=None, logprob=1.
6171875, top_logprobs=[]), ChatCompletionTokenLogprob(token='correctness', 
bytes=None, logprob=1.6484375, top_logprobs=[]), ChatCompletionTokenLogprob
(token='coherence', bytes=None, logprob=3.3125, top_logprobs=[]), 
ChatCompletionTokenLogprob(token='complexity', bytes=None, logprob=0.546875, 
top_logprobs=[]), ChatCompletionTokenLogprob(token='verbosity', bytes=None, 
logprob=0.515625, top_logprobs=[])]), message=[ChatCompletionMessage
(content='helpfulness:1.6171875,correctness:1.6484375,coherence:3.3125,
complexity:0.546875,verbosity:0.515625', role='assistant', function_call=None, 
tool_calls=None)])], created=None, model=None, object=None, 
system_fingerprint=None, usage=CompletionUsage(completion_tokens=1, 
prompt_tokens=78, total_tokens=79))
===============================================================================
'''
