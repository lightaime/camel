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

from main import O1DataGene, chat_agent

def main():
    # 定义你的问题和标准答案
    custom_golden_answers = {
        "A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.": "\\frac{1}{4}",
        
    }

    # 创建 O1DataGene实例
    solver = O1DataGene(chat_agent, custom_golden_answers)

    # 解决问题
    for question in custom_golden_answers.keys():
        print(f"\n正在解决问题: {question}")
        answer = solver.solve(question)
        print(f"最终答案: {answer}")
    
    # 导出结果到JSON文件
    solver.export_solutions('math_solutions.json')

if __name__ == "__main__":
    main()
