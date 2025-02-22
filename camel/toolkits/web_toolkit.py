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

import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image

from camel.agents import ChatAgent
from camel.configs import ChatGPTConfig
from camel.interpreters.subprocess_interpreter import SubprocessInterpreter
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.prompts import TextPrompt
from camel.toolkits.function_tool import FunctionTool
from camel.types import ModelPlatformType, ModelType, RoleType

# Define a module-level constant for the default ChatGPT configuration
_DEFAULT_CHATGPT_CONFIG_DICT = ChatGPTConfig(temperature=0.0).as_dict()


class TreeNode:
    def __init__(
        self, url: str, actions: List[str], action_taken: str, result: str
    ):
        r"""
        Initializes a tree node.

        Args:
            url (str): The URL or identifier of the node.
            actions (List[str]): Possible actions at this node.
            action_taken (str): The action taken at this node.
            result (str): The resulting state after action.

        Returns:
            None
        """
        self.url = url
        self.actions = actions
        self.action_taken = action_taken
        self.result = result
        self.next: List["TreeNode"] = []  # List to hold child nodes

    def add_child(self, node: "TreeNode") -> None:
        r"""
        Adds a child node to the current node.

        Args:
            node (TreeNode): The child node to add.

        Returns:
            None
        """
        self.next.append(node)

    def find_(self, url: str) -> Optional["TreeNode"]:
        r"""
        Finds a node with the given URL using Depth-First Search (DFS).

        Args:
            url (str): The URL to search for.

        Returns:
            TreeNode or None: The found node if it exists, otherwise None.
        """
        if self.url == url:
            return self

        for child in self.next:
            found_node = child.find_(url)
            if found_node:
                return found_node
        return None

    def add_node_at(self, parent_url: str, new_node: "TreeNode") -> bool:
        r"""
        Adds a new node as a child to the specified parent node.

        Args:
            parent_url (str): The URL of the parent node
                            where the new node should be added.
            new_node (TreeNode): The new node to add.

        Returns:
            bool: True if the node was successfully added,
                  False otherwise.
        """
        parent_node = self.find_(parent_url)
        if parent_node:
            parent_node.add_child(new_node)
            return True
        return False

    def __repr__(self) -> str:
        r"""
        Provides a string representation of the node,
        displaying its attributes.

        Returns:
            str: A formatted string representation of the node.
        """
        return (
            f"URL: {self.url}\n"
            f"Available actions: {self.actions}\n"
            f"Action taken: {self.action_taken}\n"
            f"Result: {self.result}\n"
            f"Children states: {[child.url for child in self.next]}\n"
        )


class StagehandPrompts:
    """
    A centralized class for Stagehand-related prompts,
    leveraging TextPrompt for better modularity and reuse.
    """

    def __init__(self, high_level_task: str, action_mode: str):
        self.high_level_task = high_level_task

        # switch case:
        if action_mode == "get_action":
            self.action_prompt = """
                - You call page.observe() to find all available 
                  actions on the given url.
                - At the end, output JSON in this structure:
                {{
                "status": "success",
                "updated_state": {{
                    "actions": List(str),
                    "code": "The code snippet that performed this extraction",
                    "link": "The absolute URL or final link"
                }}
                }}
                
                - Only visit the url specified by the user.
                - All the actions must be described in string format. 
                  You should not append in [Object] types to the action. 
                  Write code to describe each [Object] in actions list into a 
                  string of its description.
                - You only perform an observe() (or similar) call to list 
                  possible actions on the page.
                - actions should be a list of actions on the 
                  webpage as a list of strings.
                - Always output the final JSON with the correct keys:
                - For get_actions: updated_state.actions, updated_state.code, 
                  updated_state.link
                """
        else:
            self.action_prompt = """
                VERY IMPORTANT:
                always add this:
                - At the end, output JSON in this structure:
                
                const updated_state = {{
                status: "success",
                updated_state: {{
                    action_taken: actionTaken,  // Stores the 
                    // chosen action as a string
                    code: actionCode,  // Stores the executed code snippet
                    consequence: consequence,  // Stores an array of 
                    // outcomes from the action
                    link: await page.evaluate(() => window.location.href)  
                    // Stores the absolute 
                    // URL of the page
                }}
            }};

            Remember:
            - You call page.act({{ action: "..." }}) with the best next action 
              the user wants to perform on the given url.
            - Only visit the url specified by the user.
            - Always output the final JSON with the correct keys:
            - The "consequence" field should only be written after 
               capturing the result of the action.
            - Ensure these variables are properly 
                  assigned before outputting `updated_state`:**
                - `actionTaken`: The chosen action string.
                - `actionCode`: The executed code snippet.
                - `consequence`: Array containing the observed outcomes 
                   of the action. 
                  The values inside it should always be converted to a 
                  string or a number never a Object.
                - `link`: Extract the current URL dynamically using 
                        `await page.evaluate(() => window.location.href)`.

            Example: 
            {{
                "status": "success",
                "updated_state": {{
                    "action_taken": "Searched for the cars",
                    "code": "page.act({{ action: 'search', query: 'car' }})",
                    "consequence": [
                        { "name": "black car", "price": 200 },
                        { "name": "blue car", "price": 250 }
                    ],
                    "link": "https://www.car.com"
                }}
            }}
            """

        self.stagehand_prompt = TextPrompt(
            f"""You an assistant that helps in writing a 
        JavaScript snippet for a web automation task using 
        Stagehand. that acts 
        as a low level plan for getting the information 
        for the high 
        level task of {high_level_task}The snippet must 
        only contain 
        Stagehand action 
        commands (no imports, setup, or wrapping function).
        For example:
        - `await page.goto("https://www.example.com/");`
        - `await page.act({{ action: "Click the Sign In button." }});`
        - `const actions = await page.observe();`
        - `const data = await page.extract({{ instruction: 
          "Get user info." }});`

        Do not include:
        1. Any import statements like 
        `require('@browserbasehq/stagehand')`.
        2. Any declarations like `const stagehand = 
         new Stagehand()`.
        3. Any outer `async` function or IIFE wrapper.
        4. Console log lines for setup or imports.
        - Include a console log for each step to 
          indicate success.
        - Avoid using any CSS selectors directly in 
         `act()`—Stagehand 
        AI will infer what to do from plain language.
        - Extract structured information using 
         `await page.extract()` 
        with instructions like "Get the module details".
        - Extract structured information using 
         `await page.extract()` 
        with instructions like "Get the module details".
        - Use `observe()` to get actionable suggestions f
         rom the current page:
        
        const actions = await page.observe();
        console.log("Possible actions:", actions);

        const buttons = await page.observe({{
            instruction: "Find all the buttons on the page."
        }});

        - Use await page.extract({{ instruction: "..." }}) for 
        structured data extraction in natural language. 
        Example extractions:
        "Extract the current balance displayed on the 
         account summary page."
        "Extract the recent transactions list."
        - extract() must always use instruction, never action.
        - The `extract` function requires a `schema` that
         defines the expected 
        structure of the extracted data. 
        For example, if you are extracting module details, 
         the schema 
        should specify the fields and types, such as: 
        
        const data = await page.extract({{
            instruction: "extract the title, description, and 
            link of the quickstart",
            schema: z.object({{
                title: z.string(),
                description: z.string(),
                link: z.string()
            }})
        }});
        - IMPORTANT: Stagehand / OpenAI extraction requires 
           that all top-level 
            schemas be an 'object'.
            Therefore, if you want to extract an array of 
            items, wrap it in a 
            top-level object with afield like 'results' or 
            'items'. For example:

            // CORRECT:
            schema: z.object({{
                results: z.array(z.object({{
                title: z.string(),
                link: z.string()
                }}))
            }})

            // INCORRECT (will fail):
            schema: z.array(z.object({{
                title: z.string(),
                link: z.string()
            }}))

            So always wrap arrays in an object at the top 
            level of your 'schema'.
            
            - If needed use import {{ zodResponseFormat }} 
            from "openai/helpers/zod";

            const schema = z.object({{
                userId: z.string(),
            }});

            const format = zodResponseFormat(schema);

        - Do NOT combine multiple actions into one 
            instruction—each action 
            must be atomic.
        - Keep the script concise, and use no more than one 
          action per line.
        - Avoid any advanced planning—just deliver direct, concrete 
        instructions based on the task.
        - IMPORTANT:
            - ```javascript is NOT allowed in your response, 
            even in the beginning.
            - Do not include backticks or a "javascript" 
             label in your response. 
            Just return the plain JavaScript code.
        - First go to the link in the state.
        - If the url is google.com, then search for the term 
          you want.
        - Add a small wait right after searching on Google, 
          do something like
        - await page.act({{ action: "Wait a few seconds for 
          results to load." }}); 
        Then do the extraction. 
        (Stagehand supports a small “Wait for N seconds” or 
         “Wait for 
        results to appear” approach using 
        act({{ action: "Wait ..." }}).)
        - Address specific shortcomings highlighted in the 
         feedback, such as:
            - Missed steps.
            - Insufficient exploration of page elements.
            - Incomplete or incorrect data extraction.
        - Follow actionable suggestions to refine and expand 
          your approach.
        - Your plans should focus on exploring different 
         elements on the page, 
        especially those likely to yield useful data or 
         advance the task.
        - Include actions such as clicking buttons, links, 
          toggles, and 
        interacting with dropdowns or search bars.
        - Aim to uncover new information or pathways that could 
          help solve 
        the task.
        - Then proceed with rest of the plan.
        - If a search yields no results, do not stop. Try 
          alternative search 
        terms or synonyms.
        - If the page says “No results found,” instruct Stagehand 
         to search 
        for synonyms or check for similar items.
        - If the plan is stuck, propose an alternative approach, 
         such as 
        returning to Google and refining the query with additional 
         keywords.
        - If initial attempts fail or yield incomplete data, refine 
        or expand your approach using the feedback from the calling 
        agent or from the search results.
        - Use fallback steps like “try synonyms,” “use partial 
         matches,
        ” or “check for recommended articles” if the direct query 
         fails.
        - You can go back to a previous plan if you think that was 
        leading you in the correct direction.
        - Keep scope of the plan limited to solving the high level 
        task of {high_level_task}.
        You are a web automation assistant using Stagehand. Your role 
        is to:

        - Visit pages or perform searches.
        - Extract data from the page.
        - If needed, filter or process that data locally in your snippet.
        - Optionally, re-visit or do additional atomic actions based 
        on the new info.
        - Print final results as JSON so the calling process can read them.
        Important guidelines:
        - Atomic Stagehand instructions only. For example:
        await page.goto("https://www.example.com");
        await page.act({{ action: "Click on the Login button."}});
        const data = await page.extract({{ instruction: "...", 
        schema: z.object({ ... }) }});
        const actions = await page.observe();
        - Do not combine multiple steps into one act() instruction—each 
         line should be one discrete action.
        - Broad-to-narrow extraction pattern:
        - Broad extraction: “Extract all text, headings, or visible links.”
        - Local filter: Evaluate which items or links are relevant.
        - If you find a relevant link or portion, navigate or click.
        - Second extraction: Now specifically request the data you 
        actually need (like “Extract all the 
        - bubble metrics,” or “Extract the largest bubble's label,” etc.).
        - If the data is behind multiple clicks or expansions, continue with 
        atomic steps (act() to 
        click or scroll) until you see the data. Then extract again.
        - If you cannot find what you need, log that “No relevant data found” 
        and end gracefully, or try an alternate approach 
        (like refining your search).
        - This approach is generic and not tied to one site. It works as 
         follows:

            “Load a page or perform a search” → 
            atomic act({{ action: "Search for 'some phrase'" }}) 
            or goto(...).
            “Extract everything” with a broad instruction + broad schema.
            “Filter locally in JS,” if needed, to pick the relevant link.
            “Goto or click” to expand or open that detail.
            “Extract again” with a narrower instruction + schema.
            “Print final result.”
            Keep your snippet's instructions short and direct. 
            Provide one action per act(). 
            For extractions, use one extraction call for each chunk.

        Remember:
        - Always use absolute URLs for links in page.goto().
        - Use observe() to see potential clickable items or 
          possible actions.
        - Use extract() with a carefully chosen instruction and schema 
        to gather data. 
        - If you need more data, do another extraction.
        - Incorporate feedback from previous iterations to 
          improve the plan.
          Based on this high level task: "{high_level_task}", 
          generate a Stagehand 
          JavaScript snippet 
          with step-by-step instructions.
        
        - IMPORTANT: 
        1. You are a Low-Level Planner that writes a Stagehand 
            JavaScript snippet.  
            Remember to produce the final result as a JSON object called 
            
            {self.action_prompt}

            The the calling agent will provide you feedback on what 
            to inlcude 
            in the 'updated_state'. 
            Always define this in the code before ending it. 
            At the end of your snippet, always do the final extraction 
            to fill 
            these fields in a 
            variable called 'updated_state'. For example:

            - IMPORTANT: 
            - All of the values of the 
            'updated_state' should always be a string 
            or a number never a Object.
            - The link should be an absolute url extracted from the 
              url bar. 
            - Always extract the link in the end to 
            represent the last webpage visited.
            
            - Convert the value to a string or array before calling the 
            includes() method on it.
            - Retrieve the current date and time programmatically whenever a 
            question pertains to 'today' or involves time-related inquiries.

            
        2. Print or log the final data in a JSON-friendly 
        format so the 
        pipeline can read it. 
        For example:
        console.log("Final updated_state:", updated_state);

        3. If you cannot find the necessary info after multiple steps, 
        log "No relevant data found. 
        Attempt an alternative approach or refine the search."

        4. Keep your snippet concise.

        **Examples of valid atomic instructions** (one per line):
        await page.goto("https://www.example.com"); 
        await page.act({{ action: "Click the Sign In button." }}); 
        const data = await 
        page.extract({{ instruction: "Extract all text on page.", 
        schema: z.object({{ text: z.string() }}) }});

        **Do not** wrap multiple steps into a single act call. 
        For instance, don't do:
        await page.act({{action: "Click the sign in 
        button 
        and fill the form."}});

        That should be two lines: one for the click, 
        one for the fill.

        Please produce the Stagehand JavaScript snippet now, following all 
        of the above guidelines, 
        always ending with the final extraction snippet for `updated_state`.
        """
        )


class WebToolkit:
    def __init__(
        self,
        model_platform=ModelPlatformType.DEFAULT,
        model_type=ModelType.DEFAULT,
        model_config_dict=_DEFAULT_CHATGPT_CONFIG_DICT,
        headless_mode=True,
        debug=False,
    ):
        self.model_platform = model_platform
        self.model_type = model_type
        self.model_config_dict = model_config_dict
        self.headless_mode = headless_mode
        self.round_limit = 5
        self.debug = debug
        self.root = TreeNode(
            url="root",
            actions=["start"],
            action_taken="initialized",
            result="Initial state",
        )

        self.model = ModelFactory.create(
            model_platform=self.model_platform,
            model_type=self.model_type,
            model_config_dict=self.model_config_dict,
        )

        self.reasoning_model = ModelFactory.create(
            model_platform=self.model_platform,
            model_type=ModelType.O1,
            model_config_dict=self.model_config_dict,
        )

        # TO DO: add tree state to the prompt of web agent
        # A system message to instruct how to generate Stagehand code
        self.agent = ChatAgent(
            BaseMessage(
                role_name="Web Agent",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content="""You are an intelligent assistant that searches 
                    the web to answer the given question. Dynamically 
                    change the instructions 
                    in the tool call based on the error and the 
                    trajectory. If you are given a 
                    url or a url is known then start
                    with that url or else start with google.
                    
                  
                    IMPORTANT:

                    - The final answer should contain a 
                     single string "PASS"/"FAIL" along with the 
                    answer, if answer is found or the a 
                    new approach for the next iteration 
                    if answer is not found.
                    """,
            ),
            self.model,
            tools=[
                FunctionTool(self.stagehand_extract_text),
                FunctionTool(self.stagehand_screenshot_and_analyze_with_gpt4o),
                FunctionTool(self.take_action),
                FunctionTool(self.read_tree_state),
                FunctionTool(self.add_to_trajectory_tree),
            ],
        )

        self.stagehand_agent = ChatAgent(
            BaseMessage(
                role_name="Stagehand Agent",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content="""You are an intelligent that 
                writes stagehand code to do the 
                           requsted web automation.""",
            ),
            self.model,
        )

        self.action_plan_agent = ChatAgent(
            BaseMessage(
                role_name="Action Plan Agent",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content="""You are an intelligent that breaks 
                   down the task into 
                    a step by step plan for web interaction 
                    to answer the given question.""",
            ),
            self.reasoning_model,
        )

    def web_tool(self, task_prompt: str, url: str) -> Dict[str, Any]:
        """
        Accesses the web to fulfill the given task_prompt,
        optionally including a URL.
        Returns JSON results from web interactions or raises
        an error if unsuccessful.

        Args:
            task_prompt (str): Description of the task,
                               possibly including a URL.
            url (str): The webpage URL where the task should be
                        performed.
        Returns:
            Dict[str, Any]: Final JSON or textual answer
                            after web interactions.
        """
        if self.debug:
            print(f"[DEBUG]: web_tool called with task_prompt: {task_prompt}")

        response = None

        # 1. Extract page content (unless previously extracted)
        response_text_extract = self.stagehand_extract_text(url)

        # 2. Devise a step-by-step action plan
        plan_prompt = f"""Write down a step by step plan to solve the 
        task {task_prompt} 
        by visiting the url {url} based on the webpage 
        {response_text_extract}"""

        task_action_plan = self.action_plan_agent.step(
            input_message=plan_prompt
        )

        if self.debug:
            print(
                f"""[DEBUG]: Task action plan -> 
                {task_action_plan.msgs[-1].content}"""
            )

        # 3. Take next action
        action_prompt = f"""Take an action to solve the task {task_prompt}. 
            Use url {url}. 
            Choose actions from here {task_action_plan.msgs[-1].content}.
            Explain why you chose that step.
            Use tool take_action"""

        if self.debug:
            print(f"[DEBUG]: action_prompt: {action_prompt}")

        response_take_action = self.agent.step(input_message=action_prompt)

        if self.debug:
            print(f"[DEBUG]: response_take_action: {response_take_action}")

        # 4. Get only the available actions from the
        response_text_extract_actions = response_text_extract.split(
            "available actions :"
        )

        # 5. Update the trajectory tree
        update_prompt = f"""
            Update the trajectory tree based 
            on the actions taken {action_prompt}, a
            ctions available: {response_text_extract_actions} 
            and the result of the 
            actions: {response_take_action.msgs[-1].content}.
            Use tool add_to_trajectory_tree"""

        if self.debug:
            print(f"[DEBUG]: update_prompt trajectory: {update_prompt}")

        response_update_trajectory = self.agent.step(
            input_message=update_prompt
        )

        if self.debug:
            print(
                f"""[DEBUG]: response_update_trajectory: 
                {response_update_trajectory}"""
            )

        # 6. Attempt a final answer
        final_answer_prompt = f"""Based on the information gathered, 
        write the final answer
            to the task: {task_prompt}."""

        if self.debug:
            print(f"[DEBUG]: final_answer_prompt: {final_answer_prompt}")

        response = self.agent.step(input_message=final_answer_prompt)

        if self.debug:
            print(f"[DEBUG]: final response before loop: {response}")

        # 7. The observations on the initial state and the available
        # actions have been used to take
        # the initial action according to the fixed workflow
        # Now we let the agent do the rest of the planning
        # (it can use any tools it wants)

        if self.debug:
            print(f"[DEBUG]: current tree state: {self.read_tree_state(url)}")

        for i in range(self.round_limit):
            response = self.agent.step(
                input_message=response.msgs[-1].content
                + """use all the appropriate tools 
                at your disposal to solve this. 
            You can make multiple tool calls. 
            Make sure to read the trajectory tree 
            before taking an action with read_tree_state 
            and update the trajectory tree after taking an 
            action with add_to_trajectory_tree"""
            )
            print(f"response in iteration {i}: {response.msgs[-1].content }")
            if self.debug:
                print(
                    f"""[DEBUG]: current tree state 
                    in iteration {i}: {self.read_tree_state(url)}"""
                )

        # Return or raise error if no valid response
        if response and response.msgs:
            self.agent.reset()
            self.stagehand_agent.reset()
            return response.msgs[-1].content.strip()
        else:
            raise ValueError(
                "Failed to generate a final answer with Stagehand code."
            )

    def get_all_actions(self, url: str) -> str:
        r"""
        Retrieves all available actions on the specified
        URL using Stagehand.

        Args:
            url (str): The webpage URL to analyze.

        Returns:
            str: JSON string containing all available
            actions and the code used.
        """
        if self.debug:
            print(
                """[DEBUG]: Calling the get all 
                actions tool from stagehand agent"""
            )

        # For action specific prompt for stagehand agent
        # action_mode = "get_action"

        # Generate Stagehand code
        js_code = f"""
const {{ Stagehand }} = require('@browserbasehq/stagehand');

(async () => {{
const stagehand = new Stagehand({{ headless: false }});
await stagehand.init();
const page = stagehand.page;

// Helper function to build a unique CSS path
function createCssPath(el) {{
    if (!el) return "No Element";
    const names = [];
    while (el.parentElement) {{
        let name = el.tagName.toLowerCase();
        // If there's an ID, use it and break 
        // (IDs are usually unique)
        if (el.id) {{
            name += "#" + el.id;
            names.unshift(name);
            break;
        }} else {{
            // Count how many siblings of the same 
            // type are before this element
            let sibling = el;
            let nth = 1;
            while ((sibling = 
            sibling.previousElementSibling) !== null) 
            {{
            if (sibling.tagName.toLowerCase() === 
            el.tagName.toLowerCase()) {{
                nth++;
                }}
            }}
            if (nth > 1) {{
                name += 
                `:nth-of-type(${{nth}})`;
            }}
            names.unshift(name);
            el = el.parentElement;
        }}
    }}
    return names.join(" > ");
}}

try {{
    await page.goto("{url}", {{ timeout: 60000 }});
    console.log("Navigated to {url}");

    // Small wait to ensure the page loads fully
    await page.act({{ action: "Wait a few 
    seconds for elements to appear." }});

    // Observing all possible actions on the page
    const actions = await page.observe();

    if (!actions || actions.length === 0) {{
    console.warn("No actions detected on the page.");
    }}

    // Extract action descriptions and create a more 
    //human-friendly identifier
    const actionDetails = 
    await Promise.all(actions.map(async 
    (action, index) => {{
// This is our final JSON for 
// the action
const actionJson = {{
    description: action.description,
    // We'll store the index-based ID
    elementId: `elem-${{index}}`,
    cssPath: "No Element",
    tagName: "unknown",
    textSnippet: ""
}};

// Attempt to locate the actual DOM element 
// (either from `action.element` or `action.selector`)
if (action.element) {{
    // Evaluate in the page context
    const info = await page.evaluate((el, idx) => 
    {{
        const obj = 
        {{ tagName: el.tagName.toLowerCase() }};
        // Provide a short snippet of 
        visible text if present
        if (el.innerText) {{
            obj.textSnippet = 
            el.innerText.slice(0, 50).trim();
        }}
        return obj;
    }}, action.element, index);

    actionJson.tagName = 
    info.tagName || "unknown";
    actionJson.textSnippet = info.textSnippet || "";
    
    // Generate a CSS path from the element handle
    const path = await page.evaluate(el => {{
        // We'll run the same logic client-side
        const names = [];
        let current = el;
        while (current.parentElement) {{
            let name = current.tagName.toLowerCase();
            // If there's an 
            // ID, use it and break
            if (current.id) {{
                name += "#" + current.id;
                names.unshift(name);
                break;
            }} else {{
                let sibling = current;
                let nth = 1;
                while ((sibling = 
                sibling.previousElementSibling) 
                !== null) {{
                    if (sibling.tagName.toLowerCase() === 
                    current.tagName.toLowerCase()) {{
                        nth++;
                    }}
                }}
                if (nth > 1) {{
                    name 
                    += `:nth-of-type(${{nth}})`;
                }}
                names.unshift(name);
                current = current.parentElement;
            }}
        }}
        return names.join(" > ");
    }}, action.element);

    actionJson.cssPath = path || "No Element";

}} else if (action.selector) {{
if (action.selector.startsWith("xpath=")) {{
// Handle XPath
const xpathString = action.selector.replace(/^xpath=/, "");
const info = await page.evaluate(xpath => {{
    const el = document.evaluate(xpath, document, null, 
    XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
    if (!el) return null;
    return {{
        tagName: el.tagName.toLowerCase(),
        textSnippet: el.innerText ? 
        el.innerText.slice(0, 50).trim() : ""
    }};
}}, xpathString);

if (info) {{
    actionJson.tagName = info.tagName || "unknown";
    actionJson.textSnippet = info.textSnippet || "";
}} else {{
    // If no element found, skip
}}

// Build a CSS path from the found element, if any
const path = await page.evaluate(xpath => {{
const el = 
document.evaluate(xpath, document, null, 
XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
if (!el) return "No Element";
const names = [];
let current = el;
while (current.parentElement) {{
let name = 
current.tagName.toLowerCase();
if (current.id) {{
    name += "#" + current.id;
    names.unshift(name);
    break;
}} else {{
    let sibling = current;
    let nth = 1;
    while ((sibling = 
    sibling.previousElementSibling) 
            !== null) {{
        if (sibling.tagName.toLowerCase() 
        === current.tagName.toLowerCase()) {{
            nth++;
        }}
    }}
    if (nth > 1) {{
        name += `:nth-of-type(${{nth}})`;
    }}
    names.unshift(name);
    current = current.parentElement;
}}
}}
return names.join(" > ");
}}, xpathString);

actionJson.cssPath = path || "No Element";

}} else {{
// Handle standard CSS selector
const info = await page.evaluate(selector => {{
    const el = 
    document.querySelector
    (selector);
    if (!el) return null;
    return {{
        tagName: el.tagName.toLowerCase(),
        textSnippet: 
        el.innerText ? 
        el.innerText.slice(0, 50).trim() 
        : ""
    }};
}}, action.selector);

if (info) {{
    actionJson.tagName = info.tagName || "unknown";
    actionJson.textSnippet 
    = info.textSnippet || "";
}}

const path = await page.evaluate(selector => {{
    const el = document.querySelector(selector);
    if (!el) return "No Element";
    const names = [];
    let current = el;
    while (current.parentElement) {{
        let name = current.tagName.toLowerCase();
        if (current.id) {{
            name += "#" + current.id;
            names.unshift(name);
            break;
        }} else {{
            let sibling = current;
            let nth = 1;
            while ((sibling = 
            sibling.previousElementSibling) 
            !== null) {{
            if (sibling.tagName.toLowerCase() 
            === current.tagName.toLowerCase()) 
            {{
                nth++;
                }}
            }}
            if (nth > 1) {{
                name += `:nth-of-type(${{nth}})`;
            }}
            names.unshift(name);
            current = current.parentElement;
        }}
    }}
    return names.join(" > ");
}}, action.selector);

actionJson.cssPath = path || "No Element";
}}
    }} else {{
        // Fallback: no element, no selector -> 
        // just give a generic marker
        actionJson.tagName = "no-element";
        actionJson.cssPath = "No Element";
        actionJson.textSnippet = "";
    }}

    return actionJson;
}}));

const result = {{
    "status": "success",
    "updated_state": {{
        "actions": actionDetails,
        "code": "const actions = await page.observe();",
        "link": "{url}"
    }}
}};

console.log("Final updated_state:", 
JSON.stringify(result, null, 2));

}} catch (error) {{
console.error("Final updated_state:", 
JSON.stringify({{
    "status": "failure",
    "link": "{url}",
    "error": error.message
}}));
}} finally {{
await stagehand.close();
}}
    }})();
            """

        # Run code in Node, capture JSON in string format
        result_str = self._run_stagehand_script_in_node(js_code)

        if self.debug:
            print(f"[DEBUG]: Generated code: {js_code}")

            print(f"[DEBUG]: result_str: {result_str}")

        # Return JSON in string format
        try:
            return result_str
        except json.JSONDecodeError:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"""No valid JSON output. 
                    Last script line:\n{result_str}""",
                }
            )

    def take_action(self, task_prompt: str, url: str) -> str:
        r"""
        Take one action on the specified webpage
        based on the given task instruction.

        Args:
            task_prompt (str): Description of the task to automate.
                               You can also inlcude errors from the
                               previous task here to improve the code in
                               this iteration.
            url (str): The webpage URL on which to perform the action.

        Returns:
            str: A JSON string from the Stagehand script,
                 or an error if JSON parsing fails.
        """
        if self.debug:
            print(
                "[DEBUG]: Calling the take_action tool from Stagehand agent."
            )

        # Prepare for the 'take_action' mode
        action_mode = "take_action"

        # Generate the Stagehand code (your method should handle
        # both task and url)
        js_code = self._generate_stagehand_code(task_prompt + url, action_mode)

        # Execute the generated code in Node.js
        result_str = self._run_stagehand_script_in_node(js_code)

        if self.debug:
            print(f"[DEBUG]: Generated code:\n{js_code}")
            print(f"[DEBUG]: result_str:\n{result_str}")

        # Return the raw JSON string
        try:
            return result_str
        except json.JSONDecodeError:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"""No valid JSON output. 
                    Last script line:\n{result_str}""",
                }
            )

    def read_tree_state(self, node_url: Optional[str] = None) -> str:
        """
        Recursively reads part (or all) of the trajectory tree to give an
        LLM agent a clear overview
        of the decisions made so far. Each node in this tree corresponds
        to a particular state the
        agent encountered, the actions that were available, the action
        ultimately taken, and the result of that action.

        The tree is especially useful for future planning and
        revisiting past decisions:
        - The agent can analyze the sequence of actions (including mistakes)
          to refine its strategy.
        - The agent can identify previously failed actions (past mistakes)
          and avoid repeating them.

        Args:
            node_url (Optional[str]): The URL of the node at which
                                    to start reading.
                                    If None, starts reading from self.root.
        Returns:
            str: The formatted string representation of the subtree starting at
                either the specified node or the root if node_url is None.
        """
        if node_url is None:
            node = self.root
            if node is None:
                return "Trajectory tree is not initialized."
        else:
            node = self.root.find_(node_url) if self.root else None
            if node is None:
                return f"""Node with URL '{node_url}' not 
                found in the trajectory tree."""

        return self._build_tree_representation(node, level=0)

    def _build_tree_representation(self, node: "TreeNode", level: int) -> str:
        """
        Internal helper that builds the
        string representation from a given node.

        Args:
            node (TreeNode): The node from which to build the representation.
            level (int): Current depth level for indentation.
        Returns:
            str: The formatted, recursive representation of
                 this node and its children.
        """
        indent = "  " * level
        tree_representation = (
            f"{indent}- URL: {node.url}\n"
            f"{indent}  Available Actions: {node.actions}\n"
            f"{indent}  Action Taken: {node.action_taken}\n"
            f"{indent}  Result: {node.result}\n"
        )

        for child in node.next:
            tree_representation += self._build_tree_representation(
                child, level + 1
            )

        return tree_representation

    def add_to_trajectory_tree(
        self,
        location: str,
        url: str,
        actions: List[str],
        action_taken: str,
        result: str,
    ) -> str:
        r"""
        Dynamically adds a new node to the trajectory tree
        to reflect the agent's latest decision.

        When an LLM agent reaches a new state in its decision-making process,
        it can call this
        method to log the available actions, the specific action taken,
        and the resulting outcome.
        If any past mistakes are relevant, they can be added so that the agent
        (or future calls)
        can quickly reference them. For instance, an action that did not
        yield a useful result
        on a previous run should be noted as a mistake.

        Args:
            location (str): The URL of the parent node where we want
                            to add the new node.
            url (str): The URL of the new node.
            actions (List[str]): Possible actions at the new node.
                                Write these actions as returned by the
                                get_all_actions tool.
            action_taken (str): The action taken at the new node.
                                Write the code
                                returned by the take_action tool.
            result (str): The outcome/result of the action. Use the consequence
                        returned by the take_action tool.

        Returns:
            str: A response message indicating success or failure.
        """

        if self.debug:
            print(
                """[DEBUG]: Adding TreeNode to 
                Trajectory tree with the web agent"""
            )

        # Create new tree node
        new_node = TreeNode(
            url=url, actions=actions, action_taken=action_taken, result=result
        )

        # Add to the trajectory tree
        success = self.root.add_node_at(location, new_node)

        if success:
            return f"Node '{url}' successfully added under '{location}'."
        else:
            return f"""Failed to add node '{url}' - Parent node 
        '{location}' not found."""

    #
    # Internals
    #
    def _generate_stagehand_code(
        self, high_level_task: str, action_mode: str
    ) -> str:
        r"""
        Internal method for generating Stagehand code.

        Args:
            high_level_task (str): Description of the task to automate.
            action_mode (str): Task specific prompt of stagehand agent

        Returns:
            str: The generated JavaScript code.
        """

        # The prompt with guidelines for Stagehand snippet generation
        stagehand_prompt = StagehandPrompts(
            high_level_task, action_mode
        ).stagehand_prompt

        response = self.stagehand_agent.step(input_message=stagehand_prompt)

        if response and response.msgs:
            return response.msgs[-1].content.strip()
        else:
            raise ValueError("Failed to generate Stagehand code.")

    def _run_stagehand_script_in_node(self, js_code: str) -> str:
        r"""
        Internal method that executes the Stagehand code under
        Node.js and returns the final JSON line from stdout.

        Args:
            js_code (str): The JavaScript code to execute.

        Returns:
            str: The final output of the script or an error message.
        """

        # Wrap the user snippet with Stagehand environment
        wrapper_code = f"""
        const {{ Stagehand }} = require('@browserbasehq/stagehand');
        const z = require('zod');

        (async () => {{
            const stagehand = new Stagehand({{ headless: {"true" if 
            self.headless_mode else "false"} }});
            await stagehand.init();
            const page = stagehand.page;
            console.log("Starting Stagehand automation...");
            try {{
                // Insert the generated snippet
                {js_code}
            }} catch (error) {{
                console.error("Final updated_state: ", JSON.stringify({{
                    status: "failure",
                    "link": await page.evaluate(() => window.location.href),
                    error: error.message
                }}));
            }} finally {{
                await stagehand.close();
                console.log("Stagehand session closed.");
            }}
        }})();
        """

        # Run the script in Node.js
        node_process = SubprocessInterpreter(
            require_confirm=False, print_stdout=False, print_stderr=False
        )

        exec_result = node_process.run(wrapper_code, "node")

        # Attempt to parse final JSON from logs:
        final_json = self._parse_json_from_output(exec_result)
        if final_json is not None:
            # Return as a JSON string for the caller to parse
            return json.dumps(final_json)
        else:
            # If no valid JSON found in logs, return an error as JSON
            return json.dumps(
                {
                    "status": "error",
                    "message": "No valid JSON found in node logs.",
                }
            )

    def stagehand_extract_text(self, url: str) -> str:
        r"""
        Extracts all visible text from a webpage using Stagehand
        if the correct URL to a webpage is known.

        Args:
            url (str): The webpage URL to extract text from.

        Returns:
            Dict[str, Any]: Extracted text in JSON format.
        """

        if self.debug:
            print(
                "[DEBUG]: Calling the extract_text tool from stagehand agent"
            )

        # JavaScript code to extract text using Stagehand
        js_code = f"""
        const {{ Stagehand }} = require('@browserbasehq/stagehand');
        const z = require('zod');

        (async () => {{
            const stagehand = new Stagehand({{ headless: false }});
            await stagehand.init();
            const page = stagehand.page;
            try {{
                await page.goto("{url}");

                // Extract visible text
                const textData = await page.extract({{
                    instruction: "Extract all visible text on the page.",
                    schema: z.object({{ text: z.string() }})
                }});

                // Create final JSON object
                const extractedData = {{
                    status: "success",
                    text: textData.text,
                    link: "{url}"
                }};

                console.log("Final updated_state: ", 
                JSON.stringify(extractedData, null, 2));

            }} catch (error) {{
                console.error("Final updated_state: ", JSON.stringify({{
                    status: "failure",
                    link: "{url}",
                    error: error.message
                }}));
            }} finally {{
                await stagehand.close();
            }}
        }})();
        """

        # Run Stagehand script
        node_process = SubprocessInterpreter(
            require_confirm=False, print_stdout=False, print_stderr=False
        )
        exec_result = node_process.run(js_code, "node")

        # Attempt to parse final JSON from logs:
        result_str = self._parse_json_from_output(exec_result)

        if self.debug:
            print(f"[DEBUG]: Generated code: {js_code}")

            print(f"[DEBUG]: result_str: {result_str}")

        # get all the available actions
        actions = self.get_all_actions(url)

        if result_str is not None:
            # Return as a JSON string to the caller
            return json.dumps(result_str) + f"available actions: {actions}"
        else:
            # If no valid JSON found in logs, return an error as JSON
            return json.dumps(
                {
                    "status": "error",
                    "message": "No valid JSON found in node logs.",
                }
            )

    def stagehand_screenshot_and_analyze_with_gpt4o(
        self, question: str, url: str
    ) -> str:
        r"""
        Captures multiple screenshots while scrolling and sends each screenshot
        to GPT-4o for analysis.

        Args:
            url (str): The webpage URL to analyze.
            question (str): The question to be answered.


        Returns:
            Dict[str, Any]: JSON response containing:
                - GPT-4o analysis for each screenshot.
                - Screenshot file paths.
        """

        if self.debug:
            print(
                """[DEBUG]: Capturing screenshots for analysis with 
                stagehand agent"""
            )

        screenshot_folder = "screenshots"
        os.makedirs(screenshot_folder, exist_ok=True)
        screenshot_base = os.path.join(
            screenshot_folder, os.path.basename(url).replace("/", "_")
        )

        # JavaScript code for scrolling and taking screenshots
        js_code = f"""
          const {{ Stagehand }} = require('@browserbasehq/stagehand');
          const fs = require('fs');

          (async () => {{
              const stagehand = new Stagehand({{ headless: false }});
              await stagehand.init();
              const page = stagehand.page;
              try {{
                  await page.goto("{url}");

                  let screenshots = [];
                  let totalHeight = 0;
                  let viewportHeight = await 
                  page.evaluate(() => window.innerHeight);
                  let scrollHeight = await page.evaluate(() => 
                        document.body.scrollHeight
                    );

                  let scrollY = 0;
                  let index = 0;

                  // Scroll and take multiple screenshots
                  while (scrollY < scrollHeight) {{
                      let screenshot_path = 
                      "{screenshot_base}_" + index + ".png";
                      await page.screenshot({{ path: screenshot_path }});
                      screenshots.push(screenshot_path);

                      totalHeight += viewportHeight;
                      scrollY += viewportHeight;
                      await page.evaluate((height) => 
                      window.scrollBy(0, height), viewportHeight);
                      await page.act({{ action: "Wait a second for" + 
                      "scrolling to complete." }});
                      index++;
                  }}

                  // Final JSON object
                  const extractedData = {{
                      "status": "success",
                      "screenshots": screenshots,
                      "link": "{url}"
                  }};

                  console.log("updated_state: ", 
                  JSON.stringify(extractedData, null, 2));

              }} catch (error) {{
                  console.error("updated_state: ", JSON.stringify({{
                      "status": "failure",
                      "link": "{url}",
                      "error": error.message
                  }}));
              }} finally {{
                  await stagehand.close();
              }}
          }})();
        """

        print("[DEBUG]: Executing JavaScript in Node.js")

        # Run the script in Node.js
        node_process = SubprocessInterpreter(
            require_confirm=False, print_stdout=True, print_stderr=True
        )

        exec_result = node_process.run(js_code, "node")

        print(f"[DEBUG] Raw Node.js Output:\n{exec_result}")

        raw_json = self._parse_json_from_output(exec_result)

        # Convert JSON string to a Python dictionary
        try:
            final_json = json.loads(raw_json)
        except json.JSONDecodeError:
            print("[ERROR]: Failed to parse JSON from extracted output.")
            return json.dumps(
                {
                    "status": "error",
                    "message": "Failed to parse extracted JSON.",
                }
            )

        if "screenshots" not in final_json:
            print("[ERROR]: No valid screenshots found.")
            return json.dumps(
                {
                    "status": "error",
                    "message": "No screenshots found in output.",
                }
            )

        screenshots = final_json["screenshots"]

        print("[DEBUG]: Screenshots Captured:\n", screenshots)

        # Analyze each screenshot with GPT-4o
        gpt_results = self._analyze_screenshots_with_gpt4o(
            question, screenshots
        )

        # Final response with GPT-4o results
        return json.dumps(
            {"status": "success", "link": url, "gpt_analysis": gpt_results}
        )

    def _analyze_screenshots_with_gpt4o(
        self, question: str, screenshots: List[str]
    ) -> List[Dict[str, Any]]:
        r"""
        Sends each screenshot to GPT-4o for analysis.

        Args:
            question (str): The question to be answered.
            screenshots (List[str]): List of screenshot file paths.

        Returns:
            List[Dict[str, Any]]: List of GPT-4o responses for each screenshot.
        """

        if self.debug:
            print("[DEBUG]: Calling _analyze_screenshots_with_gpt4o")

        results = []

        for screenshot_path in screenshots:
            # Ensure the screenshot file exists before proceeding
            if not os.path.exists(screenshot_path):
                print(f"[ERROR]: Screenshot {screenshot_path} not found.")
                continue

            # Convert the screenshot to a PIL Image object
            image = Image.open(screenshot_path)

            gpt_input = BaseMessage(
                role_name="GPT-4o Screenshot Analyzer",
                role_type=RoleType.USER,
                meta_dict=None,
                image_list=[image],
                image_detail="high",
                content=f"""Analyze this screenshot and describe the 
                visual elements. {question}""",
            )

            if self.debug:
                print(f"[DEBUG] Sending GPT-4o Input for {screenshot_path}")

            # Send to GPT-4o model for **vision processing**
            response = self.agent.step(input_message=gpt_input)

            # Debugging: Check response
            if self.debug:
                print(f"[DEBUG] GPT-4o Response: {response}")

            # Extract response content
            gpt_response = (
                response.msgs[-1].content.strip()
                if response and response.msgs
                else "No response from model."
            )

            results.append(
                {"screenshot": screenshot_path, "analysis": gpt_response}
            )

        return results

    def _parse_json_from_output(self, text: str):
        r"""
        Extracts a substring that starts with the first '{' following
        the keyword 'updated_state: ' and continues until the matching
        '}' is found.

        Args:
            text (str): The input text containing the Stagehand code.

        Returns:
            str: The extracted JSON snippet or an empty string if not found.
        """
        start_marker = "updated_state:"
        start_index = text.find(start_marker)
        if start_index == -1:
            return ""
        # Locate the first '{' after the marker
        start_index = text.find("{", start_index)
        if start_index == -1:
            return ""

        stack = []
        for i in range(start_index, len(text)):
            char = text[i]
            if char == "{":
                stack.append("{")
            elif char == "}":
                stack.pop()
                if not stack:
                    # Return from the first '{' up to and including this '}'
                    return text[start_index : i + 1]
        return ""

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [
            FunctionTool(self.web_tool),
        ]
