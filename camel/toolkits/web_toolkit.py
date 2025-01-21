import os
import subprocess
import json
from typing import Any, Dict, List

from camel.toolkits.base import BaseToolkit
from camel.toolkits.function_tool import FunctionTool
from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType, RoleType
from camel.configs import ChatGPTConfig
from camel.messages import BaseMessage
from camel.interpreters.subprocess_interpreter import SubprocessInterpreter

class WebToolkit(BaseToolkit):
    r"""A class representing a toolkit for web use.

    This class provides methods for interacting with websites by writing direct JavaScript code via tools like Stagehand.
    """

    def __init__(
        self,
        model_platform=ModelPlatformType.OPENAI,
        model_type=ModelType.GPT_4O_MINI,
        model_config_dict=ChatGPTConfig(temperature=0.0).as_dict(),
    ):
        self.model_platform = model_platform
        self.model_type = model_type
        self.model_config_dict = model_config_dict

    def stagehand_tool(self, task_prompt: str) -> Dict[str, Any]:
        r"""
        Single entry point that:
         1) Generates Stagehand JavaScript code to interact with the web
         2) Executes it under Node.js
         3) Returns the final JSON result

        Args:
            task_prompt (str): Description of the task to automate.

        Returns:
            Dict[str, Any]: JSON result from the Stagehand script, or an error.
        """

        # Generate Stagehand code
        js_code = self._generate_stagehand_code(task_prompt)

        # Run code in Node, capture JSON
        result_str = self._run_stagehand_script_in_node(js_code)

        # Attempt to parse JSON
        try:
            return json.loads(result_str)
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": f"No valid JSON output. Last script line:\n{result_str}",
            }

    #
    # Internals
    #
    def _generate_stagehand_code(self, high_level_task: str) -> str:
        r"""
        Internal method for generating Stagehand code.
        """
        model = ModelFactory.create(
            model_platform=self.model_platform,
            model_type=self.model_type,
            model_config_dict=self.model_config_dict,
        )

        # A system message to instruct how to generate Stagehand code
        agent = ChatAgent(
            BaseMessage(
                role_name="Stagehand Agent",
                role_type=RoleType.ASSISTANT,
                meta_dict=None,
                content="You are an intelligent assistant that searches the web to answer the given question.",
            ),
            model,
        )

        # The prompt with guidelines for Stagehand snippet generation
        stagehand_prompt = f"""You an assistant that helps in writing a JavaScript snippet for a web automation task using Stagehand. that acts as a low level plan for getting the information for the high level task of {high_level_task}
    The snippet must only contain Stagehand action commands (no imports, setup, or wrapping function).
    For example:
    - `await page.goto("https://www.example.com/");`
    - `await page.act({{ action: "Click the Sign In button." }});`
    - `const actions = await page.observe();`
    - `const data = await page.extract({{ instruction: "Get user info." }});`

    Do not include:
    1. Any import statements like `require('@browserbasehq/stagehand')`.
    2. Any declarations like `const stagehand = new Stagehand()`.
    3. Any outer `async` function or IIFE wrapper.
    4. Console log lines for setup or imports.
    - Include a console log for each step to indicate success.
    - Avoid using any CSS selectors directly in `act()`—Stagehand AI will infer what to do from plain language.
    - Extract structured information using `await page.extract()` with instructions like "Get the module details".
    - Extract structured information using `await page.extract()` with instructions like "Get the module details".
    - Use `observe()` to get actionable suggestions from the current page:
    
    const actions = await page.observe();
    console.log("Possible actions:", actions);

    const buttons = await page.observe({{
        instruction: "Find all the buttons on the page."
    }});

    - Use await page.extract({{ instruction: "..." }}) for structured data extraction in natural language. Example extractions:
    "Extract the current balance displayed on the account summary page."
    "Extract the recent transactions list."
    - extract() must always use instruction, never action.
    - The `extract` function requires a `schema` that defines the expected structure of the extracted data. 
      For example, if you are extracting module details, the schema should specify the fields and types, such as: 
       
       const data = await page.extract({{
           instruction: "extract the title, description, and link of the quickstart",
           schema: z.object({{
               title: z.string(),
               description: z.string(),
               link: z.string()
           }})
       }});
    - IMPORTANT: Stagehand / OpenAI extraction requires that all top-level schemas be an 'object'.
        Therefore, if you want to extract an array of items, wrap it in a top-level object with a
        field like 'results' or 'items'. For example:

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

        So always wrap arrays in an object at the top level of your 'schema'.
    - Do NOT combine multiple actions into one instruction—each action must be atomic.
    - Keep the script concise, and use no more than one action per line.
    - Avoid any advanced planning—just deliver direct, concrete instructions based on the task.
    - IMPORTANT:
        - ```javascript is NOT allowed in your response, even in the beginning.
        - Do not include backticks or a "javascript" label in your response. Just return the plain JavaScript code.
    - First go to the link in the state.
    - If the url is google.com, then search for the term you want.
    - Add a small wait ight after searching on Google, do something like
    - await page.act({{ action: "Wait a few seconds for results to load." }}); Then do the extraction. (Stagehand supports a small “Wait for N seconds” or “Wait for results to appear” approach using act({{ action: "Wait ..." }}).)
     - Address specific shortcomings highlighted in the feedback, such as:
        - Missed steps.
        - Insufficient exploration of page elements.
        - Incomplete or incorrect data extraction.
    - Follow actionable suggestions to refine and expand your approach.
    - Your plans should focus on exploring different elements on the page, especially those likely to yield useful data or advance the task.
    - Include actions such as clicking buttons, links, toggles, and interacting with dropdowns or search bars.
    - Aim to uncover new information or pathways that could help solve the task.
    - Then proceed with rest of the plan.
    - If a search yields no results, do not stop. Try alternative search terms or synonyms.
    - If the page says “No results found,” instruct Stagehand to search for synonyms or check for similar items.
    - If the plan is stuck, propose an alternative approach, such as returning to Google and refining the query with additional keywords.
    - If initial attempts fail or yield incomplete data, refine or expand your approach using the feedback from the calling agent or from the search results.
    - Use fallback steps like “try synonyms,” “use partial matches,” or “check for recommended articles” if the direct query fails.
    - You can go back to a previous plan if you think that was leading you in the correct direction.
    - Keep scope of the plan limited to solving the high level task of {high_level_task}.
    You are a web automation assistant using Stagehand. Your role is to:

    Visit pages or perform searches.
    Extract data from the page.
    If needed, filter or process that data locally in your snippet.
    Optionally, re-visit or do additional atomic actions based on the new info.
    Print final results as JSON so the calling process can read them.
    Important guidelines:
    Atomic Stagehand instructions only. For example:
    await page.goto("https://www.example.com");
    await page.act({{ action: "Click on the Login button."}});
    const data = await page.extract({{ instruction: "...", schema: z.object({ ... }) }});
    const actions = await page.observe();
    Do not combine multiple steps into one act() instruction—each line should be one discrete action.
    Broad-to-narrow extraction pattern:
    Broad extraction: “Extract all text, headings, or visible links.”
    Local filter: Evaluate which items or links are relevant.
    If you find a relevant link or portion, navigate or click.
    Second extraction: Now specifically request the data you actually need (like “Extract all the bubble metrics,” or “Extract the largest bubble’s label,” etc.).
    If the data is behind multiple clicks or expansions, continue with atomic steps (act() to click or scroll) until you see the data. Then extract again.
    If you cannot find what you need, log that “No relevant data found” and end gracefully, or try an alternate approach (like refining your search).
    This approach is generic and not tied to one site. It works as follows:

    “Load a page or perform a search” → atomic act({{ action: "Search for 'some phrase'" }}) or goto(...).
    “Extract everything” with a broad instruction + broad schema.
    “Filter locally in JS,” if needed, to pick the relevant link.
    “Goto or click” to expand or open that detail.
    “Extract again” with a narrower instruction + schema.
    “Print final result.”
    Keep your snippet’s instructions short and direct. Provide one action per act(). For extractions, use one extraction call for each chunk.

    Remember:

    Use observe() to see potential clickable items or possible actions.
    Use extract() with a carefully chosen instruction and schema to gather data.
    If you need more data, do another extraction.
    - Incorporate feedback from previous iterations to improve the plan.
    Based on this high level task: "{high_level_task}", generate a Stagehand JavaScript snippet with step-by-step instructions.
    
    - IMPORTANT: 
    1. You are a Low-Level Planner that writes a Stagehand JavaScript snippet.  
        Remember to produce the final result as a JSON object called 'updated_state', which the system will read as:

        {{
        "status": "success",
        "updated_state": {{
            "title": ...,
            "finalAnswer": ...,
            "uniqueItems": [...],
            "link": ...
        }}
        }}

        The the calling agent will provide you feedback on what to inlcude in the 'updated_state'. At the end of your snippet, always do the final extraction to fill these fields in a variable called 'updated_state'. For example:

        const updated_state = {{
        status: "success",
        updated_state: {{
            title: extractedTitle,
            finalAnswer: extractedFinalAnswer,
            uniqueItems: extractedUniqueItems,
            link: extractedLink
        }}
        }};
        
    2. Print or log the final data in a JSON-friendly format so the pipeline can read it. For example:
    console.log("Final updated_state:", updated_state);

    3. If you cannot find the necessary info after multiple steps, log "No relevant data found. Attempt an alternative approach or refine the search."

    4. Keep your snippet concise.

    **Examples of valid atomic instructions** (one per line):
    await page.goto("https://www.example.com"); await page.act({{ action: "Click the Sign In button." }}); const data = await page.extract({{ instruction: "Extract all text on page.", schema: z.object({{ text: z.string() }}) }});

    **Do not** wrap multiple steps into a single act call. For instance, don’t do:
    await page.act({{ action: "Click the sign in button and fill the form." }});

    That should be two lines: one for the click, one for the fill.

    Please produce the Stagehand JavaScript snippet now, following all of the above guidelines, always ending with the final extraction snippet for `updated_state`.
    """

        response = agent.step(BaseMessage(role_name="User", role_type=RoleType.USER, meta_dict=None, content=stagehand_prompt))
        if response and response.msgs:
            return response.msgs[-1].content.strip()
        else:
            raise ValueError("Failed to generate Stagehand code.")

    def _run_stagehand_script_in_node(self, js_code: str) -> str:
        r"""
        Internal method that executes the Stagehand code under Node.js and returns
        the final JSON line from stdout.
        """

        # Wrap the user snippet with Stagehand environment
        wrapper_code = f"""
const {{ Stagehand }} = require('@browserbasehq/stagehand');
const z = require('zod');

(async () => {{
    const stagehand = new Stagehand({{ headless: false }});
    await stagehand.init();
    const page = stagehand.page;
    console.log("Starting Stagehand automation...");
    try {{
        // Insert the generated snippet
        {js_code}

        console.log(JSON.stringify({{
            status: "success",
            updated_state
        }}));
    }} catch (error) {{
        console.error(JSON.stringify({{
            status: "failure",
            error: error.message
        }}));
    }} finally {{
        await stagehand.close();
        console.log("Stagehand session closed.");
    }}
}})();
"""


        # Run the script in Node.js
        node_process = SubprocessInterpreter(require_confirm=True, print_stdout=True, print_stderr=True)
        
        exec_result = node_process.run(wrapper_code, "node")

        # Return the second-last line or fallback to last line
        if exec_result.startswith("(stderr"):
            return "Failure: No output from Node.js script."

        return exec_result

    def get_tools(self) -> List[FunctionTool]:
        r"""Returns a list of FunctionTool objects representing the
        functions in the toolkit.

        Returns:
            List[FunctionTool]: A list of FunctionTool objects
                representing the functions in the toolkit.
        """
        return [FunctionTool(self.stagehand_tool)]
