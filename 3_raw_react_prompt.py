# CHANGE 1: Add re + inspect - we'll parse tool calls from raw text instead of structured JSON
import re
import inspect

from dotenv import load_dotenv

load_dotenv()

import ollama

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4:latest"

@traceable(run_type="tool")
def get_product_price(product: str) -> float | str:
    """Lookup the price of the product from the catalog.
    Valid products: laptops, desktops, monitors (singular forms like 'laptop' also work)."""
    print(f"  >> Executing get_product_price(product='{product}')")
    prices = {"laptops": 1000.0, "desktops": 1500.0, "monitors": 300.0}
    key = product.strip().lower()
    if not key.endswith("s"):
        key = key + "s"
    if key not in prices:
        return f"Error: product '{product}' not found. Valid products: laptops, desktops, monitors"
    return prices[key]

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price and return the final price.
    Available discount tiers: gold, silver, bronze"""
    print(f"   >> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    price = float(price)
    discount_percentags={"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentags.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

tools = {
  "get_product_price": get_product_price,
  "apply_discount": apply_discount,
}

# CHANGE 3: Delete the JSON schemas. Tools now live inside the prompt as plain text.
# We derive descriptions from the functions themselves using inspect.

def get_tool_descriptions(tools_dict):
  descriptions = []
  for tool_name, tool_function in tools_dict.items():
    # __wrapped__ bypasses decorator wrappers (Eg, @traceable adds *, config=None)
    original_function = getattr(tool_function, "__wrapped__", tool_function)
    signature = inspect.signature(original_function)
    docstring = inspect.getdoc(original_function)
    descriptions.append(f"{tool_name}{signature} - {docstring}")
  return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - You must follow these rules extactly:
1. NEVER guess or assume any product prices. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price - do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always apply the discount using the apply_discount tool.
4. If the user does not specify a discount tier, ask them which discount tier to use - do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:
{tool_descriptions}

Use the following format:

Question: the input queston you must answer
Thought: you should always think about what to do
Action: the action to take, of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

# CHANGE 4: Drop tools= from ollama.chat(). The LLM has no idea its an agent -
# all agency comes from the prompt above and our regex parsing below.

@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
  return ollama.chat(
    model=MODEL,
    messages=messages,
    options=options,
  )



# ---- Agent Loop ----

@traceable(name="Langchain Agent Loop")

def run_agent(query: str) -> str:
  

  print(f"Query: {query}")
  print("="*len(query+"Query: ")+"\n\n")

  #CHANGE 5 : One prompt string replaces the system/user message split
  prompt = react_prompt.format(question=query)
  scratchpad = ""

  for iteration in range(1, MAX_ITERATIONS+1):
    print(f"--- Iteration {iteration}")
    full_prompt = prompt + scratchpad
    
    # Stop token prevents the LLM from generating its own Observation -
    # We inject the real tool result instead.
    response = ollama_chat_traced(
      model=MODEL,
      messages=[{"role": "user", "content": full_prompt}],
      options={"stop": ["\nObservation"], "temperature": 0.0},
    )
    output = response.message.content
    print(f"LLM Output:\n{output}")

    print(f" [Parsing] Looking for Final Answer in LLM Output...")
    final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
    if final_answer_match:
      final_answer = final_answer_match.group(1).strip()
      print("\n" + "=" * 60)
      print(f"\nFinal response: {final_answer}")
      return final_answer

    # Process only the FIRST tool call - force one tool per iteration
    print(f" [Parsing] Looking for Action and Action Input in LLM Output...")


    action_match = re.search(r"Action:\s*(.+)", output)
    action_input_match = re.search(r"Action Input:\s*(.+)", output)

    if not action_match or not action_input_match:
      print(
        " [Parsing] ERROR: No Action or Action Input found in LLM Output."
      )
      break


    tool_name = action_match.group(1).strip()
    tool_input_raw = action_input_match.group(1).strip()
    
    print(f" [Tool Selected] {tool_name} with args: {tool_input_raw}")

    # Split comma-separated args; strip key= prefix if LLM outputs ke=value format
    raw_args = [x.strip() for x in tool_input_raw.split(",")]
    args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

    print(f" [Tool Executing] {tool_name}({args})...")
    if tool_name not in tools:
      observation = f"Error: tool '{tool_name}' not found. Valid tools: {list[str](tools.keys())}"
    else:
      observation = tools[tool_name](*args)




    print(f" [Tool Result] {observation}")

    # CHANGE 7: History is one growing string re-sent every iteration (replaces messages.append)
    scratchpad += f"{output}\nObservation: {observation}\nThought:"

  print("ERROR: Max iterations reached without a final response.")
  return None


# ---- Main ----
if __name__ == "__main__":
  print("Hello Langchain Agent (.bind_tools()) calling tools...")
  print()
  result = run_agent("What is the price of a laptop after applying a gold discount?")