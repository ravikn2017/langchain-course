from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "gemma4:latest"

# -- LangChain @Tool decorator --
@tool
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


# -- LangChain @Tool decorator --
@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to the price and return the final price.
    Available discount tiers: gold, silver, bronze"""
    print(f"   >> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")
    discount_percentags={"gold": 23, "silver": 12, "bronze": 5}
    discount = discount_percentags.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# ---- Agent Loop ----

@traceable(name="Langchain Agent Loop")

def run_agent(query: str) -> str:
  tools = [get_product_price, apply_discount]
  tools_dict = {t.name: t for t in tools}

  llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
  llm_with_tools = llm.bind_tools(tools)

  print(f"Query: {query}")
  print("="*len(query+"Query: ")+"\n\n")

  messages = [
    SystemMessage(
      content=(
      "You are a helpful shopping assistant. "
      "You have access to a product catalog tool "
      "and a discount tool. \n\n"
      "STRICT RULES - You must follow these rules extactly:\n"
      "1. NEVER guess or assume any product prices."
      "You MUST call get_product_price first to get the real price. \n "
      "2. Only call apply_discount AFTER you have received"
      "a price from get_product_price. Pass the exact price"
      "returned by get_product_price - do NOT pass a made-up number. \n"
      "3. NEVER calculate discounts yourself using math."
      "Always apply the discount using the apply_discount tool."
      "4. If the user does not specify a discount tier,"
      "ask them which discount tier to use - do NOT assume one."
     )
     ),
    HumanMessage(content=query)
  ]

  for iteration in range(1, MAX_ITERATIONS+1):
    print(f"Iteration {iteration}")
    
    ai_message = llm_with_tools.invoke(messages)

    tool_calls = ai_message.tool_calls

    # If no further tool calls, this is our final response
    if not tool_calls:
      print(f"\nFinal response: {ai_message.content}")
      return ai_message.content

    # Process only the FIRST tool call - force one tool per iteration
    tool_call = tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")

    print(f" [Tool Selected] {tool_name} with args: {tool_args}")
    tool_to_use = tools_dict.get(tool_name)
    if not tool_to_use:
      raise ValueError(f"Tool {tool_name} not found in tools_dict")
    observation = tool_to_use.invoke(tool_args)

    print(f" [Tool Result] {observation}")

    messages.append(ai_message)
    messages.append(ToolMessage(
      content=str(observation),
      tool_call_id=tool_call_id
    ))

  print("ERROR: Max iterations reached without a final response.")
  return None

    
    
  response = llm_with_tools.invoke(messages)
  print(response.content)
  print()
  print("="*len(response.content+"Response: ")+"\n\n")
  return response.content


# ---- Main ----
if __name__ == "__main__":
  print("Hello Langchain Agent (.bind_tools()) calling tools...")
  print()
  result = run_agent("What is the price of a laptop after applying a gold discount?")