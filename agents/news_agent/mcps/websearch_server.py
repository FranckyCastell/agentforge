from mcp.server.fastmcp import FastMCP
from duckduckgo_search import DDGS
import json

server = FastMCP("websearch")


@server.tool(
    description="Search the web using DuckDuckGo. Returns title, URL, and snippet.",
)
def web_search(query: str, max_results: int = 10) -> str:
    ddgs = DDGS()
    results = list(ddgs.text(query, max_results=min(max_results, 20)))
    if not results:
        return "[]"
    output = [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in results
    ]
    return json.dumps(output, ensure_ascii=False)


@server.tool(
    description="Search for recent news using DuckDuckGo. Returns title, URL, and snippet.",
)
def news_search(query: str, max_results: int = 10) -> str:
    ddgs = DDGS()
    results = list(ddgs.news(query, max_results=min(max_results, 20)))
    if not results:
        return "[]"
    output = [
        {"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("body", "")}
        for r in results
    ]
    return json.dumps(output, ensure_ascii=False)


if __name__ == "__main__":
    server.run(transport="stdio")
