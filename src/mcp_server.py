# mcp_server.py
import os, traceback
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ChemIntelliGraph MCP Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

MCP_SECRET = os.getenv("MCP_SECRET")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY") or os.getenv("SERPAPI_KEY")

class WebSearchRequest(BaseModel):
    query: str
    top_k: int = 3

def verify_mcp_key(request: Request):
    header = request.headers.get("X-API-Key")
    if header != MCP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid X-API-Key header")
    return True

@app.get("/")
async def root():
    return {"status": "MCP server running"}

@app.post("/tool/web_search")
async def web_search(req: WebSearchRequest, request: Request):
    # Header verify
    verify_mcp_key(request)
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI key not configured on MCP server.")
    try:
        params = {"engine": "google", "q": req.query.strip(), "api_key": SERPAPI_KEY}
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=15)
        if r.status_code != 200:
            # include error for debugging
            return {"error": f"SerpAPI returned {r.status_code}", "details": r.text}
        js = r.json()
        organic = js.get("organic_results", []) or []
        hits = []
        for item in organic[: req.top_k]:
            hits.append({
                "title": item.get("title"),
                "url": item.get("link") or item.get("source"),
                "snippet": item.get("snippet") or item.get("snippet_highlighted_words") or ""
            })
        return {"results": hits}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
