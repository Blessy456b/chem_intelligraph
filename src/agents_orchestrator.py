# agents_orchestrator.py
import os, json, requests
from datetime import datetime
from typing import Dict, Any, List
from rag_assistant import RAGAssistant
from vectordb import VectorDB
from dotenv import load_dotenv

load_dotenv()

MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/tool")
MCP_KEY = os.getenv("MCP_SECRET", "changeme")
SUMMARY_PATH = "./conversation_summary.json"

def load_summary_memory():
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def append_summary(item: Dict[str, Any]):
    mem = load_summary_memory()
    mem.append(item)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(mem, f, indent=2, ensure_ascii=False)

class VectorMemory:
    def __init__(self, collection_name: str = "memory_reactions"):
        self.vdb = VectorDB(collection_name=collection_name)

    def store_reaction(self, reaction_query: str, verified_reaction: str, confidence: float, safety_status: str, sources: List[str]):
        os.makedirs("./memory_docs", exist_ok=True)
        fname = f"mem_{abs(hash(reaction_query))}_{int(datetime.utcnow().timestamp())}.txt"
        path = os.path.join("./memory_docs", fname)
        content = f"Query: {reaction_query}\nReaction: {verified_reaction}\nConfidence: {confidence}\nSafety: {safety_status}\nSources: {', '.join(sources)}"
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.vdb.add_documents([{"content": path}])
        return path

    def retrieve_similar(self, query: str, n=3):
        return self.vdb.search(query, n_results=n).get("documents", [])

# Agents
class RetrievalAgent:
    def __init__(self, rag: RAGAssistant):
        self.rag = rag

    def run(self, query: str) -> Dict[str, Any]:
        response = self.rag.invoke(query)
        found = not (not response or response.strip().lower().startswith("no known reaction"))
        return {"found": found, "response": response}

class LLMResearchAgent:
    def __init__(self, rag: RAGAssistant):
        self.rag = rag

    def run(self, query: str) -> Dict[str, Any]:
        return {"candidates": self.rag.invoke(query)}

class FactCheckAgent:
    def __init__(self, mcp_base: str = MCP_BASE, mcp_key: str = MCP_KEY):
        self.url = f"{mcp_base}/web_search"
        self.headers = {"X-API-Key": mcp_key}

    def verify_with_web(self, text: str) -> Dict[str, Any]:
        try:
            payload = {"query": text, "top_k": 5}
            r = requests.post(self.url, json=payload, headers=self.headers, timeout=15)
            if r.status_code != 200:
                return {"error": f"mcp returned {r.status_code}", "details": r.text}
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def run(self, candidate_text: str, local_check_query: str) -> Dict[str, Any]:
        vdb = VectorDB()
        local_res = vdb.search(local_check_query, n_results=3)
        web_res = self.verify_with_web(candidate_text)
        verified = False
        sources = []
        if local_res.get("documents"):
            verified = True
            sources.append("ChromaDB")
        if isinstance(web_res, dict) and web_res.get("results"):
            verified = True
            sources.append("WebSearch")
        return {"verified": verified, "sources": sources, "local_docs": local_res.get("documents", []), "web_results": web_res.get("results", []) if isinstance(web_res, dict) else []}

class SafetyAgent:
    def __init__(self):
        self.hazard_map = {
            "explosive": ["nitroglycerin", "tnt", "hmx", "rdx"],
            "toxic_gas": ["chlorine", "phosgene", "hydrogen sulfide", "h2s", "hcl"],
            "highly_reactive": ["sodium", "potassium", "fluorine"],
        }

    def basic_rule_check(self, reaction_text: str) -> Dict[str, Any]:
        text = (reaction_text or "").lower()
        flags = []
        for k, toks in self.hazard_map.items():
            for t in toks:
                if t in text:
                    flags.append(k)
        hazard_level = "Low"
        if "explosive" in flags or "toxic_gas" in flags:
            hazard_level = "High"
        elif flags:
            hazard_level = "Medium"
        return {"flags": flags, "hazard_level": hazard_level}

    def run(self, reaction_text: str, rag_assistant: RAGAssistant) -> Dict[str, Any]:
        rules = self.basic_rule_check(reaction_text)
        prompt = (
            "You are a chemistry safety checker. Given the reaction below, "
            "return a JSON object with keys: hazard_level (Low/Medium/High), reasons (short list), recommendation.\n\n"
            f"Reaction:\n{reaction_text}\n\nAnswer:"
        )
        try:
            eval_text = rag_assistant.invoke(prompt)
            try:
                parsed = json.loads(eval_text)
            except Exception:
                parsed = {"llm_eval": eval_text}
        except Exception as e:
            parsed = {"error": str(e)}
        hazard_level = parsed.get("hazard_level", rules["hazard_level"]) if isinstance(parsed, dict) else rules["hazard_level"]
        return {"hazard_level": hazard_level, "flags": rules["flags"], "llm_eval": parsed}

# Orchestrator
def run_multi_agent(reactant_a: str, reactant_b: str, mode: str = "strict") -> Dict[str, Any]:
    query = f"{reactant_a} + {reactant_b}"
    trace = []

    rag = RAGAssistant(mode=mode)
    retr_agent = RetrievalAgent(rag)
    r = retr_agent.run(query)
    trace.append({"agent": "RetrievalAgent", "result": r})

    # Strict mode: do not escalate to LLM or web — only retrieval allowed
    if mode == "strict":
        if r["found"]:
            final = {"reaction": r["response"], "confidence": 0.98, "sources": ["ChromaDB"], "trace": trace, "hazard": "Unknown"}
            vmem = VectorMemory()
            vmem.store_reaction(query, r["response"], final["confidence"], final["hazard"], final["sources"])
            append_summary({"query": query, "reaction": r["response"], "time": datetime.utcnow().isoformat(), "path": "retrieval"})
            final["memory_stored"] = True
            return final
        return {"reaction": None, "trace": trace, "message": "No known reaction found in knowledge base (Strict Mode)."}

    # Creative mode: escalate and verify
    trace.append({"note": "RAG miss — escalate to LLMResearchAgent"})
    research_rag = RAGAssistant(mode="creative")
    research_agent = LLMResearchAgent(research_rag)
    cand = research_agent.run(query)
    trace.append({"agent": "LLMResearchAgent", "result": cand})

    if not cand.get("candidates"):
        return {"reaction": None, "trace": trace, "message": "No candidate produced by LLMResearchAgent"}

    fact_agent = FactCheckAgent()
    fc = fact_agent.run(cand["candidates"], query)
    trace.append({"agent": "FactCheckAgent", "result": fc})

    if not fc.get("verified"):
        return {"reaction": cand["candidates"], "verified": False, "trace": trace, "message": "Candidate not verified by local DB or web"}

    safety_agent = SafetyAgent()
    safe = safety_agent.run(cand["candidates"], research_rag)
    trace.append({"agent": "SafetyAgent", "result": safe})

    hazard = safe.get("hazard_level", "Unknown")
    confidence = 0.7 if hazard != "High" else 0.2

    vmem = VectorMemory()
    mem_path = vmem.store_reaction(query, cand["candidates"], confidence, hazard, fc.get("sources", []))
    append_summary({"query": query, "reaction": cand["candidates"], "confidence": confidence, "hazard": hazard, "time": datetime.utcnow().isoformat(), "memory_path": mem_path})

    final = {"reaction": cand["candidates"], "confidence": confidence, "sources": fc.get("sources", []), "hazard": hazard, "trace": trace, "memory_stored": True}
    return final

if __name__ == "__main__":
    print(json.dumps(run_multi_agent("Zn", "HCl", mode="strict"), indent=2))
