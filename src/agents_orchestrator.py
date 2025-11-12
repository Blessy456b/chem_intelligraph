
"""
LangGraph-aware orchestrator (full verification flow).

Behaviors:
- RetrievalAgent -> FactCheckAgent -> SafetyAgent -> Memory (when retrieval hits)
- RetrievalAgent -> LLMResearchAgent -> FactCheckAgent -> SafetyAgent -> Memory (when retrieval misses)
- force_escalate=True forces research+factcheck+safety even if retrieval hits.
- If langgraph is installed, we print a detection message but run nodes sequentially
  to avoid runtime write semantics differences between langgraph versions.
- Falls back to procedural implementation on any unexpected exception.
"""
import os
import json
import requests
from datetime import datetime
from typing import Dict, Any, List

# constants (same as previous)
MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/tool")
MCP_KEY = os.getenv("MCP_SECRET", "changeme")
SUMMARY_PATH = "./conversation_summary.json"


# ----------------- Utilities & Memory -----------------
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
    def __init__(self, collection_name="memory_reactions"):
        from vectordb import VectorDB
        self.vdb = VectorDB(collection_name=collection_name)

    def store_reaction(self, reaction_query, verified_reaction, confidence, safety_status, sources):
        import os
        os.makedirs("./memory_docs", exist_ok=True)
        fname = f"mem_{abs(hash(reaction_query))}_{int(datetime.utcnow().timestamp())}.txt"
        path = os.path.join("./memory_docs", fname)
        content = (
            f"Query: {reaction_query}\n"
            f"Reaction: {verified_reaction}\n"
            f"Confidence: {confidence}\n"
            f"Safety: {safety_status}\n"
            f"Sources: {', '.join(sources)}"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        self.vdb.add_documents([{"content": path}])
        return path

    def retrieve_similar(self, query, n=3):
        return self.vdb.search(query, n_results=n).get("documents", [])


# ----------------- Agents -----------------
class RetrievalAgent:
    def __init__(self, rag):
        self.rag = rag

    def run(self, query: str):
        response = self.rag.invoke(query)
        found = not (not response or response.strip().lower().startswith("no known reaction"))
        return {"found": found, "response": response}


class LLMResearchAgent:
    def __init__(self, rag):
        self.rag = rag

    def run(self, query: str):
        return {"candidates": self.rag.invoke(query)}


class FactCheckAgent:
    def __init__(self, mcp_base=MCP_BASE, mcp_key=MCP_KEY):
        self.url = f"{mcp_base}/web_search"
        self.headers = {"X-API-Key": mcp_key}

    def verify_with_web(self, text):
        try:
            payload = {"query": text, "top_k": 5}
            r = requests.post(self.url, json=payload, headers=self.headers, timeout=15)
            return r.json()
        except Exception as e:
            return {"error": str(e)}

    def run(self, candidate_text, local_check_query):
        from vectordb import VectorDB
        vdb = VectorDB()
        local_res = vdb.search(local_check_query, n_results=3)
        web_res = self.verify_with_web(candidate_text)
        verified = False
        sources = []
        if local_res.get("documents"):
            verified = True
            sources.append("ChromaDB")
        if web_res.get("results"):
            verified = True
            sources.append("WebSearch")
        return {"verified": verified, "sources": sources, "local_docs": local_res.get("documents", []), "web_results": web_res.get("results", [])}


class SafetyAgent:
    def __init__(self):
        self.hazard_map = {
            "explosive": ["nitroglycerin", "tnt", "hmx", "rdx"],
            "toxic_gas": ["chlorine", "phosgene", "hydrogen sulfide", "h2s", "hcl"],
            "highly_reactive": ["sodium", "potassium", "fluorine"],
        }

    def basic_rule_check(self, reaction_text):
        text = (reaction_text or "").lower()
        flags = []
        for k, toks in self.hazard_map.items():
            for t in toks:
                if t in text:
                    flags.append(k)
        hazard = "Low"
        if "explosive" in flags or "toxic_gas" in flags:
            hazard = "High"
        elif flags:
            hazard = "Medium"
        return {"flags": flags, "hazard_level": hazard}

    def run(self, reaction_text, rag_assistant):
        rules = self.basic_rule_check(reaction_text)
        prompt = (
            "You are a chemistry safety checker. "
            "Given the reaction below, return JSON: {hazard_level, reasons, recommendation}.\n\n"
            f"Reaction:\n{reaction_text}\n"
        )
        try:
            eval_text = rag_assistant.invoke(prompt)
            parsed = json.loads(eval_text) if eval_text.strip().startswith("{") else {"llm_eval": eval_text}
        except Exception as e:
            parsed = {"error": str(e)}
        hazard = parsed.get("hazard_level", rules["hazard_level"]) if isinstance(parsed, dict) else rules["hazard_level"]
        return {"hazard_level": hazard, "flags": rules["flags"], "llm_eval": parsed}


# ----------------- Procedural fallback (unchanged behavior) -----------------
def _procedural_run_multi_agent(reactant_a: str, reactant_b: str, mode="strict"):
    from rag_assistant import RAGAssistant

    query = f"{reactant_a} + {reactant_b}"
    trace: List[Dict[str, Any]] = []

    rag = RAGAssistant(mode=mode)
    retr_agent = RetrievalAgent(rag)
    r = retr_agent.run(query)
    trace.append({"agent": "RetrievalAgent", "result": r})

    # If retrieval found, still run factcheck & safety for verification (improved behavior)
    if r.get("found"):
        # factcheck on retrieved
        fc_agent = FactCheckAgent()
        fc = fc_agent.run(r.get("response"), query)
        trace.append({"agent": "FactCheckAgent", "result": fc})

        # safety check
        safety_agent = SafetyAgent()
        safe = safety_agent.run(r.get("response"), rag)
        trace.append({"agent": "SafetyAgent", "result": safe})

        # finalize confidence/hazard
        hazard = safe.get("hazard_level", "Unknown")
        confidence = 0.98 if fc.get("verified") else (0.7 if hazard != "High" else 0.2)

        final = {"reaction": r.get("response"), "confidence": confidence, "sources": fc.get("sources", ["ChromaDB"]), "hazard": hazard, "trace": trace, "memory_stored": True}
        VectorMemory().store_reaction(query, r.get("response"), final["confidence"], final["hazard"], final["sources"])
        append_summary({"query": query, "reaction": r.get("response"), "confidence": confidence, "hazard": hazard, "time": datetime.utcnow().isoformat(), "path": "retrieval"})
        return final

    # If retrieval missed, escalate to research + verification
    trace.append({"note": "RAG miss — escalate to LLMResearchAgent"})
    research_rag = RAGAssistant(mode="creative")
    research_agent = LLMResearchAgent(research_rag)
    cand = research_agent.run(query)
    trace.append({"agent": "LLMResearchAgent", "result": cand})

    if not cand.get("candidates"):
        return {"reaction": None, "trace": trace, "message": "No candidate reaction found."}

    fact_agent = FactCheckAgent()
    fc = fact_agent.run(cand["candidates"], query)
    trace.append({"agent": "FactCheckAgent", "result": fc})

    if not fc.get("verified"):
        return {"reaction": cand["candidates"], "verified": False, "trace": trace, "message": "Candidate not verified."}

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


# ----------------- LangGraph-aware run_multi_agent (always verify retrieved results) -----------------
# Detect langgraph presence (we only use this for detection/logging; nodes run sequentially)
try:
    import langgraph  # presence detection only
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False


def run_multi_agent(reactant_a: str, reactant_b: str, mode="strict", force_escalate: bool = False) -> Dict[str, Any]:
    """
    Main entrypoint. Always verifies retrieved results (factcheck + safety).
    If retrieval misses, runs research -> factcheck -> safety.
    If langgraph is installed, prints detection but executes nodes sequentially
    (to avoid version-specific runtime write semantics).
    """
    # Fallback to procedural if something goes wrong
    try:
        from rag_assistant import RAGAssistant

        # If langgraph not installed, we still execute the same verified flow using procedural function
        if not LANGGRAPH_AVAILABLE:
            # For backwards compatibility, call the procedural function (which itself now verifies retrievals)
            return _procedural_run_multi_agent(reactant_a, reactant_b, mode=mode)

        # Langgraph present (informational)
        print("✅ LangGraph detected — using graph-based orchestration (sequential nodes).")

        query = f"{reactant_a} + {reactant_b}"
        ctx: Dict[str, Any] = {"query": query, "mode": mode, "trace": [], "result": {}, "force_escalate": force_escalate}

        # --- Node: retrieval ---
        def node_retrieval(c: Dict[str, Any]) -> Dict[str, Any]:
            rag = RAGAssistant(mode=c.get("mode", mode))
            retr = RetrievalAgent(rag)
            r = retr.run(c["query"])
            c["trace"].append({"agent": "RetrievalAgent", "result": r})
            c["retrieval"] = r
            return c

        # --- Node: decision/research orchestration (we will always verify retrieved results) ---
        def node_decision_and_research(c: Dict[str, Any]) -> Dict[str, Any]:
            r = c.get("retrieval", {})
            # If retrieval found and we are NOT forcing escalation, we WILL STILL run FactCheck & Safety,
            # but we can skip LLMResearchAgent unless force_escalate True or retrieval is insufficient.
            if r.get("found"):
                c["trace"].append({"note": "Retrieval hit — scheduling FactCheck & Safety for verification."})
                # seed candidates with retrieval response
                c["candidates"] = [r.get("response")]
                # If force_escalate is True, we will also run research and append its candidates
                if c.get("force_escalate", False):
                    c["trace"].append({"note": "force_escalate True — also running research to gather extra candidates."})
                    research_rag = RAGAssistant(mode="creative")
                    research_agent = LLMResearchAgent(research_rag)
                    cand = research_agent.run(c["query"])
                    c["trace"].append({"agent": "LLMResearchAgent", "result": cand})
                    rc = cand.get("candidates")
                    if isinstance(rc, str):
                        rc_list = [rc]
                    elif isinstance(rc, list):
                        rc_list = rc
                    else:
                        rc_list = [rc] if rc is not None else []
                    c["candidates"] = (c.get("candidates") or []) + rc_list
                return c

            # Retrieval miss -> run research always
            c["trace"].append({"note": "RAG miss — escalate to LLMResearchAgent."})
            research_rag = RAGAssistant(mode="creative")
            research_agent = LLMResearchAgent(research_rag)
            cand = research_agent.run(c["query"])
            c["trace"].append({"agent": "LLMResearchAgent", "result": cand})
            rc = cand.get("candidates")
            if isinstance(rc, str):
                rc_list = [rc]
            elif isinstance(rc, list):
                rc_list = rc
            else:
                rc_list = [rc] if rc is not None else []
            c["candidates"] = rc_list
            return c

        # --- Node: factcheck (always runs) ---
        def node_factcheck(c: Dict[str, Any]) -> Dict[str, Any]:
            candidates = c.get("candidates") or []
            # create candidate_text for fact checking
            candidate_text = "\n\n".join([str(x) for x in candidates if x])
            fact_agent = FactCheckAgent()
            fc = fact_agent.run(candidate_text, c.get("query"))
            c["trace"].append({"agent": "FactCheckAgent", "result": fc})
            c["factcheck"] = fc
            return c

        # --- Node: safety (always runs after factcheck) ---
        def node_safety(c: Dict[str, Any]) -> Dict[str, Any]:
            candidates = c.get("candidates") or []
            candidate_text = "\n\n".join([str(x) for x in candidates if x])
            research_rag = RAGAssistant(mode="creative")
            safe = SafetyAgent().run(candidate_text, research_rag)
            c["trace"].append({"agent": "SafetyAgent", "result": safe})
            c["safety"] = safe
            return c

        # --- Node: finalize & memory ---
        def node_finalize(c: Dict[str, Any]) -> Dict[str, Any]:
            fc = c.get("factcheck", {})
            safe = c.get("safety", {})
            candidates = c.get("candidates")
            # If factcheck didn't verify any candidate, return unverified list
            if not fc.get("verified"):
                c["result"] = {"reaction": candidates, "verified": False, "trace": c["trace"], "message": "Candidate not verified."}
                return c
            # Choose a main candidate (first) for storing & display
            main = None
            if isinstance(candidates, (list, tuple)) and candidates:
                main = candidates[0]
            else:
                main = candidates
            hazard = safe.get("hazard_level", "Unknown")
            confidence = 0.98 if fc.get("verified") and hazard == "Low" else (0.7 if hazard != "High" else 0.2)
            vmem = VectorMemory()
            mem_path = vmem.store_reaction(c["query"], main, confidence, hazard, fc.get("sources", []))
            append_summary({"query": c["query"], "reaction": main, "confidence": confidence, "hazard": hazard, "time": datetime.utcnow().isoformat(), "memory_path": mem_path})
            c["result"] = {"reaction": main, "all_candidates": candidates, "confidence": confidence, "sources": fc.get("sources", []), "hazard": hazard, "trace": c["trace"], "memory_stored": True}
            return c

        # Execute nodes sequentially
        ctx = node_retrieval(ctx)
        ctx = node_decision_and_research(ctx)
        ctx = node_factcheck(ctx)
        ctx = node_safety(ctx)
        ctx = node_finalize(ctx)

        return ctx.get("result", {})

    except Exception as e:
        # On any unexpected error, fallback to the procedural implementation to keep app functional.
        print("⚠️ LangGraph-run failed — falling back to procedural implementation. Error:", repr(e))
        return _procedural_run_multi_agent(reactant_a, reactant_b, mode=mode)


# allow running as a quick smoke test
if __name__ == "__main__":
    import json
    print(json.dumps(run_multi_agent("Zn", "HCl", mode="strict"), indent=2))
