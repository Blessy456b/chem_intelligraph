# agents_orchestrator.py
"""
LangGraph-enabled orchestrator (strict-match enforcement).

Key behavior:
- Strict mode:
  - RetrievalAgent only accepts a retrieval hit if the retrieved text contains BOTH reactants
    as whole-word/formula matches (case-insensitive). Single-letter reactants (e.g. "s") require
    whole-word match or element-name match (e.g. "sulfur"), not substring matches.
  - If retrieval fails strict-match -> immediate "No known reaction found for exact reactants."
  - If retrieval passes strict-match -> FactCheck verifies using user's original input ("A + B"),
    and FactCheck requires at least one evidence document/result that mentions both reactants.
- Creative mode: retrieval miss -> research -> factcheck -> safety -> memory.
- Attempts to use langgraph; falls back to procedural execution if graph build/run fails.
"""
import os
import json
import requests
import re
from datetime import datetime
from typing import Dict, Any, List

# try langgraph
try:
    import langgraph
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

MCP_BASE = os.getenv("MCP_BASE", "http://127.0.0.1:8080/tool")
MCP_KEY = os.getenv("MCP_SECRET", "changeme")
SUMMARY_PATH = "./conversation_summary.json"


# ---------------- Utilities & Memory ----------------
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
        try:
            self.vdb.add_documents([{"content": path}])
        except Exception:
            pass
        return path

    def retrieve_similar(self, query, n=3):
        return self.vdb.search(query, n_results=n).get("documents", [])


# ---------------- Agents ----------------
class RetrievalAgent:
    """
    Strict retrieval: require both reactants present using exact-match rules.
    Single-letter reactants must match as whole word or element name (not substring).
    """

    # basic element full-name map (expandable)
    ELEMENT_NAMES = {
        "h": "hydrogen",
        "he": "helium",
        "li": "lithium",
        "be": "beryllium",
        "b": "boron",
        "c": "carbon",
        "n": "nitrogen",
        "o": "oxygen",
        "f": "fluorine",
        "ne": "neon",
        "na": "sodium",
        "mg": "magnesium",
        "al": "aluminum",
        "si": "silicon",
        "p": "phosphorus",
        "s": "sulfur",
        "cl": "chlorine",
        "k": "potassium",
        "ca": "calcium",
        "zn": "zinc",
    }

    def __init__(self, rag):
        self.rag = rag

    @staticmethod
    def _norm_formula(tok: str) -> str:
        # remove whitespace and common punctuation, lowercase
        if not tok:
            return ""
        return re.sub(r"[\s\-\_\(\)\[\],\.]+", "", tok).lower()

    @staticmethod
    def _whole_word_present(text: str, token: str) -> bool:
        if not token:
            return False
        # whole-word match, case-insensitive
        return bool(re.search(rf"\b{re.escape(token)}\b", text, flags=re.IGNORECASE))

    def _matches_token(self, text: str, token: str) -> bool:
        """
        Matching rules:
        - If token length <= 1 (single character like 's'), require whole-word match OR element-name whole-word match.
        - If token length > 1:
            - accept whole-word match (e.g., 'HCl' as word),
            - OR normalized-formula substring match in normalized text (e.g., 'hcl' in 'hcl+naoh...').
        """
        if not token:
            return False
        text = text or ""
        token_stripped = token.strip()
        if token_stripped == "":
            return False

        # whole-word check first (covers names, formulas appearing with separators)
        if self._whole_word_present(text, token_stripped):
            return True

        norm_token = self._norm_formula(token_stripped)
        norm_text = self._norm_formula(text)

        # single-letter reactant (e.g., "s", "n") => don't allow substring match (would match many words)
        if len(norm_token) <= 1:
            # element full-name match allowed (e.g., 's' -> 'sulfur')
            elem = self.ELEMENT_NAMES.get(norm_token)
            if elem and self._whole_word_present(text, elem):
                return True
            # else require whole-word match only (already checked), so return False
            return False

        # multi-character token: allow normalized substring match (formula-like)
        if norm_token and norm_token in norm_text:
            return True

        return False

    def run(self, query: str, reactant_a: str = None, reactant_b: str = None):
        """
        Returns dict:
          {"found": bool, "response": str, "matches": {"a": bool, "b": bool}, "note": optional}
        """
        response = self.rag.invoke(query)
        found_basic = bool(response and not response.strip().lower().startswith("no known reaction"))
        if not found_basic:
            return {"found": False, "response": response, "matches": {"a": False, "b": False}, "note": "empty or sentinel"}

        a_ok = self._matches_token(response, reactant_a) if reactant_a else False
        b_ok = self._matches_token(response, reactant_b) if reactant_b else False

        if a_ok and b_ok:
            return {"found": True, "response": response, "matches": {"a": a_ok, "b": b_ok}}
        else:
            note = "strict-mode mismatch: retrieved response does not contain both reactants as exact matches"
            return {"found": False, "response": response, "matches": {"a": a_ok, "b": b_ok}, "note": note}


class LLMResearchAgent:
    def __init__(self, rag):
        self.rag = rag

    def run(self, query: str):
        return {"candidates": self.rag.invoke(query)}


class FactCheckAgent:
    """
    Verifies using local vectordb and web search.
    In strict mode we require at least one evidence document/result that contains BOTH reactants
    under the same exact-match rules used by RetrievalAgent.
    """

    def __init__(self, mcp_base=MCP_BASE, mcp_key=MCP_KEY):
        self.url = f"{mcp_base}/web_search"
        self.headers = {"X-API-Key": mcp_key}

    def verify_with_web(self, text: str):
        try:
            payload = {"query": text, "top_k": 5}
            r = requests.post(self.url, json=payload, headers=self.headers, timeout=15)
            return r.json() if r.status_code == 200 else {"error": f"status {r.status_code}", "body": r.text}
        except Exception as e:
            return {"error": str(e)}

    def _evidence_contains_both(self, evidence_text: str, reactant_a: str, reactant_b: str) -> bool:
        """Use same strict matching rules as RetrievalAgent for evidence validation."""
        # reuse a lightweight local matcher
        matcher = RetrievalAgent(None)
        return matcher._matches_token(evidence_text, reactant_a) and matcher._matches_token(evidence_text, reactant_b)

    def run(self, search_query: str, reactant_a: str = None, reactant_b: str = None, strict_verify: bool = False):
        """
        search_query: string used for vectordb/web search (e.g., "A + B" or candidate text)
        reactant_a/b: original reactants for strict evidence checking
        strict_verify: if True, require at least one evidence doc that mentions both reactants
        Returns: dict {verified: bool, sources: [...], local_docs: [...], web_results: [...], search_query_used: ...}
        """
        from vectordb import VectorDB
        vdb = VectorDB()

        try:
            local_res = vdb.search(search_query, n_results=5)
        except Exception as e:
            local_res = {"error": str(e), "documents": []}

        web_res = self.verify_with_web(search_query)

        verified = False
        sources = []

        local_docs = local_res.get("documents", []) if isinstance(local_res, dict) else []
        web_results = web_res.get("results", []) if isinstance(web_res, dict) else []

        # If strict verification requested, ensure at least one doc/result contains both reactants
        if strict_verify and reactant_a and reactant_b:
            evidence_ok = False
            # check local docs
            for d in local_docs:
                # document representation may vary; try to get text fields
                doc_text = ""
                if isinstance(d, dict):
                    # flatten likely fields
                    doc_text = " ".join([str(v) for v in d.values() if isinstance(v, (str, int, float))])
                else:
                    doc_text = str(d)
                if self._evidence_contains_both(doc_text, reactant_a, reactant_b):
                    evidence_ok = True
                    sources.append("ChromaDB")
                    break
            # check web results if still not found
            if (not evidence_ok) and isinstance(web_results, list):
                for w in web_results:
                    w_text = ""
                    if isinstance(w, dict):
                        w_text = " ".join([str(v) for v in w.values() if isinstance(v, (str, int, float))])
                    else:
                        w_text = str(w)
                    if self._evidence_contains_both(w_text, reactant_a, reactant_b):
                        evidence_ok = True
                        sources.append("WebSearch")
                        break
            verified = evidence_ok
        else:
            # non-strict: any documents or web results count
            if local_docs:
                verified = True
                sources.append("ChromaDB")
            if web_results:
                verified = True
                if "WebSearch" not in sources:
                    sources.append("WebSearch")

        return {
            "verified": verified,
            "sources": sources,
            "local_docs": local_docs,
            "web_results": web_results,
            "search_query_used": search_query,
        }


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


# ---------------- Procedural fallback ----------------
def _procedural_run_multi_agent(reactant_a: str, reactant_b: str, mode="strict"):
    from rag_assistant import RAGAssistant

    query = f"{reactant_a} + {reactant_b}"
    trace: List[Dict[str, Any]] = []

    rag = RAGAssistant(mode=mode)
    retr_agent = RetrievalAgent(rag)
    r = retr_agent.run(query, reactant_a=reactant_a, reactant_b=reactant_b)
    trace.append({"agent": "RetrievalAgent", "result": r})

    # Strict mode: do NOT escalate to research if retrieval mismatch
    if mode == "strict":
        if not r.get("found"):
            return {
                "reaction": None,
                "verified": False,
                "message": "No known reaction found for exact reactants.",
                "trace": trace
            }

        # If retrieved matched both reactants: fact-check using user's original input (A + B)
        fc_agent = FactCheckAgent()
        fc = fc_agent.run(query, reactant_a=reactant_a, reactant_b=reactant_b, strict_verify=True)
        trace.append({"agent": "FactCheckAgent", "result": fc})

        if not fc.get("verified"):
            return {
                "reaction": r.get("response"),
                "verified": False,
                "message": "Retrieved reaction did not verify against local/web evidence for the provided reactants.",
                "trace": trace
            }

        # Safety on retrieved reaction text
        safe = SafetyAgent().run(r.get("response"), rag)
        trace.append({"agent": "SafetyAgent", "result": safe})

        hazard = safe.get("hazard_level", "Unknown")
        confidence = 0.98 if fc.get("verified") and hazard == "Low" else (0.7 if hazard != "High" else 0.2)

        VectorMemory().store_reaction(query, r.get("response"), confidence, hazard, fc.get("sources", []))
        append_summary({"query": query, "reaction": r.get("response"), "confidence": confidence, "hazard": hazard, "time": datetime.utcnow().isoformat(), "path": "retrieval_strict"})

        return {
            "reaction": r.get("response"),
            "confidence": confidence,
            "hazard": hazard,
            "sources": fc.get("sources", []),
            "verified": True,
            "trace": trace,
            "memory_stored": True
        }

    # Creative mode: retrieval hit -> verify candidate_text; miss -> research -> verify
    trace.append({"note": "creative-mode flow"})
    if r.get("found"):
        candidates = [r.get("response")]
    else:
        candidates = []

    if not candidates:
        research_rag = RAGAssistant(mode="creative")
        research_agent = LLMResearchAgent(research_rag)
        cand = research_agent.run(query)
        trace.append({"agent": "LLMResearchAgent", "result": cand})
        rc = cand.get("candidates")
        if isinstance(rc, str):
            candidates = [rc]
        elif isinstance(rc, list):
            candidates = rc
        else:
            candidates = [rc] if rc is not None else []

    if not candidates:
        return {"reaction": None, "message": "No candidate reaction found.", "trace": trace}

    # factcheck candidate(s) (non-strict)
    fact_agent = FactCheckAgent()
    cand_text = "\n\n".join(candidates)
    fc = fact_agent.run(cand_text, strict_verify=False)
    trace.append({"agent": "FactCheckAgent", "result": fc})

    if not fc.get("verified"):
        return {"reaction": candidates, "verified": False, "trace": trace, "message": "Candidate not verified."}

    safe = SafetyAgent().run(candidates[0], RAGAssistant(mode="creative"))
    trace.append({"agent": "SafetyAgent", "result": safe})

    hazard = safe.get("hazard_level", "Unknown")
    confidence = 0.7 if hazard != "High" else 0.2

    mem_path = VectorMemory().store_reaction(query, candidates[0], confidence, hazard, fc.get("sources", []))
    append_summary({"query": query, "reaction": candidates[0], "confidence": confidence, "hazard": hazard, "time": datetime.utcnow().isoformat(), "memory_path": mem_path})

    return {"reaction": candidates[0], "confidence": confidence, "sources": fc.get("sources", []), "hazard": hazard, "trace": trace, "memory_stored": True}


# ---------------- LangGraph orchestration attempt ----------------
def _langgraph_run_multi_agent(reactant_a: str, reactant_b: str, mode="strict", force_escalate: bool = False):
    """
    Build and run a small callable-node graph with langgraph.
    Falls back to throwing if graph APIs differ; caller will fallback to procedural.
    """
    from rag_assistant import RAGAssistant

    def node_retrieval(c: Dict[str, Any]) -> Dict[str, Any]:
        rag = RAGAssistant(mode=c.get("mode", mode))
        retr = RetrievalAgent(rag)
        parts = [p.strip() for p in c["query"].split("+")]
        a = parts[0] if parts else c.get("reactant_a")
        b = parts[1] if len(parts) > 1 else c.get("reactant_b")
        r = retr.run(c["query"], reactant_a=a, reactant_b=b)
        c["trace"].append({"agent": "RetrievalAgent", "result": r})
        c["retrieval"] = r
        return c

    def node_decision_and_research(c: Dict[str, Any]) -> Dict[str, Any]:
        r = c.get("retrieval", {})
        if c.get("mode") == "strict":
            if not r.get("found"):
                c["result"] = {"reaction": None, "verified": False, "message": "No known reaction found for exact reactants.", "trace": c["trace"]}
                return c
            c["candidates"] = [r.get("response")]
            return c

        # creative mode behavior
        if r.get("found"):
            c["candidates"] = [r.get("response")]
            if c.get("force_escalate", False):
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

    def node_factcheck(c: Dict[str, Any]) -> Dict[str, Any]:
        fc_agent = FactCheckAgent()
        if c.get("mode") == "strict":
            # verify using the user's original query AND strict evidence check
            fc = fc_agent.run(c.get("query"), reactant_a=c.get("reactant_a"), reactant_b=c.get("reactant_b"), strict_verify=True)
        else:
            cand = c.get("candidates") or []
            cand_text = "\n\n".join([str(x) for x in cand if x])
            fc = fc_agent.run(cand_text, strict_verify=False)
        c["trace"].append({"agent": "FactCheckAgent", "result": fc})
        c["factcheck"] = fc
        return c

    def node_safety(c: Dict[str, Any]) -> Dict[str, Any]:
        cand = c.get("candidates") or []
        cand_text = cand[0] if isinstance(cand, (list, tuple)) and cand else (cand if isinstance(cand, str) else "")
        research_rag = RAGAssistant(mode="creative")
        safe = SafetyAgent().run(cand_text, research_rag)
        c["trace"].append({"agent": "SafetyAgent", "result": safe})
        c["safety"] = safe
        return c

    def node_finalize(c: Dict[str, Any]) -> Dict[str, Any]:
        if c.get("mode") == "strict" and c.get("retrieval", {}).get("found") is False:
            if "result" not in c:
                c["result"] = {"reaction": None, "verified": False, "message": "No known reaction found for exact reactants.", "trace": c["trace"]}
            return c

        fc = c.get("factcheck", {})
        safe = c.get("safety", {})
        candidates = c.get("candidates")
        if not fc.get("verified"):
            c["result"] = {"reaction": candidates, "verified": False, "trace": c["trace"], "message": "Candidate not verified."}
            return c

        main = candidates[0] if isinstance(candidates, (list, tuple)) and candidates else candidates
        hazard = safe.get("hazard_level", "Unknown")
        confidence = 0.98 if fc.get("verified") and hazard == "Low" else (0.7 if hazard != "High" else 0.2)
        vmem = VectorMemory()
        mem_path = vmem.store_reaction(c["query"], main, confidence, hazard, fc.get("sources", []))
        append_summary({"query": c["query"], "reaction": main, "confidence": confidence, "hazard": hazard, "time": datetime.utcnow().isoformat(), "memory_path": mem_path})
        c["result"] = {"reaction": main, "all_candidates": candidates, "confidence": confidence, "sources": fc.get("sources", []), "hazard": hazard, "trace": c["trace"], "memory_stored": True}
        return c

    # Build initial context
    ctx: Dict[str, Any] = {"query": f"{reactant_a} + {reactant_b}", "mode": mode, "trace": [], "result": {}, "force_escalate": force_escalate, "reactant_a": reactant_a, "reactant_b": reactant_b}

    # Attempt a conservative langgraph build & run
    try:
        G = langgraph.Graph()
        G.add_node("retrieval", func=node_retrieval)
        G.add_node("decision", func=node_decision_and_research)
        G.add_node("factcheck", func=node_factcheck)
        G.add_node("safety", func=node_safety)
        G.add_node("finalize", func=node_finalize)

        G.add_edge("retrieval", "decision")
        G.add_edge("decision", "factcheck")
        G.add_edge("factcheck", "safety")
        G.add_edge("safety", "finalize")

        result_ctx = G.run(ctx)
        if isinstance(result_ctx, dict) and "result" in result_ctx:
            return result_ctx["result"]
        return result_ctx
    except Exception:
        # try alternate API shape or fail up to caller
        try:
            G = langgraph.SimpleGraph()
            G.add("retrieval", node_retrieval)
            G.add("decision", node_decision_and_research)
            G.add("factcheck", node_factcheck)
            G.add("safety", node_safety)
            G.add("finalize", node_finalize)
            G.link("retrieval", "decision")
            G.link("decision", "factcheck")
            G.link("factcheck", "safety")
            G.link("safety", "finalize")
            result_ctx = G.execute(ctx)
            if isinstance(result_ctx, dict) and "result" in result_ctx:
                return result_ctx["result"]
            return result_ctx
        except Exception as e:
            # surface to caller to fallback to procedural
            raise RuntimeError("langgraph graph construction/execution failed") from e


def run_multi_agent(reactant_a: str, reactant_b: str, mode="strict", force_escalate: bool = False) -> Dict[str, Any]:
    """
    Main entrypoint: try langgraph first, fallback procedurally.
    """
    if LANGGRAPH_AVAILABLE:
        try:
            return _langgraph_run_multi_agent(reactant_a, reactant_b, mode=mode, force_escalate=force_escalate)
        except Exception:
            return _procedural_run_multi_agent(reactant_a, reactant_b, mode=mode)
    return _procedural_run_multi_agent(reactant_a, reactant_b, mode=mode)


# quick smoke when run as script
if __name__ == "__main__":
    import json
    print("LANGGRAPH_AVAILABLE =", LANGGRAPH_AVAILABLE)
    print(json.dumps(run_multi_agent("s", "HCl", mode="strict"), indent=2))
    print(json.dumps(run_multi_agent("Zn", "HCl", mode="creative"), indent=2))
