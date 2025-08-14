from flask import Flask, request, jsonify
from processor import load_all_documents, split_into_chunks, build_faiss_index, load_faiss_index, rag_chain
from pathlib import Path
import os, json, re, traceback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

index_cache = {}

AGENT_DEFINITIONS = {
    "HYPE":   "HYPE (Marketing Intelligence Agent): Turns marketing into a revenue engine via automation and measurement.",
    "STRIKE": "STRIKE (Sales Acceleration Agent): Scales relationship-led selling via pipeline intelligence and outreach automation.",
    "CARE":   "CARE (Customer Experience Agent): Delivers 24/7 omnichannel support with human escalation.",
    "VISION": "VISION (Strategic Intelligence Agent): Provides exec-level insight for decisions.",
    "FLOW":   "FLOW (Operations Excellence Agent): Orchestrates back-office processes.",
    "ASSET":  "ASSET (Financial Intelligence Agent): Automates finance ops and planning.",
    "TEAM":   "TEAM (Human Capital Agent): Optimizes talent lifecycle.",
    "CODE":   "CODE (Technology Intelligence Agent): Provides integration & data platform."
}

GUARDRAILS = r"""
Select ONLY from: HYPE, STRIKE, CARE, VISION, FLOW, ASSET, TEAM, CODE.
- Evidence-gated: Include an agent ONLY if supported by retrieved text.
- Uniqueness: One capability belongs to one owner.
- Output strict JSON: { "pain_points":[], "agents":[], "rationale":{...} }
"""

def _safe_json_extract(text: str):
    try:
        return json.loads(text)
    except Exception:
        try:
            s, e = text.find("{"), text.rfind("}")
            return json.loads(text[s:e+1]) if (s!=-1 and e!=-1 and e>s) else None
        except Exception:
            return None

def strip_meta(text: str) -> str:
    t = text or ""
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    return "\n".join(
        ln for ln in t.splitlines()
        if not ln.strip().lower().startswith("sources:")
    ).strip()

FORBIDDEN_BY_AGENT = {
    "HYPE":   ["reconciliation","invoice","claims","ar/ap","cash flow","24/7","chatbot","ticket","knowledge base","data platform"],
    "STRIKE": ["reconciliation","invoice","claims","ar/ap","cash flow","24/7","chatbot","feedback","knowledge base","data platform"],
    "CARE":   ["reconciliation","invoice","claims","ar/ap","cash flow","pipeline","proposal","ab testing","data platform"],
    "VISION": ["reconciliation","invoice","claims","ar/ap","24/7","chatbot","feedback","data platform ownership"],
    "FLOW":   ["reconciliation","invoice","claims","24/7","chatbot","feedback","emr extraction"],
    "ASSET":  ["24/7","chatbot","feedback","knowledge base","pipeline","ab testing","data platform"],
    "TEAM":   ["reconciliation","invoice","claims","ar/ap","24/7","chatbot","ab testing","pipeline","data platform"],
    "CODE":   ["reconciliation","invoice","claims","ar/ap","24/7","chatbot","feedback","collections"]
}

CATEGORY_HINTS = {
    "CARE":   [r"\bcustomer service\b", r"\bsupport\b", r"\bticket(s|ing)?\b", r"\bescalation\b", r"\bknowledge base\b", r"\b24\/?7\b"],
    "STRIKE": [r"\bsales\b", r"\bpipeline\b", r"\blead(s)?\b", r"\bqualification\b", r"\boutreach\b", r"\bmeeting(s)?\b", r"\bproposal(s)?\b"],
    "FLOW":   [r"\boperation(s)?\b", r"\bSLA(s)?\b", r"\bprocess\b", r"\bfulfillment\b", r"\bhandoff\b", r"\bqueue(s)?\b", r"\bvendor(s)?\b"],
    "HYPE":   [r"\bmarketing\b", r"\bA\/B testing\b", r"\baudience segmentation\b", r"\bsocial media\b", r"\bemail campaign(s)?\b"],
    "VISION": [r"\bKPI(s)?\b", r"\bdecision support\b", r"\bmarket intelligence\b", r"\bcompetitive\b", r"\bchurn\b", r"\bLTV\b"],
    "ASSET":  [r"\bfinance\b", r"\breconciliation\b", r"\binvoice(s)?\b", r"\bar\/ap\b", r"\bcash[- ]?flow\b"],
    "TEAM":   [r"\brecruit(ing|ment)?\b", r"\bscreening\b", r"\bonboarding\b", r"\btraining\b", r"\bperformance review\b"],
    "CODE":   [r"\bAPI(s)?\b", r"\bintegration(s)?\b", r"\bdata pipeline(s)?\b", r"\bETL\b", r"\bcloud\b", r"\bsecurity\b"]
}
def detect_agents_from_text(text: str):
    found=set(); t=text or ""
    for agent, pats in CATEGORY_HINTS.items():
        if any(re.search(p, t, flags=re.I) for p in pats): found.add(agent)
    return found

SAFE_PHRASES = {
    "CARE":   [r"\bknowledge base article\b"],
    "STRIKE": [r"\bfollow-?up feedback\b"],
    "FLOW":   [r"\bSLA reporting\b"],
    "HYPE":   [r"\bA\/B testing\b"],
    "VISION": [r"\bKPI dashboard\b"],
    "ASSET":  [r"\bcash flow statement\b"],
    "TEAM":   [r"\btraining feedback\b"],
    "CODE":   [r"\bdata platform integration\b"]
}

def find_violations(agent: str, text: str):
    t = text or ""

    for sp in SAFE_PHRASES.get(agent, []):
        t = re.sub(sp, "__SAFE__", t, flags=re.I)
    violations=set()
    for term in FORBIDDEN_BY_AGENT.get(agent, []):
        pat = r"\b" + re.escape(term).replace(r"\ ", r"\s+").replace(r"\/", r"[\/]") + r"\b"
        if re.search(pat, t, flags=re.I):
            violations.add(term)
    return sorted(violations)

SAFE_REPLACEMENTS = {
    "STRIKE": { r"\bfeedback\b": "response", r"\bchatbot\b": "automated outreach" },
    "FLOW":   { r"\b24\/?7\b": "continuous", r"\bchatbot\b": "workflow automation" }
}
def apply_safe_replacements(agent: str, text: str):
    out=text
    for pat,repl in SAFE_REPLACEMENTS.get(agent, {}).items():
        out = re.sub(pat, repl, flags=re.I)
    return out.replace("__SAFE__", "safe-phrase")

# ========== Routes ==========
@app.route("/match_agents", methods=["POST"])
def match_agents():
    try:
        transcript = request.files["transcript"]; agent_file = request.files["agent_manual"]
        tp = Path(UPLOAD_FOLDER)/transcript.filename; ap = Path(UPLOAD_FOLDER)/agent_file.filename
        transcript.save(tp); agent_file.save(ap)

        company_info = request.form.get("company_info","")

        sections = load_all_documents([tp, ap])
        if company_info:
            sections.append({"title":"Company Info","text":company_info,"source":"web_summary","type":"company"})
        chunks = split_into_chunks(sections, chunk_size=500, chunk_overlap=50)
        index  = build_faiss_index(chunks)

        session_id = request.form.get("session_id","default")
        index_cache[session_id] = {"index": index, "transcript_path": str(tp), "agent_path": str(ap), "company_info": company_info}

        query = (
            "You are an AI strategy consultant. From the transcript + agent manual, extract top concrete business pain points (3–8 items, concise). "
            "Then pick ONLY the most relevant agents that directly solve those pains. "
            "Return STRICT JSON with keys: pain_points (array of strings), agents (array of agent codes), rationale (object mapping agent->one-sentence reason). "
            "DO NOT add prose outside JSON.\n\n"
            "Agent reference:\n" + "\n".join([f"- {k}: {v}" for k, v in AGENT_DEFINITIONS.items()]) + "\n\n"
            f"{GUARDRAILS}\n"
            "JSON schema example:\n"
            "{\n"
            "  \"pain_points\": [\"...\", \"...\"],\n"
            "  \"agents\": [\"TEAM\", \"FLOW\"],\n"
            "  \"rationale\": {\"TEAM\": \"...\", \"FLOW\": \"...\"}\n"
            "}\n"
        )
        
        result_text = rag_chain(index, query)
        parsed = _safe_json_extract(result_text) or {}

        pain_points   = parsed.get("pain_points", []) or []
        agents        = parsed.get("agents", []) or []
        rationale     = parsed.get("rationale", {}) or {}

        index_cache[session_id].update({"pain_points": pain_points, "matched_agents": agents, "rationale": rationale})

        return jsonify({"session_id": session_id, "pain_points": pain_points, "matched_agents": agents, "rationale": rationale})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate_agent_module", methods=["POST"])
def generate_agent_module():
    try:
        agent_name = request.json.get("agent")
        session_id = request.json.get("session_id","default")
        company_info = request.json.get("company_info","")
        force = bool(request.json.get("force", False))

        cached = index_cache.get(session_id)
        if not cached:
            save_dir = f"/tmp/faiss_{session_id}"
            if os.path.isdir(save_dir):
                index_cache[session_id] = {"index": load_faiss_index(save_dir), "company_info": company_info,
                                           "pain_points": [], "matched_agents": [], "rationale": {}}
                cached = index_cache[session_id]
        if not cached: return jsonify({"error":"No index found for session."}), 400
        if agent_name not in AGENT_DEFINITIONS: return jsonify({"error": f"Invalid agent name: {agent_name}"}), 400

        index = cached["index"]
        pain_points = request.json.get("pain_points") or cached.get("pain_points", [])
        matched_agents = cached.get("matched_agents", [])
        merged_info = (company_info or cached.get("company_info","")).strip()

        protected = set(matched_agents) | detect_agents_from_text(" ".join(pain_points) + "\n" + merged_info)

        pains_block = "\n".join([f"- {pp}" for pp in pain_points]) if pain_points else "- (none)"

        context_hints = {
            "STRIKE": "Stay in sales (leads, scoring, outreach, meetings, proposals). Avoid chatbots, 24/7, finance, HR.",
            "FLOW":   "Stay in operations (SLA, handoffs, queues, vendors). Avoid 24/7 ownership, chatbots, finance.",
            "CARE":   "Stay in support (24/7, tickets, KB, escalation). Avoid finance and sales pipeline.",
            "HYPE":   "Stay in marketing automation/experiments. Avoid sales/finance/HR/support/data-platform ownership.",
            "VISION": "Stay in strategy/KPIs/market intel. Avoid operational ownership/finance/data-platform ownership.",
            "ASSET":  "Stay in finance ops (AR/AP, invoices, reconciliation, cashflow). Avoid 24/7/chatbots/sales.",
            "TEAM":   "Stay in HR lifecycle (recruiting, onboarding, performance, knowledge). Avoid finance and sales.",
            "CODE":   "Stay in integrations/data (APIs, pipelines, governance, serving, monitoring). Avoid business ownership."
        }
        agent_hint = context_hints.get(agent_name,"")

        query = (
            f"Write an industry-agnostic solution module for {agent_name} using ONLY the pains below.\n"
            f"Agent definition:\n{AGENT_DEFINITIONS[agent_name]}\n\n"
            f"PAIN POINTS:\n{pains_block}\n\n"
            "Rules:\n- Plain text only; no JSON or code fences.\n- Map features/impacts to pains only; avoid forbidden cross-domain claims.\n"
            "- If nothing valid remains, output exactly: No relevant outcomes for this agent given the pains.\n\n"
            f"Agent context hint: {agent_hint}\n"
        )

        draft = strip_meta(rag_chain(index, query)).strip()
        if draft == "No relevant outcomes for this agent given the pains.":
            return jsonify({"agent_module": ""})

        draft = apply_safe_replacements(agent_name, draft)
        violations = find_violations(agent_name, draft)

        if violations:

            lines = draft.splitlines()
            kept = []
            for ln in lines:
                if any(re.search(r"\b"+re.escape(term).replace(r"\ ", r"\s+")+r"\b", ln, flags=re.I) for term in violations):
                    continue
                kept.append(ln)
            draft = "\n".join(kept).strip()

        return jsonify({"agent_module": draft})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
