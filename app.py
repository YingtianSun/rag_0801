from flask import Flask, request, jsonify
from processor import (
    load_all_documents, split_into_chunks, build_faiss_index, rag_chain
)
from pathlib import Path
import os, json, traceback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

AGENT_DEFINITIONS = {
    "HYPE": "HYPE (Marketing Intelligence Agent): Transforms marketing into a revenue engine via intelligent automation and data-driven campaign optimization. Core scope includes AI social content creation, cross-platform social management, A/B testing and optimization, and personalized email campaign automation.",
    "STRIKE": "STRIKE (Sales Acceleration Agent): Transforms sales through intelligent pipeline management and automated relationship building. Core scope includes lead scoring, automated pipeline progression, cold outreach (research/persona/outreach/follow-ups), meeting scheduling and follow-up, and proposal/quotation automation.",
    "CARE": "CARE (Customer Experience Agent): Delivers omnichannel customer support that blends AI efficiency with human escalation. Core scope includes AI voice receptionist, omnichannel chatbot/ticket routing with sentiment, customer onboarding automation, and feedback/experience analytics.",
    "VISION": "VISION (Strategic Intelligence Agent): Provides executive-level strategic intelligence turning data into actionable insights and competitive advantage. Core scope includes business health diagnostics, cross-functional performance dashboards, competitive response coordination, and customer intelligence/LTV optimization.",
    "FLOW": "FLOW (Operations Excellence Agent): Optimizes operational efficiency via intelligent process automation and supply chain intelligence. Core scope includes inventory management/forecasting, supplier/vendor management, quality/compliance monitoring, and order fulfillment & delivery optimization.",
    "ASSET": "ASSET (Financial Intelligence Agent): Transforms financial management via intelligent automation and predictive planning. Core scope includes intelligent accounts receivable, automated invoice processing and matching, expense management and approvals, and cash-flow management and forecasting.",
    "TEAM": "TEAM (Human Capital Agent): Optimizes human potential with intelligent recruitment, development, and performance management. Core scope includes automated onboarding, recruitment & screening, performance monitoring & development, and internal knowledge management.",
    "CODE": "CODE (Technology Intelligence Agent): Provides the technology foundation for AI-driven transformation through intelligent infrastructure management. Core scope includes business tool connection hub, data architecture & ML foundation, cloud infrastructure & cybersecurity, and predictive BI & ML engines."
}

GUARDRAILS = r"""
Select ONLY from: HYPE, STRIKE, CARE, VISION, FLOW, ASSET, TEAM, CODE.

EVIDENCE RULES
- Mark an agent eligible ONLY if explicit, text-grounded evidence exists in retrieved context.
- For every eligible agent, cite 1–3 verbatim snippets (<=30 words each).
- For commonly confused agents you did NOT select (e.g., sales vs recruitment), add a brief 'exclusion_reason'.

DISCIPLINE
- No hallucinations: absence of explicit capability => eligible=false.
- If evidence conflicts, prefer transcript/company context over generic agent manual.

OPTIONAL PRIORITY
- If 'priority' list is provided, rank agents by that order first, then by confidence/evidence count.
- If 'low_priority' list is provided, require >=2 independent evidence snippets to mark those agents eligible.

MAPPING & COVERAGE
- Build a pain→agent mapping table; if a pain has no agent, mark it 'uncovered'.
- coverage_score in [0,1] = (#pains with >=1 agent) / (total pains).

OUTPUT (STRICT JSON ONLY):
{
  "pain_points": ["..."],
  "eligibility": {
    "HYPE":   {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "STRIKE": {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "CARE":   {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "VISION": {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "FLOW":   {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "ASSET":  {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "TEAM":   {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""},
    "CODE":   {"eligible": false, "reason": "", "evidence": [], "confidence": 0.0, "exclusion_reason": ""}
  },
  "agents_ranked": [],
  "mapping": [ {"pain": "...", "agents": ["..."], "coverage": "high|medium|low|uncovered"} ],
  "coverage_score": 0.0,
  "notes": ["uncovered pains, conflicts, or risks"]
}
"""

def _safe_json_extract(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e+1])
    except Exception:
        return None
    return None

def _parse_csv(s: str):
    return [x.strip() for x in s.split(",") if x and x.strip()]

def _enforce_thresholds(parsed, priority_list, low_priority_list):
    if not isinstance(parsed, dict) or "eligibility" not in parsed:
        return parsed

    elig = parsed["eligibility"]

    for agent in low_priority_list:
        if agent in elig and elig[agent].get("eligible"):
            if len(elig[agent].get("evidence") or []) < 2:
                elig[agent]["eligible"] = False
                elig[agent]["reason"] = "Insufficient independent evidence for low-priority agent."

    def sort_key(a):
        prio_idx = priority_list.index(a) if a in priority_list else 10**6
        conf = float(elig[a].get("confidence") or 0.0)
        evid = len(elig[a].get("evidence") or [])
        return (prio_idx, -conf, -evid)

    selected = [a for a, v in elig.items() if v.get("eligible")]
    parsed["agents_ranked"] = sorted(selected, key=sort_key)

    parsed["agents"] = parsed.get("agents_ranked", [])[:4]
    parsed.setdefault("rationale", {})
    for a in parsed["agents"]:
        if not parsed["rationale"].get(a):
            parsed["rationale"][a] = (elig[a].get("reason") or "")[:240]

    if not parsed.get("coverage_score"):
        mapping = parsed.get("mapping") or []
        pains = parsed.get("pain_points") or []
        covered = 0
        names = {m.get("pain"): m for m in mapping}
        for p in pains:
            entry = names.get(p)
            if entry and entry.get("agents"):
                covered += 1
        parsed["coverage_score"] = round(covered / max(1, len(pains)), 3)

    return parsed

@app.route("/analyze", methods=["POST"])
def analyze():
    try:

        transcript_file = request.files["transcript"]
        agent_file = request.files["agent_manual"]
        transcript_path = Path(UPLOAD_FOLDER) / transcript_file.filename
        agent_path = Path(UPLOAD_FOLDER) / agent_file.filename
        transcript_file.save(transcript_path); agent_file.save(agent_path)

        company_info = request.form.get("company_info", "") or ""
        priority = request.form.get("priority", "")          
        low_priority = request.form.get("low_priority", "") 
        excluded_agents = request.form.get("excluded_agents", "") 

        sections = load_all_documents([transcript_path, agent_path])
        if company_info:
            sections.append({"title": "Company Info", "text": company_info, "source": "web_summary", "type": "company"})

        chunks = split_into_chunks(sections, chunk_size=500, chunk_overlap=50)
        index = build_faiss_index(chunks)

        priority_line = f"PRIORITY (optional): {priority}" if priority else "PRIORITY (optional):"
        low_priority_line = f"LOW_PRIORITY (optional): {low_priority}" if low_priority else "LOW_PRIORITY (optional):"
        excluded_line = f"EXCLUDED_AGENTS (optional): {excluded_agents}" if excluded_agents else "EXCLUDED_AGENTS (optional):"

        query = (
            "You are an AI strategy consultant.\n"
            "Task:\n"
            "1) Extract 3–8 concrete pains from the provided context.\n"
            "2) Apply GUARDRAILS to decide agent eligibility strictly with evidence snippets.\n"
            "3) Build a pain→agent mapping and compute coverage_score.\n"
            "4) Output STRICT JSON only per schema.\n\n"
            f"{priority_line}\n{low_priority_line}\n{excluded_line}\n\n"
            "Agent reference:\n" + "\n".join([f"- {k}: {v}" for k, v in AGENT_DEFINITIONS.items()]) + "\n\n"
            + GUARDRAILS + "\n"
        )


        result_text = rag_chain(index, query)
        parsed = _safe_json_extract(result_text) or {}

        parsed = _enforce_thresholds(parsed, _parse_csv(priority), _parse_csv(low_priority))

        return jsonify({
            "pain_points": parsed.get("pain_points", []),
            "eligibility": parsed.get("eligibility", {}),
            "agents_ranked": parsed.get("agents_ranked", []),
            "agents": parsed.get("agents", []),
            "mapping": parsed.get("mapping", []),
            "coverage_score": parsed.get("coverage_score", 0.0),
            "rationale": parsed.get("rationale", {}),
            "notes": parsed.get("notes", []),
            "raw": result_text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
