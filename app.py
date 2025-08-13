from flask import Flask, request, jsonify
from processor import load_all_documents, split_into_chunks, build_faiss_index, load_faiss_index, rag_chain
from pathlib import Path
import os
import traceback
import json
import re

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

index_cache = {}

# === OFFICIAL, CASE-AGNOSTIC DEFINITIONS (from 3echo 360 Circuit) ===
AGENT_DEFINITIONS = {
    "HYPE": "HYPE (Marketing Intelligence Agent): Turns marketing into a revenue engine via automation and measurement. Core scope: content ops, cross-platform orchestration, audience segmentation, A/B testing & optimization.",
    "STRIKE": "STRIKE (Sales Acceleration Agent): Scales relationship-led selling via pipeline intelligence and outreach automation. Core scope: lead scoring, stage progression, outreach/scheduling, proposals.",
    "CARE": "CARE (Customer Experience Agent): Delivers 24/7 omnichannel support with human escalation. Core scope: FAQ/knowledge base, chatbot/voice, ticket routing, onboarding & feedback.",
    "VISION": "VISION (Strategic Intelligence Agent): Provides exec-level insight for decisions. Core scope: cross-functional KPIs, customer/LTV & churn, competitive/market intel, strategy dashboards.",
    "FLOW": "FLOW (Operations Excellence Agent): Orchestrates back-office processes. Core scope: process automation, fulfillment/SLAs, quality/compliance monitoring, vendor coordination.",
    "ASSET": "ASSET (Financial Intelligence Agent): Automates finance ops and planning. Core scope: AR/AP, invoice processing & matching, reconciliation, expense control, cash-flow forecasting.",
    "TEAM": "TEAM (Human Capital Agent): Optimizes talent lifecycle. Core scope: recruiting/screening, onboarding, performance & development, internal knowledge.",
    "CODE": "CODE (Technology Intelligence Agent): Provides integration & data platform. Core scope: APIs/connectors, data pipelines & governance, ML/BI foundation, cloud/security."
}

# === Text-only GUARDRAILS (prompt-enforced) ===
GUARDRAILS = r"""
Select ONLY from: HYPE, STRIKE, CARE, VISION, FLOW, ASSET, TEAM, CODE.

GENERAL RULES:
- Industry-agnostic. Use domain-neutral wording (no client/vendor names).
- Evidence-gated: Include an agent ONLY if explicitly supported by retrieved text.
- Do NOT infer or assume missing functions. Absence of evidence = not eligible.
- For each eligible agent, attach 1–3 short verbatim snippets (≤30 words each) as evidence.
- If no agents qualify, return an empty list for agents.
- Keep specific processes distinct (e.g., 'monthly timesheets/payroll' vs. '24/7 client support/FAQ').

UNIQUENESS / NO-OVERLAP:
- A capability may be assigned to ONE owner agent only (see owners below).
- Do NOT list the same capability under multiple agents, even if context suggests overlap.
- If ambiguous, choose the single best-fit agent per capability and state the reason concisely.

OFFICIAL CAPABILITY OWNERS (reference, no cross-listing):
- Finance → ASSET: reconciliation, invoice matching, AR/AP, expense control, cash-flow forecasting, payroll/timesheets/attendance.
- CX → CARE: 24/7 support, FAQ/knowledge base, chatbot/voice, ticket routing, escalation-to-human.
- Marketing → HYPE: content automation, cross-platform orchestration, audience segmentation, A/B testing & optimization.
- Sales → STRIKE: lead scoring/qualification, pipeline stage progression, outreach/scheduling, proposals/quotations.
- Strategy → VISION: exec KPIs, strategic decision support, competitive/market intelligence, LTV/churn.
- Operations → FLOW: process orchestration, vendor/supplier coordination, fulfillment/SLAs, quality/compliance monitoring.
- HR → TEAM: recruiting/sourcing/screening, onboarding, performance/development, internal knowledge.
- Tech/Data → CODE: APIs/integration, data pipelines/sync, model ops, BI/ML platform, cloud/security/monitoring.

PAIN-POINT EXTRACTION:
- Extract 3–8 concise, non-duplicative pain points.
- Each pain point MUST include 1–2 short quotes (≤30 words) from retrieved text.
- Use neutral wording (no client/vendor names).

ELIGIBILITY BY AGENT (INCLUDE IF / EXCLUDE IF):
- HYPE: INCLUDE IF social/content/email automation, ads, A/B testing, segmentation, marketing ROI optimization. EXCLUDE IF no marketing activity.
- STRIKE: INCLUDE IF pipeline/stage, lead scoring/qualification, cold outreach, meeting scheduling, proposals. EXCLUDE IF no sales.
- CARE: INCLUDE IF 24/7 or after-hours, FAQ/KB/help center, chatbot/voice, ticket routing, escalation, compliance/policy, onboarding, multi-channel (web/WhatsApp/email/call). EXCLUDE IF no CX/KB/escalation.
- VISION: INCLUDE IF cross-functional KPIs, competitive/market intel, decision support, LTV/churn. EXCLUDE IF purely operational.
- FLOW: INCLUDE IF process orchestration, vendor/supplier mgmt, quality/compliance, order/fulfillment, back-office automation. EXCLUDE IF only front-office.
- ASSET: INCLUDE IF AR/AP, invoicing, invoice matching, reconciliation, expense control, cash-flow forecasting; OR payroll/timesheet/attendance. EXCLUDE IF no finance/payroll/accounting.
- TEAM: INCLUDE IF recruiting/sourcing/screening, onboarding, performance tracking/development, internal knowledge base, skills matching. EXCLUDE IF no HR/talent.
- CODE: INCLUDE IF integration/API, data pipelines/sync, model ops, cloud/security/monitoring, BI/ML platform. EXCLUDE IF no integration/infrastructure.

OUTPUT (STRICT JSON ONLY):
{
  "pain_points": ["..."],
  "eligibility": {
    "HYPE": {"eligible": false, "reason": "...", "evidence": []},
    "STRIKE": {"eligible": true,  "reason": "...", "evidence": []},
    "CARE": {"eligible": false, "reason": "...", "evidence": []},
    "VISION": {"eligible": false, "reason": "...", "evidence": []},
    "FLOW": {"eligible": true,  "reason": "...", "evidence": []},
    "ASSET": {"eligible": false, "reason": "...", "evidence": []},
    "TEAM": {"eligible": true,  "reason": "...", "evidence": []},
    "CODE": {"eligible": false, "reason": "...", "evidence": []}
  },
  "agents": ["TEAM","FLOW","STRIKE"],
  "rationale": {"TEAM": "...","FLOW": "...","STRIKE": "..."}
}
"""

def _safe_json_extract(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end+1])
    except Exception:
        return None
    return None

# --- Minimal ownership: forbid obvious cross-agent outcomes (lowercase match) ---
FORBIDDEN_BY_AGENT = {
    "HYPE":  ["reconciliation","invoice","insurance form","claims","ar/ap","cash flow","24/7","chatbot","ticket","knowledge base","emr extraction","data platform"],
    "STRIKE":["reconciliation","invoice","insurance form","claims","ar/ap","cash flow","24/7","chatbot","feedback","knowledge base","emr extraction","data platform"],
    "CARE":  ["reconciliation","invoice","insurance form","claims","ar/ap","cash flow","pipeline","proposal","ab testing","emr extraction","data platform"],
    "VISION":["reconciliation","invoice","insurance form","claims","ar/ap","24/7","chatbot","feedback","onboarding (hr)","emr extraction","data platform ownership"],
    "FLOW":  ["reconciliation","invoice","insurance form","claims","24/7","chatbot","feedback","emr extraction"],
    "ASSET": ["24/7","after-hours","chatbot","feedback","knowledge base","pipeline","ab testing","data platform","emr extraction","unified data platform"],
    "TEAM":  ["reconciliation","invoice","insurance form","claims","ar/ap","24/7","chatbot","ab testing","pipeline","data platform","emr extraction"],
    "CODE":  ["reconciliation","invoice","insurance form","claims","ar/ap","24/7","chatbot","feedback","collections"]
}

def find_violations(agent: str, text: str):
    t = (text or "").lower()
    return sorted({term for term in FORBIDDEN_BY_AGENT.get(agent, []) if term in t})

def remove_forbidden_lines(agent: str, text: str):
    """Delete any line containing a forbidden term; keep structure simple."""
    forb = [f.lower() for f in FORBIDDEN_BY_AGENT.get(agent, [])]
    lines = (text or "").splitlines()
    kept, removed = [], []
    for ln in lines:
        ll = ln.lower()
        if any(term in ll for term in forb):
            removed.append(ln)
        else:
            kept.append(ln)
    return "\n".join(kept).strip(), removed

def strip_meta(text: str):
    """Remove code fences and SOURCES lines to keep plain text only."""
    t = text or ""
    t = re.sub(r"```.*?```", "", t, flags=re.S)  # remove fenced blocks
    lines = []
    for ln in t.splitlines():
        if ln.strip().lower().startswith("sources:"):
            continue
        if ln.strip().startswith("```"):
            continue
        lines.append(ln)
    return "\n".join(lines).strip()

@app.route("/match_agents", methods=["POST"])
def match_agents():
    try:
        transcript_file = request.files["transcript"]
        agent_file = request.files["agent_manual"]

        transcript_path = Path(UPLOAD_FOLDER) / transcript_file.filename
        agent_path = Path(UPLOAD_FOLDER) / agent_file.filename
        transcript_file.save(transcript_path)
        agent_file.save(agent_path)

        company_info = request.form.get("company_info", "") or ""

        sections = load_all_documents([transcript_path, agent_path])
        if company_info:
            sections.append({"title":"Company Info","text":company_info,"source":"web_summary","type":"company"})

        chunks = split_into_chunks(sections, chunk_size=500, chunk_overlap=50)
        index = build_faiss_index(chunks)

        session_id = request.form.get("session_id", "default")
        index_cache[session_id] = {
            "index": index,
            "transcript_path": str(transcript_path),
            "agent_path": str(agent_path),
            "company_info": company_info
        }

        save_dir = f"/tmp/faiss_{session_id}"
        try:
            index.save_local(save_dir)
            index_cache[session_id]["save_dir"] = save_dir
        except Exception:
            pass

        query = (
            "You are an AI strategy consultant.\n"
            "- Extract 3–8 general, industry-agnostic pain points (concise, no client/vendor names).\n"
            "- Then select ONLY the agents that directly solve them, strictly following the guardrails and uniqueness rules (no capability overlap).\n"
            "Return STRICT JSON only with keys: pain_points, eligibility, agents, rationale.\n"
            "Attach 1–3 short verbatim snippets (≤30 words) inside eligibility for each eligible agent.\n\n"
            "Agent reference:\n" + "\n".join([f"- {k}: {v}" for k,v in AGENT_DEFINITIONS.items()]) + "\n\n"
            f"{GUARDRAILS}\n"
        )

        result_text = rag_chain(index, query)
        parsed = _safe_json_extract(result_text) or {}

        pain_points = parsed.get("pain_points", []) or []
        matched_agents = parsed.get("agents", []) or []
        rationale = parsed.get("rationale", {}) or {}

        index_cache[session_id].update({
            "pain_points": pain_points,
            "matched_agents": matched_agents,
            "rationale": rationale
        })

        return jsonify({
            "session_id": session_id,
            "pain_points": pain_points,
            "matched_agents": matched_agents,
            "rationale": rationale,
            "raw": result_text
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/generate_agent_module", methods=["POST"])
def generate_agent_module():
    try:
        agent_name = request.json.get("agent")
        session_id = request.json.get("session_id", "default")
        company_info = request.json.get("company_info", "")
        force = bool(request.json.get("force", False))

        cached = index_cache.get(session_id)
        if not cached:
            save_dir = f"/tmp/faiss_{session_id}"
            if os.path.isdir(save_dir):
                index_cache[session_id] = {
                    "index": load_faiss_index(save_dir),
                    "transcript_path": "",
                    "agent_path": "",
                    "company_info": company_info,
                    "save_dir": save_dir,
                    "pain_points": [],
                    "matched_agents": [],
                    "rationale": {}
                }
                cached = index_cache[session_id]

        if not cached:
            return jsonify({"error": "No index found for session."}), 400
        if agent_name not in AGENT_DEFINITIONS:
            return jsonify({"error": f"Invalid agent name: {agent_name}"}), 400

        index = cached["index"]
        agent_definition = AGENT_DEFINITIONS[agent_name]
        pain_points = request.json.get("pain_points") or cached.get("pain_points", [])
        matched_agents = cached.get("matched_agents", [])
        merged_company_info = (company_info or cached.get("company_info", "")).strip()

        pain_points_block = "\n".join([f"- {pp}" for pp in pain_points]) if pain_points else "- (no extracted pain points)"

        OUTCOME_OWNERSHIP = (
            "Outcome ownership (no overlap; examples, not exhaustive):\n"
            "- HYPE Allowed: CTR/engagement, conversion lift from experiments, segmentation precision, creative turnaround, content calendar automation, A/B win-rate, lower CPA/CPM. "
            "Forbidden: sales pipeline, finance (reconciliation/invoices/AR/AP), 24/7/ticketing, EMR extraction, HR, data platform.\n"
            "- STRIKE Allowed: lead-qualification accuracy, meeting-book rate, pipeline velocity/stage conversion, follow-up SLA, proposal turnaround, forecast accuracy, win-rate. "
            "Forbidden: marketing A/B/ROI, finance, 24/7/chatbot/feedback, HR, EMR extraction, data platform.\n"
            "- CARE Allowed: 24/7 coverage, first-response/resolution time, deflection, CSAT/NPS, escalation quality, multi-channel consistency, KB coverage. "
            "Forbidden: finance (reconciliation/invoices/claims/AR/AP), marketing/sales pipeline, HR, data platform, EMR extraction.\n"
            "- VISION Allowed: cross-functional KPI alignment, exec dashboard adoption, decision-latency reduction, churn/LTV insights, competitive-intel to action, scenario planning. "
            "Forbidden: operational task ownership (support/reconciliation/outreach/claims), data platform ownership, HR staffing, EMR extraction.\n"
            "- FLOW Allowed: SLA adherence, handoff error reduction, batch-job timeliness, queue-time reduction, workload balancing, appointment-slot utilization (ops orchestration only). "
            "Forbidden: reconciliation, insurance forms/claims, 24/7 ownership, feedback ops, EMR extraction ownership.\n"
            "- ASSET Allowed: monthly reconciliation time/accuracy, invoice matching, DSO reduction, claim acceptance rate, payment follow-ups/collections, close-cycle duration, expense-policy compliance. "
            "Forbidden: 24/7/chatbot/feedback, marketing A/B/ROI, sales pipeline, HR, data platform, EMR extraction.\n"
            "- TEAM Allowed: time-to-hire, screening throughput/quality, offer-acceptance, onboarding completion/time, training completion, performance review cadence, internal knowledge usage. "
            "Forbidden: finance ops, 24/7/chatbot/feedback, sales pipeline, marketing A/B, data platform, EMR extraction.\n"
            "- CODE Allowed: unified data platform availability/latency/sync completeness, API uptime/error rate, EMR data extraction & structuring availability, pipeline freshness, monitoring/alert MTTR, model/BI serving latency. "
            "Forbidden: finance outcomes (reconciliation/invoices/AR/AP/claims), 24/7/chatbot/feedback ops, marketing/sales/HR outcomes.\n"
        )

        query = (
            f"You are an AI consultant. Write an industry-agnostic solution module for {agent_name}, "
            f"anchored to the pains below and following uniqueness rules and outcome ownership.\n\n"
            f"OFFICIAL AGENT DEFINITION (reference only):\n{agent_definition}\n\n"
            f"PAIN POINTS (authoritative anchors):\n{pain_points_block}\n\n"
            "HARD RULES:\n"
            "● Return PLAIN TEXT only. Do NOT output JSON, code fences, or a 'SOURCES:' line.\n"
            "● Use the pains as PRIMARY constraints; only include features/impacts/outcomes that directly map to these pains.\n"
            "● Do NOT explain or restate forbidden capabilities. Do NOT mention guardrails/ownership in the output.\n"
            "● If no valid content remains for this agent given the pains, output exactly:\n"
            "  No relevant outcomes for this agent given the pains.\n"
            "  (Do not add any other text.)\n"
            f"● Outcome ownership:\n{OUTCOME_OWNERSHIP}\n"
            "STRUCTURE (use plain text, no Markdown emphasis):\n"
            f"{agent_name} – [AGENT TITLE]\n"
            "This solution [≥120 words...]\n\n"
            "Key Features:\n"
            "● Title: description\n● Title: description\n● Title: description\n\n"
            "Business Impact:\n"
            "● Impact 1\n● Impact 2\n● Impact 3\n\n"
            "Transformation Summary:\n"
            "[One short paragraph]\n\n"
            "Expected Outcomes:\n"
            "● Outcome 1\n● Outcome 2\n● Outcome 3\n● Outcome 4\n● Outcome 5\n\n"
            f"Guardrails to follow:\n{GUARDRAILS}\n"
        )

        result = rag_chain(index, query)

        # --- sanitize meta noise (code fences / SOURCES) ---
        result = strip_meta(result)

        # --- Silent handling for "no relevant outcomes" ---
        safe_empty = "No relevant outcomes for this agent given the pains."
        if result.strip() == safe_empty:
            return jsonify({"agent_module": ""})

        # --- remove lines containing forbidden terms ---
        cleaned, _removed_lines = remove_forbidden_lines(agent_name, result)

        # --- final guard: still violations? return empty string silently ---
        if find_violations(agent_name, cleaned):
            return jsonify({"agent_module": ""})

        # success, only return agent_module
        return jsonify({"agent_module": cleaned})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
