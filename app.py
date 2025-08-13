from flask import Flask, request, jsonify
from processor import load_all_documents, split_into_chunks, build_faiss_index, load_faiss_index, rag_chain
from pathlib import Path
import os
import traceback
import json

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

# === Text-only GUARDRAILS (prompt-enforced; no programmatic ownership logic) ===
GUARDRAILS = r"""
Select ONLY from: HYPE, STRIKE, CARE, VISION, FLOW, ASSET, TEAM, CODE.

GENERAL RULES (apply to all):
- Industry-agnostic. Use domain-neutral wording (no client/vendor names).
- Evidence-gated: Include an agent ONLY if explicitly supported by retrieved text.
- Do NOT infer or assume missing functions. Absence of evidence = not eligible.
- For each eligible agent, attach 1–3 short verbatim snippets (≤30 words each) as evidence.
- If no agents qualify, return an empty list for agents.
- Keep specific processes distinct (e.g., 'monthly timesheets/payroll' vs. '24/7 client support/FAQ').

UNIQUENESS / NO-OVERLAP RULES (prompt-level; model must comply):
- A capability may be assigned to ONE official owner agent only. When multiple agents appear to share the same capability, assign it ONLY to its official owner below.
- Do NOT list the same capability under more than one agent, even if context suggests overlap.
- If text is ambiguous, choose the single best-fit agent per capability and state the reason concisely.

OFFICIAL CAPABILITY OWNERS (reference only; do not cross-list):
- Finance-only → ASSET: reconciliation, invoice matching, AR/AP, expense control, cash-flow forecasting, payroll/timesheets/attendance.
- CX-only → CARE: 24/7 support, FAQ/knowledge base, chatbot/voice receptionist, ticket routing, escalation-to-human.
- Marketing-only → HYPE: content automation, cross-platform orchestration, audience segmentation, A/B testing & optimization.
- Sales-only → STRIKE: lead scoring/qualification, pipeline stage progression, outreach/scheduling, proposals/quotations.
- Strategy-only → VISION: exec KPIs, strategic decision support, competitive/market intelligence, LTV/churn.
- Operations-only → FLOW: process orchestration, vendor/supplier coordination, fulfillment/SLAs, quality/compliance monitoring.
- HR-only → TEAM: recruiting/sourcing/screening, onboarding, performance/development, internal knowledge.
- Tech-only → CODE: system integration/APIs, data pipelines/sync, model ops, BI/ML platform, cloud/security/monitoring.

PAIN-POINT EXTRACTION (case-agnostic, evidence-gated):
- Extract 3–8 concise, non-duplicative pain points.
- Each pain point MUST include 1–2 short quotes (≤30 words) from retrieved text.
- Use neutral wording (no client/vendor names).

ELIGIBILITY BY AGENT (INCLUDE IF / EXCLUDE IF):

- HYPE (Marketing):
  INCLUDE IF: social/content/email automation, ads, A/B testing, segmentation, marketing ROI optimization.
  EXCLUDE IF: no marketing activities.

- STRIKE (Sales):
  INCLUDE IF: pipeline/stage management, lead scoring/qualification, cold outreach, meeting scheduling, proposals/quotations.
  EXCLUDE IF: no sales activity.

- CARE (Customer Experience/Support):
  INCLUDE IF: 24/7 or after-hours support; FAQ/knowledge base/help center; chatbot/voice; ticket routing; escalation; compliance/policy guidance; onboarding support; multi-channel service (web/WhatsApp/email/call).
  EXCLUDE IF: no CX/support or KB/escalation workflows.

- VISION (Strategy/Executive Intelligence):
  INCLUDE IF: cross-functional KPIs, competitive/market intel, strategic decision support, LTV/churn analysis.
  EXCLUDE IF: purely operational, no executive insights.

- FLOW (Operations/Back-office):
  INCLUDE IF: process orchestration, vendor/supplier management, quality/compliance monitoring, order/fulfillment, back-office automation.
  EXCLUDE IF: only front-office (marketing/sales/support).

- ASSET (Finance):
  INCLUDE IF: AR/AP, invoicing, invoice matching, reconciliation, expense control, cash-flow forecasting; OR payroll/timesheet/attendance (clock-in/out, overtime, daily-rate/penalty, benefits/tax/social security).
  EXCLUDE IF: no finance/payroll/accounting.

- TEAM (Human Capital/Talent):
  INCLUDE IF: recruiting/sourcing/screening, onboarding, performance tracking/development, internal knowledge base, skills matching.
  EXCLUDE IF: no HR/talent themes.

- CODE (Tech Foundation/Integration):
  INCLUDE IF: system integration/API, data pipelines/sync, model ops, cloud/security/monitoring, BI/ML platform.
  EXCLUDE IF: no integration/infrastructure needs.

OUTPUT: Return STRICT JSON ONLY (no prose). Use this exact schema:
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
  "rationale": {
    "TEAM": "...",
    "FLOW": "...",
    "STRIKE": "..."
  }
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
            candidate = text[start:end+1]
            return json.loads(candidate)
    except Exception:
        return None
    return None


@app.route("/match_agents", methods=["POST"])
def match_agents():
    try:
        transcript_file = request.files["transcript"]
        agent_file = request.files["agent_manual"]

        transcript_path = Path(UPLOAD_FOLDER) / transcript_file.filename
        agent_path = Path(UPLOAD_FOLDER) / agent_file.filename
        transcript_file.save(transcript_path)
        agent_file.save(agent_path)

        company_info = request.form.get("company_info", None) or ""

        sections = load_all_documents([transcript_path, agent_path])

        if company_info:
            sections.append({
                "title": "Company Info",
                "text": company_info,
                "source": "web_summary",
                "type": "company"
            })

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

        # Prompt: industry-agnostic + uniqueness rules (prompt-level only)
        query = (
            "You are an AI strategy consultant.\n"
            "- Extract 3–8 general, industry-agnostic pain points (concise, no client/vendor names).\n"
            "- Then select ONLY the agents that directly solve them, strictly following the guardrails and uniqueness rules below (no capability overlap across agents).\n"
            "Return STRICT JSON only with keys: pain_points (array), agents (array of agent codes), rationale (map agent->one sentence reason).\n"
            "Attach 1–3 short verbatim snippets (≤30 words) inside eligibility for each eligible agent.\n\n"
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
        parsed = _safe_json_extract(result_text)

        pain_points = []
        matched_agents = []
        rationale = {}

        if isinstance(parsed, dict):
            pain_points = parsed.get("pain_points", []) or []
            matched_agents = parsed.get("agents", []) or []
            rationale = parsed.get("rationale", {}) or {}

        index_cache[session_id]["pain_points"] = pain_points
        index_cache[session_id]["matched_agents"] = matched_agents
        index_cache[session_id]["rationale"] = rationale

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
                try:
                    index = load_faiss_index(save_dir)
                    index_cache[session_id] = {
                        "index": index,
                        "transcript_path": "",
                        "agent_path": "",
                        "company_info": company_info,
                        "save_dir": save_dir,
                        "pain_points": [],
                        "matched_agents": [],
                        "rationale": {}
                    }
                    cached = index_cache[session_id]
                except Exception:
                    pass

        if not cached:
            return jsonify({"error": "No index found for session."}), 400

        if agent_name not in AGENT_DEFINITIONS:
            return jsonify({"error": f"Invalid agent name: {agent_name}"}), 400

        index = cached["index"]
        agent_definition = AGENT_DEFINITIONS[agent_name]
        pain_points = request.json.get("pain_points") or cached.get("pain_points", [])
        matched_agents = cached.get("matched_agents", [])
        company_info_cached = cached.get("company_info", "")

        merged_company_info = company_info.strip() or company_info_cached

        if matched_agents and (agent_name not in matched_agents) and not force:
            return jsonify({
                "error": f"Agent '{agent_name}' is not relevant to extracted pain points.",
                "matched_agents": matched_agents,
                "pain_points": pain_points,
                "hint": "Pass force=true to override, or choose one of the matched agents."
            }), 400

        pain_points_block = "\n".join([f"- {pp}" for pp in pain_points]) if pain_points else "- (no extracted pain points)"

        # Prompt: general framing; rely on textual guardrails; no programmatic capability allow/ban
        query = (
            f"You are an AI consultant. Write a general (industry-agnostic) solution module for the agent {agent_name}, "
            f"anchored to the pains below and following the uniqueness (no-overlap) rules in the guardrails.\n\n"
            f"OFFICIAL AGENT DEFINITION (reference only, do not copy blindly):\n{agent_definition}\n\n"
            f"PAIN POINTS (authoritative anchors):\n{pain_points_block}\n\n"
            "HARD RULES:\n"
            "● Use the pains as PRIMARY constraints; only include features/impacts/outcomes that directly map to these pains.\n"
            "● Do not reference other agents. Use plain text only; no Markdown emphasis.\n\n"
            "Follow this structure exactly:\n\n"
            f"{agent_name} – [AGENT TITLE]\n"
            "This solution [≥120 words; describe what this agent does based on the pains; focus ONLY on relevant workflows].\n\n"
            "Key Features:\n"
            "● Each feature as 'Title: description' (1–2 sentences).\n"
            "● ...\n"
            "● ...\n"
            "● ...\n\n"
            "Business Impact:\n"
            "● Realistic, observable impacts aligned to the pains; avoid unverifiable precise numbers.\n"
            "● ...\n"
            "● ...\n"
            "● ...\n\n"
            "Transformation Summary:\n"
            "Summarize how this agent transforms operations, anchored to pains (general framing).\n\n"
            "Expected Outcomes:\n"
            "List 4–5 achievable outcomes directly supported by pains/manual.\n"
            "● [Outcome 1]\n"
            "● [Outcome 2]\n"
            "● [Outcome 3]\n"
            "● [Outcome 4]\n"
            "● [Outcome 5]\n\n"
            f"Guardrails to follow:\n{GUARDRAILS}\n"
        )

        result = rag_chain(index, query)
        return jsonify({
            "agent_module": result
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
