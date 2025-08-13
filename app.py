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

GENERAL RULES (apply to all):
- Include an agent ONLY if there is explicit, text-grounded evidence in the retrieved context/transcript/agent manual.
- Do NOT infer or assume missing functions. Absence of evidence = not eligible.
- For each agent you mark eligible, you must attach 1–3 short verbatim snippets (≤30 words each) that justify eligibility.
- If no agents qualify, return an empty list for agents.
- Do NOT merge specific operational processes into generic buckets. If 'monthly timesheets/payroll' or '24/7 client support/FAQ' appears, keep them as distinct pain points.

PAIN-POINT EXTRACTION (case-agnostic, evidence-gated):
- Extract 3–8 concise, distinct pain points.
- Each pain point MUST be supported by 1–2 short quotes (≤30 words) from the retrieved text.
- Use domain-neutral wording (no client/vendor names).

ELIGIBILITY BY AGENT (INCLUDE IF / EXCLUDE IF):

- HYPE (Marketing):
  INCLUDE IF: Mentions social media/content/email automation, ads, A/B testing, audience segmentation, marketing ROI optimization.
  EXCLUDE IF: No marketing-related activities are mentioned.

- STRIKE (Sales):
  INCLUDE IF: Mentions sales pipeline/stage management, lead scoring/qualification, cold outreach, meeting scheduling, proposals/quotations.
  EXCLUDE IF: No sales activity is mentioned.

- CARE (Customer Experience/Support):
  INCLUDE IF: Mentions 24/7 or round-the-clock client support; FAQ/knowledge base/help center; chatbot/voice receptionist; ticket routing; escalation-to-human; compliance/regulation/policy guidance; client onboarding support; multi-channel (web/WhatsApp/email/call) service.
  STRONG SIGNALS: "FAQ", "knowledge base", "helpdesk", "ticket", "escalation", "SLA", "24/7", "after-hours", "compliance", "regulations", "policy".
  EXCLUDE IF: No customer support/experience or knowledge-base/escalation workflows are present.

- VISION (Strategy/Executive Intelligence):
  INCLUDE IF: Mentions cross-functional KPI analysis, market/competitive intelligence, strategic decision support, LTV/churn analysis.
  EXCLUDE IF: Purely operational with no strategic or executive insights.

- FLOW (Operations/Back-office Automation):
  INCLUDE IF: Mentions process orchestration, vendor/supplier management, quality/compliance monitoring, order/fulfillment, back-office workflow automation.
  EXCLUDE IF: Only front-office activities (marketing/sales/support) are present.

- ASSET (Finance):
  INCLUDE IF: Mentions AR/AP, invoicing, invoice matching, expense control, cash-flow forecasting, financial reconciliation; OR payroll/timesheet/attendance processing including clock-in/out extraction, overtime, daily-rate/penalty calculations, and benefits/tax/social-security contributions.
  STRONG SIGNALS: "timesheet", "attendance", "clock-in", "clock-out", "payroll", "overtime", "daily rate", "penalty", "reconciliation", "AR", "AP", "invoice", "cash flow", "tax", "social security".
  EXCLUDE IF: No finance, payroll, or accounting operations are mentioned.
  
- TEAM (Human Capital/Talent):
  INCLUDE IF: Mentions recruiting/sourcing/screening, onboarding, performance tracking/development, internal knowledge base, skills matching.
  EXCLUDE IF: No HR/talent themes are present.

- CODE (Tech Foundation/Integration):
  INCLUDE IF: Mentions system integration/API, data pipelines/sync, model operations, cloud/security/monitoring, BI/ML platforming.
  EXCLUDE IF: No integration/infrastructure needs are present.

OUTPUT: Return STRICT JSON ONLY (no prose). Use this exact schema:
{
  "pain_points": ["..."],                  
  "eligibility": {                         
    "HYPE": {"eligible": false, "reason": "...", "evidence": []},
    "STRIKE": {"eligible": true, "reason": "...", "evidence": []},
    "CARE": {"eligible": false, "reason": "...", "evidence": []},
    "VISION": {"eligible": false, "reason": "...", "evidence": []},
    "FLOW": {"eligible": true, "reason": "...", "evidence": []},
    "ASSET": {"eligible": false, "reason": "...", "evidence": []},
    "TEAM": {"eligible": true, "reason": "...", "evidence": []},
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

        chunks = split_into_chunks(sections, chunk_size=300, chunk_overlap=50)
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

       query = (
        f"You are an AI consultant. Write a full structured solution module for the agent {agent_name}, "
        f"tailored specifically to the client's industry and the pain points below.\n\n"
        f"OFFICIAL AGENT DEFINITION (reference only, do not copy blindly):\n{agent_definition}\n\n"
        f"CLIENT PAIN POINTS (authoritative, must anchor your content):\n{pain_points_block}\n\n"
        "HARD RULES:\n"
        "● Use the pain points and transcript context as PRIMARY constraints.\n"
        "● Include ONLY features/impacts/outcomes that directly map to these pains and the client's industry context.\n"
        "● If a feature is not supported by the agent manual or not relevant to the pains, exclude it.\n"
        "● Do NOT mention or reference any other agents.\n\n"
        "Company description (if any):\n"
        f"{merged_company_info}\n\n"
        "Follow this exact structure:\n\n"
        f"{agent_name} – [AGENT TITLE]\n"
        "This solution [≥120 words; describe what this agent does for the business based on the pains above; focus ONLY on relevant workflows].\n\n"
        "Key Features:\n"
        "● [Pick features that match the pains; >2 full sentences total.]\n"
        "● ...\n"
        "● ...\n"
        "● ...\n\n"
        "Business Impact:\n"
        "● [Summarize only impact metrics realistic for the client's industry given these pains; ≥2 full sentences.]\n"
        "● ...\n"
        "● ...\n"
        "● ...\n\n"
        "Transformation Summary:\n"
        "Summarize how this agent transforms the client’s operations, anchored to the pains and the industry context.\n\n"
        "Expected Outcomes:\n"
        "List 4–5 outcomes that are directly supported by the pains/manual, e.g.:\n"
        "● Operational Efficiency: Automation of manual tasks frees up staff time\n"
        "● [Outcome 2]\n"
        "● [Outcome 3]\n"
        "● [Outcome 4]\n"
        "● [Outcome 5]\n"
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
