from flask import Flask, request, jsonify
from processor import load_all_documents, split_into_chunks, build_faiss_index, rag_chain
from pathlib import Path
import os
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

index_cache = {}

AGENT_DEFINITIONS = {
    "HYPE": "HYPE (Marketing Intelligence Agent): Transforms marketing into a revenue engine through intelligent automation and data-driven campaign optimization, including social media content creation and multi-platform management, A/B testing optimization, and personalized email marketing.",
    "STRIKE": "STRIKE (Sales Acceleration Agent): Optimizes sales processes with intelligent lead scoring, automated outreach, meeting scheduling, proposal and contract generation, increasing conversion rates and shortening sales cycles.",
    "CARE": "CARE (Customer Experience Agent): Provides 24/7 customer support via AI voice receptionist and omnichannel chatbot; automates onboarding, collects and analyzes feedback to improve customer satisfaction and retention.",
    "VISION": "VISION (Strategic Intelligence Agent): Delivers executive-level intelligence including business health diagnostics, cross-functional performance analysis, competitive intelligence, and customer behavior analytics to enhance decision-making and market response.",
    "FLOW": "FLOW (Operations Excellence Agent): Improves operational efficiency through supply chain optimization, vendor management, quality control, and automated order fulfillment, reducing costs and improving delivery performance and compliance.",
    "ASSET": "ASSET (Financial Intelligence Agent): Automates accounts receivable, invoice processing, expense management, and cash flow forecasting to improve collections, reduce processing costs, and enhance financial health.",
    "TEAM": "TEAM (Human Capital Agent): Enhances recruitment, onboarding, performance management, and knowledge management to reduce turnover, accelerate time-to-productivity, and improve employee satisfaction.",
    "CODE": "CODE (Technology Intelligence Agent): Provides intelligent infrastructure management including system integration, data architecture and quality management, machine learning deployment, and cloud infrastructure with cybersecurity."
}

@app.route("/match_agents", methods=["POST"])
def match_agents():
    try:
        transcript_file = request.files["transcript"]
        agent_file = request.files["agent_manual"]

        transcript_path = Path(UPLOAD_FOLDER) / transcript_file.filename
        agent_path = Path(UPLOAD_FOLDER) / agent_file.filename
        transcript_file.save(transcript_path)
        agent_file.save(agent_path)

        company_info = request.form.get("company_info", None)

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
            "agent_path": str(agent_path)
        }

        query = (
            "You are an AI strategy consultant. Based on the transcript content, identify business pain points.\n"
            "Then recommend only about 3-4 most relevant agents that directly address these pain points. You must choose from the following 8 official agents only — do not invent new ones.\n\n"
            + "\n".join([f"- {v}" for v in AGENT_DEFINITIONS.values()]) + "\n\n"
            "Return a JSON array of relevant agent names, e.g., [\"CARE\", \"STRIKE\"] only."
        )
        
        result = rag_chain(index, query)
        return jsonify({"matched_agents": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/generate_agent_module", methods=["POST"])
def generate_agent_module():
    try:
        agent_name = request.json.get("agent")
        session_id = request.json.get("session_id", "default")
        company_info = request.json.get("company_info", "") 

        cached = index_cache.get(session_id)
        if not cached:
            return jsonify({"error": "No index found for session."}), 400

        if agent_name not in AGENT_DEFINITIONS:
            return jsonify({"error": f"Invalid agent name: {agent_name}"}), 400

        index = cached["index"]
        agent_definition = AGENT_DEFINITIONS[agent_name]

        query = (
        f"You are an AI consultant. Write a full structured solution module for the agent {agent_name}, "
        f"tailored specifically to the client's industry and needs described in the transcript.\n\n"
        f"Use the official definition below for this agent as a reference only:\n\n"
        f"{agent_definition}\n\n"
        "When writing, you must:\n"
        "● Use the transcript context and company information as the primary source.\n"
        "● Only include features, impacts, and outcomes that are relevant to the client's industry and pain points.\n"
        "● The agent manual can be used to fill in missing details, but avoid including unrelated workflows or metrics.\n"
        "● Remove any generic or irrelevant features not supported by the transcript or the company's profile.\n\n"
        "The company's official description is as follows:\n"
        f"{company_info}\n\n"
        "Follow this exact structure:\n\n"
        f"{agent_name} – [AGENT TITLE]\n"
        "This solution [describe what this agent does for the business based on transcript context. Write at least 120 words, focusing only on relevant pain points and workflows].\n\n"
        "Key Features:\n"
        "● [Select only features that match the industry and pain points. More than 2 full sentences.]\n"
        "● ...\n"
        "● ...\n"
        "● ...\n\n"
        "Business Impact:\n"
        "● [Summarize only impact metrics that are realistic for the client's industry. At least 2 full sentences.]\n"
        "● ...\n"
        "● ...\n"
        "● ...\n\n"
        "Transformation Summary:\n"
        "Summarize how this agent transforms the client’s operations, focusing on the specific industry context.\n\n"
        "Expected Outcomes:\n"
        "List 4–5 outcomes that are directly supported by the transcript or manual, in this format:\n"
        "● Operational Efficiency: Automation of manual tasks frees up staff time\n"
        "● [Outcome 2]\n"
        "● [Outcome 3]\n"
        "● [Outcome 4]\n"
        "● [Outcome 5]\n\n"
        "**Important:**\n"
        "● Do not mention or reference any other agents.\n"
        "● Do not include unrelated industries or workflows.\n"
        "● If the transcript lacks certain details, adapt the agent manual to the industry context instead of copying generic text.\n"
        "● Always keep the language and examples consistent with the client's business environment."
        )

        result = rag_chain(index, query)
        return jsonify({"agent_module": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f"Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
