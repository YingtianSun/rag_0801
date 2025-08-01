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
    "STRIKE": "STRIKE (Sales Acceleration Agent): Streamlines sales processes through lead scoring, automated follow-ups, intelligent proposal generation, and meeting scheduling. Boosts conversion rates and shortens sales cycles by managing pipelines and optimizing prospect interactions.",
    "CARE": "CARE (Customer Experience Agent): Delivers 24/7 omnichannel support via AI voice and chatbot systems. Handles FAQs, schedules appointments, resolves inquiries, and manages post-care feedback. Escalates complex queries to human agents and improves customer satisfaction through sentiment analysis.",
    "FLOW": "FLOW (Operations Excellence Agent): Automates backend workflows including document processing, inventory handling, and order fulfillment. Increases efficiency in admin-heavy tasks, enables seamless handovers across departments, and reduces operational bottlenecks.",
    "HYPE": "HYPE (Marketing Intelligence Agent): Automates and optimizes cross-platform marketing campaigns, social media management, and email outreach. Generates AI-driven content, performs trend analysis, and tracks campaign performance to improve lead generation and engagement.",
    "TEAM": "TEAM (Human Capital Agent): Supports workforce transformation through digital onboarding, AI-powered recruitment, performance tracking, and internal knowledge management. Facilitates upskilling and career progression through structured training and development.",
    "CODE": "CODE (Technology Intelligence Agent): Acts as the system integrator and infrastructure backbone. Connects chatbots, and third-party APIs. Enables real-time data flow, system interoperability, and future scalability of AI functions through a unified tech stack.",
    "VISION": "VISION (Strategic Intelligence Agent): Uncovers actionable insights by analyzing operational, customer, and feedback data. Provides dashboards, KPI monitoring, trend analysis, and strategic decision support. Enables intelligent prioritization and segmentation based on sentiment, behavior, and historical records to guide targeted outreach and long-term planning.",
    "ASSET": "ASSET (Financial Intelligence Agent): Manages financial processes including invoice matching, payment tracking, cash flow forecasting, and expense optimization. Supports insurance claims reconciliation and automates follow-ups for delayed payments."
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

        sections = load_all_documents([transcript_path, agent_path])
        chunks = split_into_chunks(sections, chunk_size=800, chunk_overlap=200)
        index = build_faiss_index(chunks)

        session_id = request.form.get("session_id", "default")
        index_cache[session_id] = {
            "index": index,
            "transcript_path": str(transcript_path),
            "agent_path": str(agent_path)
        }

        query = (
            "You are an AI strategy consultant. Based on the transcript content, identify business pain points.\n"
            "Then recommend only those agents that directly address these pain points. You must choose from the following 8 official agents only — do not invent new ones.\n\n"
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

        cached = index_cache.get(session_id)
        if not cached:
            return jsonify({"error": "No index found for session."}), 400

        if agent_name not in AGENT_DEFINITIONS:
            return jsonify({"error": f"Invalid agent name: {agent_name}"}), 400

        index = cached["index"]
        agent_definition = AGENT_DEFINITIONS[agent_name]

        query = (
            f"You are an AI consultant. Write a full structured solution module for the agent {agent_name}, based on the transcript context.\n\n"
            f"Use the official definition below for this agent only:\n\n"
            f"{agent_definition}\n\n"
            "Follow this exact structure:\n\n"
            f"{agent_name} – [AGENT TITLE]\n"
            "This solution [describe what this agent does for the business based on transcript context. Write at least 120 words, using transcript language and pain points].\n\n"
            "Key Features:\n"
            "● [Only use features copied or inferred from the agent manual. Write more than 2 full sentences.]\n"
            "● ...\n"
            "● ...\n"
            "● ...\n\n"
            "Business Impact:\n"
            "● [Summarize impact metrics as described in the agent manual. At least 2 full sentences.]\n"
            "● ...\n"
            "● ...\n"
            "● ...\n\n"
            "Transformation Summary:\n"
            "Summarize how this agent transforms the client’s operations, improves efficiency, and enhances the service, operations, or sales process.\n\n"
            "Expected Outcomes:\n"
            "List 4–5 outcomes in this format. Do not invent outcomes — only use approved ones:\n"
            "● Operational Efficiency: Automation of manual tasks frees up staff time\n"
            "● Revenue Growth Potential: 25–40% increase in follow-up appointments via structured data\n"
            "● ...\n"
            "● ...\n\n"
            "**Important:**\n"
            "● Do not mention or reference any other agents.\n"
            "● Do not respond with 'I don't know', 'insufficient data', or similar vague answers.\n"
            "● If a match seems unclear, still attempt to provide a response by inferring from pain points or transcript language.\n"
            "● Always generate a complete answer, even if the context is sparse."
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
