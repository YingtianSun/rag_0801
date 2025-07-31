from flask import Flask, request, jsonify
from processor import load_all_documents, split_into_chunks, build_faiss_index, rag_chain
from pathlib import Path
import os
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

index_cache = {}

AGENT_DEFINITIONS = (
    "- STRIKE (Sales Acceleration Agent): Streamlines sales processes through lead scoring, automated follow-ups, intelligent proposal generation, and meeting scheduling. Boosts conversion rates and shortens sales cycles by managing pipelines and optimizing prospect interactions.\n"
    "- CARE (Customer Experience Agent): Delivers 24/7 omnichannel support via AI voice and chatbot systems. Handles FAQs, schedules appointments, resolves inquiries, and manages post-care feedback. Escalates complex queries to human agents and improves customer satisfaction through sentiment analysis.\n"
    "- FLOW (Operations Excellence Agent): Automates backend workflows including document processing, inventory handling, and order fulfillment. Increases efficiency in admin-heavy tasks, enables seamless handovers across departments, and reduces operational bottlenecks.\n"
    "- HYPE (Marketing Intelligence Agent): Automates and optimizes cross-platform marketing campaigns, social media management, and email outreach. Generates AI-driven content, performs trend analysis, and tracks campaign performance to improve lead generation and engagement.\n"
    "- TEAM (Human Capital Agent): Supports workforce transformation through digital onboarding, AI-powered recruitment, performance tracking, and internal knowledge management. Facilitates upskilling and career progression through structured training and development.\n"
    "- CODE (Technology Intelligence Agent): Acts as the system integrator and infrastructure backbone. Connects chatbots, and third-party APIs. Enables real-time data flow, system interoperability, and future scalability of AI functions through a unified tech stack.\n"
    "- VISION (Strategic Intelligence Agent): Uncovers actionable insights by analyzing operational, customer, and feedback data. Provides dashboards, KPI monitoring, trend analysis, and strategic decision support. Enables intelligent prioritization and segmentation based on sentiment, behavior, and historical records to guide targeted outreach and long-term planning.\n"
    "- ASSET (Financial Intelligence Agent): Manages financial processes including invoice matching, payment tracking, cash flow forecasting, and expense optimization. Supports insurance claims reconciliation and automates follow-ups for delayed payments.\n"
)

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
            "You are an AI strategy consultant. From the transcript, identify business pain points, then recommend relevant agents only from the agent document.\n"
            "Use the following official agent list for reference:\n\n{AGENT_DEFINITIONS} as well."
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

        index = cached["index"]
        query = (
            "You are an AI consultant. Write a full structured solution module for agent {agent_name}, including: key features, business impact, transformation summary, and expected outcomes.\n"
            "Use the official definition for this agent:\n\n{AGENT_DEFINITIONS}"
            "Your output must strictly follow this format:\n\n"
        
            "Proposed Solution:\n"
            "Write a detailed paragraph (at least 150 words) describing how the selected agents collectively address the company’s challenges. Be specific about operational improvements, sales, or customer experience enhancements.\n\n"
            
            "Solution Modules:\n"
            "[AGENT NAME] – [AGENT TITLE]\n"
            "Write at least 100–120 words describing how this agent resolves the client’s problems with examples from the transcript.\n\n"
            
            "Key Features:\n"
            "● Each feature must be described with 2–3 sentences.\n"
            "  Example: 'AI chatbot on the website to handle client inquiries and provide information from an integrated medical knowledge base. This ensures that customers receive instant, accurate answers 24/7 without requiring staff intervention. It also reduces repetitive queries, allowing staff to focus on high-value tasks.'\n"
            "● ...\n"
            "● ...\n"
            "● ...\n\n"
            
            "Business Impact:\n"
            "● Each impact must include metrics and be explained in 2–3 sentences.\n"
            "  Example: 'Reduces inquiry handling workload by 60–70%. This frees staff from repetitive tasks and improves response speed, leading to faster conversion and higher customer satisfaction.'\n"
            "● ...\n"
            "● ...\n"
            "● ...\n\n"
            
            "Summarize (50+ words) how this agent transforms operations, increases efficiency, and enhances customer or sales processes.\n\n"
    
            "Repeat this structure for each relevant agent. Then give:"
            
            "Expected Outcomes:\n"
            "List 4–5 outcomes from the manual. Use the format:\n"
            "● Operational Efficiency: Automation of manual tasks frees up staff time.\n"
            "● Revenue Growth Potential: 25–40% increase in follow-up appointments via structured data.\n"
            "● ...\n\n"
            
            "Rules:\n"
            "- Each agent block must be at least 150 words.\n"
            "- Total output should exceed 600–800 words.\n"
            "- Use a consulting tone, clear structure, and professional language.\n"
            "- Provide detailed explanations for Key Features and Business Impact."
        )

        result = rag_chain(index, query)
        return jsonify({"agent_module": result})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    print(f" Starting server on port {port} ...")
    app.run(host="0.0.0.0", port=port)
