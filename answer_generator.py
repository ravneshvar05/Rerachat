"""
answer_generator.py — Uses Groq LLM to generate a natural language response
based on the search results and the user's original query.
"""

from groq import Groq
from config import settings
from logger import logger


_SYSTEM_PROMPT = """\
You are a friendly and knowledgeable real estate assistant helping buyers find their ideal home in India.

You will be given:
1. The user's query
2. A list of matching unique real estate projects from our database, each containing one or more matching unit variations.

Your job is to write a warm, helpful, conversational recommendation.

RULES:
- Be EXTREMELY concise. For each project, write ONLY 1 to 2 lines maximum.
- It should look like a quick, scannable suggestion list.
- If a project has multiple unit variations (e.g. different sizes or configurations for a 3 BHK), MUST explicitly mention the different options available in that single project.
- Mention the project name, developer, city, and key standout features briefly.
- Do NOT write a big paragraph at the start. Dive straight into the suggestions after a brief friendly greeting.
- Do NOT make up information. Only use what is given.
- Do NOT use markdown headers in your response — keep it conversational but structured.
- End with an invitation: "Would you like more details about any of these projects?"
"""


def _format_results_for_llm(results: list[dict]) -> str:
    """Format grouped search results into a readable block for the LLM prompt."""
    if not results:
        return "No matching projects found."

    lines = []
    for i, p in enumerate(results, 1):
        lines.append(f"[Project {i}] {p.get('project_name')} by {p.get('developer_name')}")
        lines.append(f"  Location: {p.get('city')}, {p.get('neighbourhood') or ''}")
        
        # Amenities summary
        amenities = []
        if p.get("has_clubhouse"): amenities.append("Clubhouse")
        if p.get("has_pool"): amenities.append("Pool")
        if p.get("has_park"): amenities.append("Park")
        if amenities:
            lines.append(f"  Project Amenities: {', '.join(amenities)}")

        # List all the matching unit variations found in this project
        lines.append("  Matching Unit Variations:")
        
        for j, u in enumerate(p.get("matching_units", []), 1):
            unit_line = f"    - Variation {j}: {u.get('bhk')} BHK {u.get('property_type')} "
            
            # Area info
            if u.get("super_builtup_sqft"):
                unit_line += f"| {u['super_builtup_sqft']} sqft "
            elif u.get("carpet_area_sqft"):
                unit_line += f"| {u['carpet_area_sqft']} sqft "
                
            # Unit specific features
            features = []
            if u.get("has_pooja_room"):   features.append("Pooja")
            if u.get("has_study_room"):    features.append("Study")
            if u.get("has_terrace"):       features.append("Terrace")
            if u.get("has_garden"):        features.append("Garden")
            if u.get("has_servant_room"):  features.append("Servant")
            if features:
                unit_line += f"| Features: {', '.join(features)}"
                
            lines.append(unit_line)

        lines.append("")   # blank line between projects
    return "\n".join(lines)


def generate_answer(
    user_query: str,
    results: list[dict],
    conversation_history: list[dict] | None = None,
) -> str:
    """
    Generate a natural language recommendation given search results.

    Args:
        user_query: The user's original message.
        results: List of matched unit+project dicts from search.py.
        conversation_history: Previous chat turns for context.

    Returns:
        A friendly, informative answer string.
    """
    client = Groq(api_key=settings.GROQ_API_KEY)

    formatted = _format_results_for_llm(results)
    user_content = (
        f"User's request: {user_query}\n\n"
        f"Matching projects from database:\n{formatted}"
    )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if conversation_history:
        messages.extend(conversation_history[-4:])
    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=0.6,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
        logger.debug(f"Generated answer ({len(answer)} chars)")
        return answer
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return (
            "I found some matching projects but couldn't generate a full response right now. "
            "Please try again in a moment."
        )
