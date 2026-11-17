from app.generation.prompts import INTENT_SYSTEM_INSTRUCTION, intent_prompt


def test_intent_prompt_delegates_casual_or_notes_decision_to_model() -> None:
    prompt = intent_prompt("What are the important concepts?", "")

    assert "USER MESSAGE:" in prompt
    assert "CASUAL" in INTENT_SYSTEM_INSTRUCTION
    assert "NOTES" in INTENT_SYSTEM_INSTRUCTION
