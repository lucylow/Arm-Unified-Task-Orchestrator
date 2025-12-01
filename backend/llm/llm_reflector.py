class LLMReflector:
    def __init__(self, llm_agent):
        self.llm_agent = llm_agent

    def reflect_and_replan(self, failed_plan, error_msg):
        prompt = f"""The previous plan failed with the following error: {error_msg}
        Plan steps: {failed_plan}
        Please suggest a revised action plan that avoids the failure."""
        corrected_plan = self.llm_agent.generate(prompt)
        return corrected_plan
