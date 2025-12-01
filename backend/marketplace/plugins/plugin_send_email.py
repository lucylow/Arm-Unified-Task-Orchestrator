from marketplace.plugin_base import WorkflowPlugin

class SendEmailPlugin(WorkflowPlugin):
    def run(self, input_data):
        # ... logic using input_data ...
        return {"success": True, "msg": "Email sent!"}
