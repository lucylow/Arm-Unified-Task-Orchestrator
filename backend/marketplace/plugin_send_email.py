from ..plugin_base import WorkflowPlugin


class SendEmailPlugin(WorkflowPlugin):
    metadata = {"name": "send_email", "author": "example", "version": "0.1", "description": "Sends a mock email"}

    def run(self, input_data):
        to = input_data.get("to")
        subject = input_data.get("subject")
        if not to:
            return {"success": False, "msg": "missing 'to'"}
        # In production, integrate with an email provider adapter
        return {"success": True, "msg": f"Email sent to {to} (mock)", "to": to, "subject": subject}
