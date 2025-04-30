from app.tool.base import BaseTool
import os
from app.tool.update_infinigen import update_infinigen

_TERMINATE_DESCRIPTION = """Terminate the scene synthesis when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work."""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "The finish status of the interaction.",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    def execute(self, status: str) -> str:
        """Finish the current execution"""
        user_demand = os.getenv("UserDemand")
        iter = int(os.getenv("iter"))
        roomtype = os.getenv("roomtype")
        action = self.name
        try:
            success = update_infinigen("finalize_scene", iter, "")
            assert success
            return f"Successfully terminate."
        except Exception as e:
            return f"Error terminate"
        return f"The scene synthesis has been completed with status: {status}"
