import math

class ToolRegistry:
    def __init__(self):
        self.tools = {
            "calculator": self._calculate,
            "company_db": self._query_company_db,
            "unit_converter": self._convert_units
        }
        
    def _calculate(self, expression: str):
        try:
            # Safe arithmetic eval
            allowed = {"math": math, "abs": abs, "round": round}
            clean_expr = expression.replace("^", "**")
            return str(eval(clean_expr, {"__builtins__": {}}, allowed))
        except Exception as e:
            return f"Error evaluating expression: {e}"
            
    def _query_company_db(self, company_name: str):
        db = {
            "apple": {"revenue_2025": "$391B", "headcount": 161000, "sector": "Consumer Electronics"},
            "microsoft": {"revenue_2025": "$245B", "headcount": 228000, "sector": "Cloud & Software"},
            "nvidia": {"revenue_2025": "$96B", "headcount": 30000, "sector": "Semiconductors & AI"}
        }
        res = db.get(company_name.lower().strip())
        return str(res) if res else "Company not found in registry."
        
    def _convert_units(self, query: str):
        # simple unit converter
        if "km to miles" in query:
            val = float(query.split()[0])
            return f"{val * 0.621371:.2f} miles"
        return "Unsupported unit conversion."
        
    def execute(self, tool_name: str, argument: str):
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not available."
        return self.tools[tool_name](argument)
