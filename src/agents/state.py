from typing import TypedDict, Dict, Any, Optional

class AgentState(TypedDict):
    user_inputs: Dict[str, Any]      
    route: str                        
    model_prediction: Optional[Dict[str, Any]] 
    rule_findings: Optional[Dict[str, Any]]    
    draft: Optional[Dict[str, Any]]   
    review_status: str               
    retry_count: int                 
    final_report: Optional[Dict[str, Any]] 