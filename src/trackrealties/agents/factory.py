"""Factory utilities for creating agent instances and loading fine-tuned models."""

from pathlib import Path
from typing import Optional, Type

from .agent import AgentAgent
from .base import AgentDependencies, BaseAgent
from .buyer import BuyerAgent
from .developer import DeveloperAgent
from .investor import InvestorAgent
from .roles import UserRole

_ROLE_TO_CLASS = {
    UserRole.INVESTOR: InvestorAgent,
    UserRole.DEVELOPER: DeveloperAgent,
    UserRole.BUYER: BuyerAgent,
    UserRole.AGENT: AgentAgent,
}


def get_agent_class(role: UserRole) -> Type[BaseAgent]:
    """Return the agent class for a user role."""
    if role in _ROLE_TO_CLASS:
        return _ROLE_TO_CLASS[role]
    raise NotImplementedError(f"No agent implemented for role: {role.value}")


def _get_model_path(role: UserRole) -> Optional[str]:
    """Return path to the fine-tuned model for the given role if available."""
    model_dir = Path(f"models/{role.value}_llm")
    return model_dir.as_posix() if model_dir.exists() else None


def create_agent(role: UserRole, deps: Optional[AgentDependencies] = None) -> BaseAgent:
    """Instantiate an agent with its fine-tuned model path if it exists."""
    agent_class = get_agent_class(role)
    model_path = _get_model_path(role)
    return agent_class(deps=deps, model_path=model_path)
