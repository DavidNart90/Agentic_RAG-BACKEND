"""
Context Management for TrackRealties AI Agents

This module provides conversation context management, session handling,
and user preference tracking for maintaining state across agent interactions.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.trackrealties.models.conversation import \
    ConversationMessage as Message
from src.trackrealties.models.enums import MessageRole

from ..models.agent import ValidationResult
from ..models.search import SearchResponse

logger = logging.getLogger(__name__)


class ConversationContext(BaseModel):
    """
    Context for a single conversation session.

    This model tracks the state of an ongoing conversation including
    message history, user preferences, search results, and validation outcomes.
    """

    session_id: str
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    messages: List[Message] = Field(default_factory=list)
    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    search_history: List[SearchResponse] = Field(default_factory=list)
    validation_results: List[ValidationResult] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True

    def add_message(self, message: Message):
        """Add a message to the conversation history."""
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc)

    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Get the most recent messages from the conversation."""
        return self.messages[-limit:] if self.messages else []

    def update_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences for this session."""
        self.user_preferences.update(preferences)
        self.updated_at = datetime.now(timezone.utc)

    def add_search_result(self, search_result: SearchResponse):
        """Add a search result to the history."""
        self.search_history.append(search_result)
        self.updated_at = datetime.now(timezone.utc)

    def add_validation_result(self, validation_result: ValidationResult):
        """Add a validation result to the history."""
        self.validation_results.append(validation_result)
        self.updated_at = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if this context has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def extend_expiration(self, hours: int = 24):
        """Extend the expiration time for this context."""
        self.expires_at = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class UserProfile:
    """
    User profile containing preferences and historical data.

    This class tracks user-specific information that persists across
    multiple conversation sessions.
    """

    user_id: str
    role: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)
    search_patterns: List[str] = field(default_factory=list)
    favorite_locations: List[str] = field(default_factory=list)
    budget_ranges: Dict[str, Any] = field(default_factory=dict)
    interaction_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_preferences(self, new_preferences: Dict[str, Any]):
        """Update user preferences."""
        self.preferences.update(new_preferences)
        self.updated_at = datetime.now(timezone.utc)

    def add_search_pattern(self, pattern: str):
        """Add a search pattern to the user's history."""
        if pattern not in self.search_patterns:
            self.search_patterns.append(pattern)
            # Keep only the most recent 50 patterns
            if len(self.search_patterns) > 50:
                self.search_patterns = self.search_patterns[-50:]
        self.updated_at = datetime.now(timezone.utc)

    def add_interaction(self, interaction: Dict[str, Any]):
        """Add an interaction to the user's history."""
        interaction["timestamp"] = datetime.now(timezone.utc).isoformat()
        self.interaction_history.append(interaction)
        # Keep only the most recent 100 interactions
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
        self.updated_at = datetime.now(timezone.utc)


class ContextManager:
    """
    Manager for conversation contexts and user profiles.

    This class provides centralized management of conversation state,
    user preferences, and session handling across the application.
    """

    def __init__(self, default_expiration_hours: int = 24):
        self.contexts: Dict[str, ConversationContext] = {}
        self.user_profiles: Dict[str, UserProfile] = {}
        self.default_expiration_hours = default_expiration_hours
        logger.info("Initialized ContextManager")

    def get_or_create_context(
        self, session_id: str, user_id: Optional[str] = None, user_role: Optional[str] = None
    ) -> ConversationContext:
        """
        Get existing context or create a new one.

        Args:
            session_id: Session identifier
            user_id: User identifier
            user_role: User's role

        Returns:
            Conversation context for the session
        """
        # Check if context exists and is not expired
        if session_id in self.contexts:
            context = self.contexts[session_id]
            if not context.is_expired():
                # Update user info if provided
                if user_id and not context.user_id:
                    context.user_id = user_id
                if user_role and not context.user_role:
                    context.user_role = user_role
                return context
            else:
                # Remove expired context
                del self.contexts[session_id]

        # Create new context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            user_role=user_role,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=self.default_expiration_hours),
        )

        # Load user preferences if user_id is provided
        if user_id:
            user_profile = self.get_or_create_user_profile(user_id, user_role)
            context.user_preferences = user_profile.preferences.copy()

        self.contexts[session_id] = context
        logger.info(f"Created new context for session {session_id}")
        return context

    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """
        Get existing context by session ID.

        Args:
            session_id: Session identifier

        Returns:
            Conversation context if found and not expired
        """
        if session_id in self.contexts:
            context = self.contexts[session_id]
            if not context.is_expired():
                return context
            else:
                del self.contexts[session_id]
        return None

    def update_context(self, session_id: str, context: ConversationContext):
        """
        Update an existing context.

        Args:
            session_id: Session identifier
            context: Updated context
        """
        context.updated_at = datetime.now(timezone.utc)
        self.contexts[session_id] = context

        # Update user profile if user_id is available
        if context.user_id:
            self.update_user_profile_from_context(context)

    def clear_context(self, session_id: str):
        """
        Clear context for a session.

        Args:
            session_id: Session identifier
        """
        if session_id in self.contexts:
            del self.contexts[session_id]
            logger.info(f"Cleared context for session {session_id}")

    def get_or_create_user_profile(self, user_id: str, role: Optional[str] = None) -> UserProfile:
        """
        Get existing user profile or create a new one.

        Args:
            user_id: User identifier
            role: User's role

        Returns:
            User profile
        """
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            # Update role if provided and not set
            if role and not profile.role:
                profile.role = role
                profile.updated_at = datetime.now(timezone.utc)
            return profile

        # Create new profile
        profile = UserProfile(user_id=user_id, role=role)
        self.user_profiles[user_id] = profile
        logger.info(f"Created new user profile for {user_id}")
        return profile

    def update_user_profile_from_context(self, context: ConversationContext):
        """
        Update user profile based on conversation context.

        Args:
            context: Conversation context to extract data from
        """
        if not context.user_id:
            return

        profile = self.get_or_create_user_profile(context.user_id, context.user_role)

        # Extract search patterns from messages
        for message in context.messages:
            if message.role == MessageRole.USER:
                profile.add_search_pattern(message.content[:100])  # First 100 chars

        # Update preferences from context
        if context.user_preferences:
            profile.update_preferences(context.user_preferences)

        # Add interaction summary
        if context.messages:
            interaction = {
                "session_id": context.session_id,
                "message_count": len(context.messages),
                "tools_used": context.tools_used,
                "validation_count": len(context.validation_results),
                "duration_minutes": (context.updated_at - context.created_at).total_seconds() / 60,
            }
            profile.add_interaction(interaction)

    def get_user_context_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of user's context across all sessions.

        Args:
            user_id: User identifier

        Returns:
            Summary of user's interaction patterns and preferences
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}

        # Find active sessions for this user
        active_sessions = [
            context for context in self.contexts.values() if context.user_id == user_id and not context.is_expired()
        ]

        return {
            "user_id": user_id,
            "role": profile.role,
            "preferences": profile.preferences,
            "search_patterns": profile.search_patterns[-10:],  # Recent patterns
            "favorite_locations": profile.favorite_locations,
            "active_sessions": len(active_sessions),
            "total_interactions": len(profile.interaction_history),
            "last_interaction": profile.updated_at.isoformat() if profile.interaction_history else None,
        }

    def cleanup_expired_contexts(self):
        """Remove expired contexts from memory."""
        expired_sessions = [session_id for session_id, context in self.contexts.items() if context.is_expired()]

        for session_id in expired_sessions:
            del self.contexts[session_id]

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired contexts")

    def get_context_stats(self) -> Dict[str, Any]:
        """
        Get statistics about current contexts.

        Returns:
            Statistics about active contexts and users
        """
        active_contexts = len([c for c in self.contexts.values() if not c.is_expired()])
        total_users = len(self.user_profiles)

        role_distribution = {}
        for profile in self.user_profiles.values():
            role = profile.role or "unknown"
            role_distribution[role] = role_distribution.get(role, 0) + 1

        return {
            "active_contexts": active_contexts,
            "total_contexts": len(self.contexts),
            "total_users": total_users,
            "role_distribution": role_distribution,
            "average_messages_per_context": sum(len(c.messages) for c in self.contexts.values()) / max(len(self.contexts), 1),
        }

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """
        Export all data for a specific user.

        Args:
            user_id: User identifier

        Returns:
            Complete user data export
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            return {}

        # Get all contexts for this user
        user_contexts = [
            {
                "session_id": context.session_id,
                "messages": [msg.dict() for msg in context.messages],
                "created_at": context.created_at.isoformat(),
                "updated_at": context.updated_at.isoformat(),
            }
            for context in self.contexts.values()
            if context.user_id == user_id
        ]

        return {
            "user_profile": {
                "user_id": profile.user_id,
                "role": profile.role,
                "preferences": profile.preferences,
                "search_patterns": profile.search_patterns,
                "favorite_locations": profile.favorite_locations,
                "interaction_history": profile.interaction_history,
                "created_at": profile.created_at.isoformat(),
                "updated_at": profile.updated_at.isoformat(),
            },
            "conversations": user_contexts,
        }

    # ------------------------------------------------------------------
    # CRUD-style helper methods for managing conversation sessions
    # ------------------------------------------------------------------

    def create_session(
        self, session_id: str, user_id: Optional[str] = None, user_role: Optional[str] = None
    ) -> ConversationContext:
        """Explicit wrapper to create a new session context."""
        return self.get_or_create_context(session_id, user_id, user_role)

    def read_session(self, session_id: str) -> Optional[ConversationContext]:
        """Retrieve an existing session context if available."""
        return self.get_context(session_id)

    def update_session(
        self, session_id: str, *, message: Optional[Message] = None, preferences: Optional[Dict[str, Any]] = None
    ) -> Optional[ConversationContext]:
        """Update session data such as messages or preferences."""
        context = self.get_context(session_id)
        if not context:
            return None
        if message is not None:
            context.add_message(message)
        if preferences:
            context.update_preferences(preferences)
        self.update_context(session_id, context)
        return context

    def delete_session(self, session_id: str) -> None:
        """Remove a session and its context from memory."""
        self.clear_context(session_id)
