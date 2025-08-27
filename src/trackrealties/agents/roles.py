from enum import Enum


class UserRole(str, Enum):
    INVESTOR = "investor"
    DEVELOPER = "developer"
    BUYER = "buyer"
    AGENT = "agent"
    GENERAL = "general"
