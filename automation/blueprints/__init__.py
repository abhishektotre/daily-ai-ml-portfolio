"""
Daily AI/ML/Data Science Blueprint Catalog
Provides structured, production-grade blueprints across:
- Data Science
- Data Analytics
- Machine Learning
- Deep Learning
- Natural Language Processing (NLP)
- Artificial Intelligence (AI & Agents)
"""

from .data_science import DATA_SCIENCE_PROJECTS
from .data_analytics import DATA_ANALYTICS_PROJECTS
from .machine_learning import MACHINE_LEARNING_PROJECTS
from .deep_learning import DEEP_LEARNING_PROJECTS
from .nlp import NLP_PROJECTS
from .ai_agents import AI_AGENTS_PROJECTS

DOMAIN_REGISTRY = {
    "data_science": {
        "name": "Data Science",
        "badge_color": "blue",
        "projects": DATA_SCIENCE_PROJECTS,
    },
    "data_analytics": {
        "name": "Data Analytics",
        "badge_color": "green",
        "projects": DATA_ANALYTICS_PROJECTS,
    },
    "machine_learning": {
        "name": "Machine Learning",
        "badge_color": "orange",
        "projects": MACHINE_LEARNING_PROJECTS,
    },
    "deep_learning": {
        "name": "Deep Learning",
        "badge_color": "red",
        "projects": DEEP_LEARNING_PROJECTS,
    },
    "nlp": {
        "name": "Natural Language Processing",
        "badge_color": "purple",
        "projects": NLP_PROJECTS,
    },
    "ai_agents": {
        "name": "Artificial Intelligence",
        "badge_color": "yellow",
        "projects": AI_AGENTS_PROJECTS,
    },
}

DOMAIN_ORDER = [
    "data_science",
    "data_analytics",
    "machine_learning",
    "deep_learning",
    "nlp",
    "ai_agents",
]
